"""Tests de l'exposition Prometheus et du deploiement Blue/Green.

Aucun conteneur n'est lance : les commandes docker et les appels HTTP de
switch_traffic.py sont remplaces par des faux (runner / getter injectes).
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from scripts.switch_traffic import (
    copy_bundle,
    init_deployment,
    other_color,
    read_active_color,
    render_upstream,
    switch_traffic,
    wait_for_version,
    write_upstream,
)
from scripts.traffic_during_switch import summarize
from src.api.main import app
from src.monitoring import prometheus_metrics as prom
from tests.test_api import VALID_CUSTOMER

DRIFT_REPORT = {
    "scenario": "strong",
    "drift_share": 0.375,
    "critical": True,
    "drifted_features": ["Age", "Balance", "Geography"],
    "features": {"Age": {"psi": 0.9}, "Balance": {"psi": 0.2}, "Gender": {"psi": 0.01}},
}


# ------------------------------------------------------------ Prometheus
def test_update_drift_metrics_from_valid_report(tmp_path: Path) -> None:
    """Un rapport valide alimente share, critical et le PSI par variable."""
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(DRIFT_REPORT), encoding="utf-8")
    assert prom.update_drift_metrics(path) is True
    assert prom.DRIFT_SHARE._value.get() == pytest.approx(0.375)
    assert prom.DRIFT_CRITICAL._value.get() == 1
    assert prom.DRIFT_REPORT_AVAILABLE._value.get() == 1
    assert prom.DRIFT_FEATURE_PSI.labels(feature="Age")._value.get() == pytest.approx(
        0.9
    )


@pytest.mark.parametrize("content", [None, "{pas du json", '{"critical": true}', "[]"])
def test_update_drift_metrics_never_fails(tmp_path: Path, content: Any) -> None:
    """Rapport absent, illisible ou incomplet : pas d'exception, available = 0."""
    path = tmp_path / "drift.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert prom.update_drift_metrics(path) is False
    assert prom.DRIFT_REPORT_AVAILABLE._value.get() == 0


@pytest.fixture
def metrics_client(
    monkeypatch: pytest.MonkeyPatch, model_files: Dict[str, Path], tmp_path: Path
) -> Iterator[TestClient]:
    """API avec mini-modele et rapport de drift critique."""
    report = tmp_path / "drift_report.json"
    report.write_text(json.dumps(DRIFT_REPORT), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(model_files["model"]))
    monkeypatch.setenv("METADATA_PATH", str(model_files["metadata"]))
    monkeypatch.setenv("SCALER_PATH", str(model_files["scaler"]))
    monkeypatch.setenv("DRIFT_REPORT_PATH", str(report))
    with TestClient(app) as client:
        yield client


def test_metrics_endpoint_exposes_all_metrics(metrics_client: TestClient) -> None:
    """/metrics expose predictions, latence, version du modele et drift."""
    before = prom.PREDICTIONS.labels(endpoint="predict")._value.get()
    assert metrics_client.post("/predict", json=VALID_CUSTOMER).status_code == 200
    metrics_client.post("/predict/batch", json={"customers": [VALID_CUSTOMER] * 3})

    response = metrics_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert prom.PREDICTIONS.labels(endpoint="predict")._value.get() == before + 1
    assert 'churn_predictions_total{endpoint="predict_batch"}' in text
    assert "churn_prediction_latency_seconds_bucket" in text
    assert 'churn_model_info{algorithm="RandomForestClassifier"' in text
    assert "churn_drift_share 0.375" in text
    assert "churn_drift_critical 1.0" in text


# ------------------------------------------------------------ Blue/Green
class FakeResponse:
    """Reponse HTTP minimale pour les tests."""

    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


def make_getter(status: int, version: str) -> Any:
    """Faux httpx.get : /health -> status, /info -> version."""

    def getter(url: str, timeout: float = 3) -> FakeResponse:
        return FakeResponse(status, {"version": version})

    return getter


@pytest.fixture
def deploy_cfg(tmp_path: Path) -> Dict[str, Any]:
    """Config de deploiement isolee + modele v5 a deployer dans tmp_path/new."""
    upstream = tmp_path / "upstream.conf"
    upstream.write_text(render_upstream("blue"), encoding="utf-8")
    new_model = tmp_path / "new"
    new_model.mkdir()
    for name in ["model.joblib", "scaler_params.json"]:
        (new_model / name).write_text("x", encoding="utf-8")
    (new_model / "metadata.json").write_text(json.dumps({"version": "5"}), "utf-8")
    return {
        "deployment": {
            "blue_url": "http://blue",
            "green_url": "http://green",
            "nginx_upstream_file": str(upstream),
            "active_color_file": str(tmp_path / "active.txt"),
            "models_dir": str(tmp_path / "models"),
            "health_timeout_seconds": 0,
        },
        "_new_model": new_model,
    }


def test_colors_and_upstream(tmp_path: Path) -> None:
    """Couleurs, valeur par defaut et reecriture en place de upstream.conf."""
    assert other_color("blue") == "green" and other_color("green") == "blue"
    with pytest.raises(ValueError):
        other_color("red")
    assert read_active_color(tmp_path / "absent.txt") == "blue"
    (tmp_path / "bad.txt").write_text("purple", encoding="utf-8")
    assert read_active_color(tmp_path / "bad.txt") == "blue"

    path = tmp_path / "upstream.conf"
    path.write_text("x" * 500, encoding="utf-8")  # contenu plus long : tronque
    write_upstream(path, "green")
    assert path.read_text(encoding="utf-8") == render_upstream("green")
    assert "server api_green:8000;" in render_upstream("green")


def test_copy_bundle_requires_all_files(tmp_path: Path) -> None:
    """Un modele incomplet n'est jamais deploye."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.joblib").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        copy_bundle(tmp_path / "src", tmp_path / "dst")


def test_wait_for_version() -> None:
    """Pret seulement si /health = 200 ET la version attendue est servie."""
    assert wait_for_version("u", "5", 0, 0, make_getter(200, "5")) is True
    assert wait_for_version("u", None, 0, 0, make_getter(200, "1")) is True
    assert wait_for_version("u", "5", 0, 0, make_getter(200, "4")) is False
    assert wait_for_version("u", "5", 0, 0, make_getter(503, "5")) is False

    def down(url: str, timeout: float = 3) -> Any:
        raise httpx.ConnectError("refuse")

    assert wait_for_version("u", "5", 0, 0, down) is False


def test_switch_traffic_success(deploy_cfg: Dict[str, Any]) -> None:
    """Bascule reussie : modele copie sur green, green redemarre, nginx recharge."""
    commands: List[List[str]] = []
    out = switch_traffic(
        deploy_cfg,
        deploy_cfg["_new_model"],
        runner=commands.append,
        getter=make_getter(200, "5"),
        interval=0,
    )
    dep = deploy_cfg["deployment"]
    assert out == {"switched": True, "active": "green", "previous": "blue"}
    assert commands[0] == ["docker", "compose", "restart", "api_green"]
    assert commands[1][-3:] == ["nginx", "-s", "reload"]
    assert "api_green" in Path(dep["nginx_upstream_file"]).read_text(encoding="utf-8")
    assert read_active_color(Path(dep["active_color_file"])) == "green"
    assert (Path(dep["models_dir"]) / "green" / "metadata.json").exists()


def test_switch_traffic_aborts_when_green_unhealthy(deploy_cfg: Dict[str, Any]) -> None:
    """Green ne repond pas : pas de bascule, blue continue de servir."""
    commands: List[List[str]] = []
    out = switch_traffic(
        deploy_cfg,
        deploy_cfg["_new_model"],
        runner=commands.append,
        getter=make_getter(503, "5"),
        interval=0,
    )
    dep = deploy_cfg["deployment"]
    assert out["switched"] is False and out["active"] == "blue"
    assert commands == [["docker", "compose", "restart", "api_green"]]  # pas de reload
    assert "api_blue" in Path(dep["nginx_upstream_file"]).read_text(encoding="utf-8")


def test_rollback_after_switch(deploy_cfg: Dict[str, Any]) -> None:
    """Rollback : retour sur blue sans copier de modele ni redemarrer."""
    switch_traffic(
        deploy_cfg,
        deploy_cfg["_new_model"],
        runner=lambda c: None,
        getter=make_getter(200, "5"),
        interval=0,
    )
    commands: List[List[str]] = []
    out = switch_traffic(
        deploy_cfg,
        deploy_cfg["_new_model"],
        rollback=True,
        runner=commands.append,
        getter=make_getter(200, "1"),
        interval=0,
    )
    assert out == {"switched": True, "active": "blue", "previous": "green"}
    assert all("restart" not in cmd for cmd in commands)


def test_init_deployment(deploy_cfg: Dict[str, Any]) -> None:
    """--init : meme modele sur les deux couleurs, trafic sur blue."""
    init_deployment(deploy_cfg, deploy_cfg["_new_model"])
    dep = deploy_cfg["deployment"]
    for color in ["blue", "green"]:
        assert (Path(dep["models_dir"]) / color / "model.joblib").exists()
    assert read_active_color(Path(dep["active_color_file"])) == "blue"


def test_summarize_traffic() -> None:
    """Le resume compte les erreurs et l'ordre des versions servies."""
    records = [
        {"status": 200, "version": "1"},
        {"status": 200, "version": "1"},
        {"status": 0, "version": "ConnectError"},
        {"status": 200, "version": "3"},
    ]
    summary = summarize(records)
    assert summary["total"] == 4 and summary["errors"] == 1
    assert summary["versions"] == ["1", "3"]


def test_compose_and_prometheus_files_are_consistent() -> None:
    """docker-compose, Nginx et Prometheus referencent les memes services."""
    with open("docker-compose.yml", encoding="utf-8") as f:
        services = yaml.safe_load(f)["services"]
    assert {"api_blue", "api_green", "nginx", "prometheus"} <= set(services)
    with open("prometheus/prometheus.yml", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)
    targets = [
        t
        for c in prom_cfg["scrape_configs"]
        for s in c["static_configs"]
        for t in s["targets"]
    ]
    assert targets == ["api_blue:8000", "api_green:8000"]
    with open("prometheus/alerts.yml", encoding="utf-8") as f:
        rules = yaml.safe_load(f)["groups"][0]["rules"]
    assert any("churn_drift_critical" in r["expr"] for r in rules)
