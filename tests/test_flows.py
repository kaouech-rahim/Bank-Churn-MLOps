"""Tests des flows Prefect (entrainement, drift et boucle de retroaction).

prefect_test_harness demarre une base Prefect temporaire : aucun serveur
distant, aucune donnee reelle (config isolee dans tmp_path).
"""

import json
from pathlib import Path
from typing import Iterator

import pytest
import yaml
from mlflow.tracking import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from pipelines.drift_flow import drift_flow
from pipelines.training_flow import training_flow
from src.config import load_config
from src.models.promotion import score_current_model, should_promote
from src.models.registry import get_production_version
from tests.test_models import stages_of, use_weak_candidate


@pytest.fixture(autouse=True, scope="module")
def prefect_harness() -> Iterator[None]:
    """Base Prefect ephemere partagee par les tests du module."""
    with prefect_test_harness():
        yield


# ------------------------------------------------------------ regle de promotion
@pytest.mark.parametrize(
    "new, current, min_gain, expected",
    [
        (0.80, None, 0.0, True),  # aucun modele en service : on promeut
        (0.86, 0.85, 0.0, True),  # meilleur
        (0.85, 0.85, 0.0, True),  # egal : "meilleur ou egal"
        (0.84, 0.85, 0.0, False),  # moins bon : on garde l'ancien
        (0.855, 0.85, 0.01, False),  # gain insuffisant
        (0.86, 0.85, 0.01, True),  # gain suffisant
    ],
)
def test_should_promote(
    new: float, current: float, min_gain: float, expected: bool
) -> None:
    """La promotion suit exactement la regle de la config."""
    assert should_promote(new, current, min_gain) is expected


# ------------------------------------------------------------ flow d'entrainement
def test_training_flow_registers_and_exports(ml_config_path: Path) -> None:
    """Le DAG complet enregistre v1 en Production et exporte le modele."""
    cfg = load_config(ml_config_path)
    result = training_flow(str(ml_config_path))
    assert str(result["version"]) == "1"
    assert Path(cfg["models"]["model_path"]).exists()
    # Le scaler exporte vient du dossier de travail, pas de data/processed
    assert Path(
        cfg["prefect"]["retrain_dir"], "training", "scaler_params.json"
    ).exists()
    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    prod = get_production_version(cfg["mlflow"]["registered_model_name"], client)
    assert prod is not None


# ------------------------------------------------------------ flow de drift
def test_drift_flow_without_drift_does_not_retrain(ml_config_path: Path) -> None:
    """Scenario none : rapport ecrit, pas d'alerte, pas de reentrainement."""
    cfg = load_config(ml_config_path)
    result = drift_flow(str(ml_config_path), scenario="none")
    assert result["critical"] is False
    assert result["retraining"] == {"triggered": False}
    with open(cfg["monitoring"]["drift_report_json"], encoding="utf-8") as f:
        report = json.load(f)
    assert report["scenario"] == "none"
    assert Path(result["reports"]["html"]).exists()


def test_drift_flow_strong_triggers_retraining(ml_config_path: Path) -> None:
    """Scenario strong : drift critique -> reentrainement -> decision coherente."""
    cfg = load_config(ml_config_path)
    training_flow(str(ml_config_path))  # modele v1 en service
    pool_before = Path(cfg["data"]["production_pool_path"]).read_bytes()

    result = drift_flow(str(ml_config_path), scenario="strong")

    assert result["critical"] is True
    assert set(result["drifted_features"]) >= {"Age", "Geography"}
    retraining = result["retraining"]
    assert retraining["triggered"] is True
    assert retraining["current_roc_auc"] is not None
    expected = retraining["reason"] == "better_or_equal"
    assert retraining["promoted"] is expected
    if expected:  # promu : la metrique principale n'a pas baisse
        assert retraining["new_roc_auc"] >= retraining["current_roc_auc"]

    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    prod = get_production_version(cfg["mlflow"]["registered_model_name"], client)
    assert str(prod.version) == ("2" if expected else "1")
    # Le pool de production sur disque n'a pas ete modifie par la simulation
    assert Path(cfg["data"]["production_pool_path"]).read_bytes() == pool_before


def test_drift_flow_keeps_model_when_new_one_is_worse(ml_config_path: Path) -> None:
    """Avec un gain minimal impossible (+1.0), l'ancien modele est conserve."""
    training_flow(str(ml_config_path))
    with open(ml_config_path, encoding="utf-8") as f:
        local = yaml.safe_load(f)
    local["prefect"]["min_improvement"] = 1.0
    with open(ml_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f)

    result = drift_flow(str(ml_config_path), scenario="strong")
    assert result["retraining"]["promoted"] is False
    assert result["retraining"]["version"] is None
    with open(local["models"]["metadata_path"], encoding="utf-8") as f:
        assert str(json.load(f)["version"]) == "1"


def test_score_current_model_without_model(ml_config_path: Path) -> None:
    """Sans modele en service, le score actuel vaut None (promotion directe)."""
    cfg = load_config(ml_config_path)
    import pandas as pd

    assert score_current_model(pd.DataFrame(), cfg) is None


def test_drift_flow_without_labels_skips_retraining(
    ml_config_path: Path, tmp_path: Path
) -> None:
    """Lot de production sans cible : aucun label invente, reentrainement saute."""
    import pandas as pd

    with open(ml_config_path, encoding="utf-8") as f:
        local = yaml.safe_load(f)
    pool = pd.read_csv(local["data"]["production_pool_path"]).drop(columns=["Exited"])
    unlabeled = tmp_path / "pool_sans_labels.csv"
    pool.to_csv(unlabeled, index=False)
    local["data"]["production_pool_path"] = str(unlabeled)
    with open(ml_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f)

    result = drift_flow(str(ml_config_path), scenario="strong")
    assert result["critical"] is True
    assert result["retraining"] == {"triggered": False, "reason": "labels_missing"}


def test_training_flow_does_not_promote_worse_candidate(ml_config_path: Path) -> None:
    """Flow planifie : un candidat moins bon n'est ni promu ni exporte."""
    cfg = load_config(ml_config_path)
    training_flow(str(ml_config_path))
    served = Path(cfg["models"]["model_path"]).read_bytes()
    use_weak_candidate(ml_config_path)

    result = training_flow(str(ml_config_path))

    assert result["decision"]["promote"] is False
    assert result["decision"]["reason"] == "worse"
    assert stages_of(cfg) == {"1": "Production", "2": "Staging"}
    assert Path(cfg["models"]["model_path"]).read_bytes() == served
