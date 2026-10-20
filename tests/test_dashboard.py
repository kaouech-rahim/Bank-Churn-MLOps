"""Tests du dashboard HTML Evidently."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.split_data import run_split
from src.monitoring.dashboard import build_drift_report, run_dashboard
from src.monitoring.drift_simulator import DriftSimulator
from tests.conftest import make_churn_df


def test_build_drift_report_writes_html(tmp_path: Path, cfg: Dict[str, Any]) -> None:
    """Le rapport HTML est genere et mentionne les variables surveillees."""
    reference = make_churn_df(1500, seed=20)
    current = DriftSimulator.from_config(cfg).apply(
        make_churn_df(1500, seed=21), "strong", seed=1
    )
    output = tmp_path / "reports" / "drift.html"

    path = build_drift_report(reference, current, cfg, output)

    assert path == output
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert len(html) > 10000
    assert "Age" in html and "Geography" in html


@pytest.mark.parametrize(
    "scenario, expected_critical", [("none", False), ("strong", True)]
)
def test_run_dashboard_end_to_end(
    tmp_config_path: Path, scenario: str, expected_critical: bool
) -> None:
    """Chaine complete : split, simulation, HTML + JSON, verdict selon le scenario."""
    run_split(str(tmp_config_path))
    report = run_dashboard(str(tmp_config_path), scenario)

    assert report["critical"] is expected_critical
    cfg_report = Path(tmp_config_path).parent / "reports"
    assert (cfg_report / "drift_report.html").exists()
    json_text = (cfg_report / "drift_report.json").read_text(encoding="utf-8")
    assert json_text.endswith("\n")  # fichier propre pour pre-commit
    saved = json.loads(json_text)
    assert saved["scenario"] == scenario
    assert saved["critical"] is expected_critical
