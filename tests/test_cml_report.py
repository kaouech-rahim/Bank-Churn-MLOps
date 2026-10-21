"""Test du rapport CML (execute en CI sur les Pull Requests)."""

import json
from pathlib import Path

from scripts.cml_report import build_report


def test_cml_report_outputs(ml_config_path: Path, tmp_path: Path) -> None:
    """Le rapport produit metrics.json, les 2 graphiques et le Markdown."""
    out = tmp_path / "cml"
    summary = build_report(str(ml_config_path), str(out), n_rows=600)
    for name in ["metrics.json", "confusion_matrix.png", "roc_curve.png", "report.md"]:
        assert (out / name).exists(), name
    with open(out / "metrics.json", encoding="utf-8") as f:
        assert json.load(f)["champion"] == summary["champion"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "synthetique" in report and summary["champion"] in report
