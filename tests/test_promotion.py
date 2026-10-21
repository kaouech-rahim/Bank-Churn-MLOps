"""Tests unitaires de la regle de promotion (patch 01b).

Le modele en service est remplace par des metriques fixes (monkeypatch) : on
teste la decision seule, sans entrainement ni MLflow.
"""

import copy
import math
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from src.config import load_config
from src.models import promotion
from src.models.promotion import decide_promotion, promotion_settings

EVAL = pd.DataFrame(
    {"Exited": [0, 1]}
)  # non utilise : score_current_model est remplace


def config(**rules: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["models"]["promotion"] = {
        "min_improvement": 0.0,
        "max_recall_drop": 0.0,
        "recall_floor": None,
        **rules,
    }
    return cfg


def with_current(monkeypatch: pytest.MonkeyPatch, current: Optional[Dict]) -> None:
    monkeypatch.setattr(promotion, "score_current_model", lambda e, c: current)


CURRENT = {"roc_auc": 0.80, "recall": 0.70, "f1": 0.60}


@pytest.mark.parametrize(
    "candidate, reason, promote",
    [
        ({"roc_auc": 0.80, "recall": 0.70}, "better_or_equal", True),  # egalite
        ({"roc_auc": 0.85, "recall": 0.75}, "better_or_equal", True),
        ({"roc_auc": 0.79, "recall": 0.90}, "worse", False),
        ({"roc_auc": 0.90, "recall": 0.60}, "recall_regression", False),
    ],
)
def test_rules_against_current_model(
    monkeypatch: pytest.MonkeyPatch, candidate: Dict, reason: str, promote: bool
) -> None:
    """Metrique principale ET non-regression du rappel."""
    with_current(monkeypatch, CURRENT)
    d = decide_promotion(candidate, EVAL, config())
    assert (d["reason"], d["promote"]) == (reason, promote)
    assert d["current_score"] == 0.80 and d["current_recall"] == 0.70


def test_recall_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_recall_drop autorise une baisse limitee du rappel."""
    with_current(monkeypatch, CURRENT)
    candidate = {"roc_auc": 0.85, "recall": 0.69}
    assert decide_promotion(candidate, EVAL, config())["promote"] is False
    tolerant = config(max_recall_drop=0.02)
    assert decide_promotion(candidate, EVAL, tolerant)["promote"] is True


def test_recall_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le plancher absolu s'applique avec ou sans modele en service."""
    candidate = {"roc_auc": 0.85, "recall": 0.72}
    with_current(monkeypatch, CURRENT)
    assert decide_promotion(candidate, EVAL, config(recall_floor=0.75))["reason"] == (
        "recall_below_floor"
    )
    with_current(monkeypatch, None)
    assert decide_promotion(candidate, EVAL, config(recall_floor=0.75))["reason"] == (
        "recall_below_floor"
    )
    assert decide_promotion(candidate, EVAL, config())["reason"] == "no_current_model"


@pytest.mark.parametrize(
    "candidate",
    [
        {"roc_auc": math.nan, "recall": 0.7},
        {"roc_auc": math.inf, "recall": 0.7},
        {"roc_auc": 0.9},  # rappel absent
        {"roc_auc": None, "recall": 0.7},
        {"roc_auc": "abc", "recall": 0.7},
    ],
)
def test_invalid_candidate_metrics_block(
    monkeypatch: pytest.MonkeyPatch, candidate: Dict
) -> None:
    """F8 : metrique candidate absente ou non finie -> evaluation_failed."""
    with_current(monkeypatch, CURRENT)
    d = decide_promotion(candidate, EVAL, config())
    assert d["promote"] is False and d["reason"].startswith(
        "evaluation_failed: candidate"
    )


def test_invalid_current_metrics_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metrique du modele en service non finie -> pas de promotion."""
    with_current(monkeypatch, {"roc_auc": math.nan, "recall": 0.7})
    d = decide_promotion({"roc_auc": 0.9, "recall": 0.9}, EVAL, config())
    assert d["promote"] is False and d["reason"].startswith(
        "evaluation_failed: current"
    )


def test_current_model_error_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Erreur de chargement / prediction -> echec ferme."""

    def boom(e: Any, c: Any) -> None:
        raise ValueError("colonnes incompatibles")

    monkeypatch.setattr(promotion, "score_current_model", boom)
    d = decide_promotion({"roc_auc": 0.99, "recall": 0.99}, EVAL, config())
    assert d["promote"] is False
    assert d["reason"] == "evaluation_failed: ValueError: colonnes incompatibles"


def test_configured_metric_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """F2 : la metrique comparee suit models.champion_metric."""
    with_current(monkeypatch, CURRENT)
    cfg = config()
    cfg["models"]["champion_metric"] = "f1"
    candidate = {"roc_auc": 0.99, "recall": 0.80, "f1": 0.55}  # meilleur AUC, pire F1
    d = decide_promotion(candidate, EVAL, cfg)
    assert d["metric"] == "f1" and d["reason"] == "worse"
    assert d["candidate_score"] == 0.55 and d["current_score"] == 0.60


def test_unsupported_metric_is_rejected() -> None:
    """F2 : une metrique non calculable est refusee des la configuration."""
    cfg = config()
    cfg["models"]["champion_metric"] = "pr_auc"
    with pytest.raises(ValueError, match="non supportee"):
        promotion_settings(cfg)


def test_real_config_rules_are_explicit() -> None:
    """La configuration du projet declare les trois regles."""
    rules = load_config("configs/config.yaml")["models"]["promotion"]
    assert set(rules) == {"min_improvement", "max_recall_drop", "recall_floor"}


def test_missing_bundle_with_production_is_not_first_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F9 : pas de bundle mais une Production au Registry -> echec ferme."""
    with_current(monkeypatch, None)
    candidate = {"roc_auc": 0.5, "recall": 0.0}
    d = decide_promotion(candidate, EVAL, config(), production_exists=True)
    assert d["promote"] is False
    assert d["reason"] == "evaluation_failed: current_bundle_missing"
    assert decide_promotion(candidate, EVAL, config())["promote"] is True
