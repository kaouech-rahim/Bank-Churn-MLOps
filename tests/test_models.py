"""Tests de la modelisation : metriques, champion, MLflow Tracking et Registry.

Tout tourne sur des donnees synthetiques avec un backend SQLite temporaire :
aucun serveur MLflow, aucun vrai churn.csv, aucun reseau.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.models.evaluate import (
    compute_metrics,
    predict_with_threshold,
    save_confusion_matrix,
    save_roc_curve,
)
from src.models.predict_production import predict
from src.models.registry import get_production_version, load_production_model
from src.models.train import build_model, run_training, select_champion, split_xy


# ------------------------------------------------------------------ evaluate
def test_predict_with_threshold() -> None:
    """Une proba egale au seuil compte comme churn."""
    proba = np.array([0.1, 0.5, 0.49, 0.9])
    assert predict_with_threshold(proba, 0.5).tolist() == [0, 1, 0, 1]
    assert predict_with_threshold(proba, 0.2).tolist() == [0, 1, 1, 1]


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_predict_with_threshold_rejects_invalid(threshold: float) -> None:
    """Un seuil hors de ]0, 1[ est refuse."""
    with pytest.raises(ValueError, match="seuil"):
        predict_with_threshold(np.array([0.3]), threshold)


def test_compute_metrics_hand_example() -> None:
    """Metriques verifiees a la main : TP=2, FP=1, FN=1, TN=2."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])
    m = compute_metrics(y_true, y_proba, 0.5)
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_compute_metrics_single_class_raises() -> None:
    """Sans les deux classes, le ROC-AUC est indefini : erreur explicite."""
    with pytest.raises(ValueError, match="deux classes"):
        compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_plots_are_written(tmp_path: Path) -> None:
    """La matrice de confusion et la courbe ROC sont ecrites en PNG."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.4, 0.6, 0.3, 0.1])
    cm = save_confusion_matrix(
        y_true, (y_proba > 0.5).astype(int), "t", tmp_path / "a/cm.png"
    )
    roc = save_roc_curve(y_true, y_proba, "t", tmp_path / "b/roc.png")
    assert cm.exists() and cm.stat().st_size > 0
    assert roc.exists() and roc.stat().st_size > 0


# ------------------------------------------------------------------ train
@pytest.mark.parametrize(
    "algorithm, expected",
    [
        ("logistic_regression", "LogisticRegression"),
        ("random_forest", "RandomForestClassifier"),
        ("xgboost", "XGBClassifier"),
    ],
)
def test_build_model_known_algorithms(algorithm: str, expected: str) -> None:
    """Chaque algorithme de la config donne la bonne classe."""
    assert type(build_model(algorithm, {})).__name__ == expected


def test_build_model_unknown_raises() -> None:
    """Un algorithme inconnu est refuse."""
    with pytest.raises(ValueError, match="Algorithme inconnu"):
        build_model("svm", {})


def test_split_xy_missing_target_raises() -> None:
    """La cible doit etre presente."""
    with pytest.raises(ValueError, match="cible"):
        split_xy(pd.DataFrame({"a": [1]}), "Exited")


def _result(name: str, auc: float) -> dict:
    return {"run_name": name, "metrics": {"roc_auc": auc, "f1": 0.5}}


def test_select_champion_takes_best_metric() -> None:
    """Le champion est le run au meilleur ROC-AUC."""
    results = [_result("A", 0.80), _result("B", 0.86), _result("C", 0.84)]
    assert select_champion(results, "roc_auc")["run_name"] == "B"


def test_select_champion_tie_keeps_first() -> None:
    """En cas d'egalite, le premier run (ordre de la config) est garde."""
    results = [_result("A", 0.85), _result("B", 0.85)]
    assert select_champion(results, "roc_auc")["run_name"] == "A"


def test_select_champion_errors() -> None:
    """Liste vide ou metrique absente : erreur explicite."""
    with pytest.raises(ValueError, match="Aucun run"):
        select_champion([], "roc_auc")
    with pytest.raises(ValueError, match="absente"):
        select_champion([_result("A", 0.8)], "pr_auc")


def test_all_configured_candidates_are_buildable() -> None:
    """Les 6 candidats de la vraie config s'instancient (hyperparametres valides)."""
    cfg = load_config("configs/config.yaml")
    assert len(cfg["models"]["candidates"]) == 6
    for spec in cfg["models"]["candidates"].values():
        build_model(spec["algorithm"], spec["params"])


# ------------------------------------------------------------------ MLflow
def test_run_training_logs_runs_and_promotes(ml_config_path: Path) -> None:
    """Pipeline complet : runs logues, champion en Production, export pour l'API."""
    cfg = load_config(ml_config_path)
    out = run_training(str(ml_config_path))

    # 1. Un run par candidat, avec params, 5 metriques et 2 graphiques
    assert len(out["results"]) == len(cfg["models"]["candidates"])
    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    for res in out["results"]:
        run = client.get_run(res["run_id"])
        assert run.data.params["algorithm"] == res["algorithm"]
        assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(
            run.data.metrics
        )
        plots = [
            a.path for a in client.list_artifacts(res["run_id"], "diagnostic_plots")
        ]
        assert len(plots) == 2

    # 2. Le champion est bien le meilleur ROC-AUC
    best = max(r["metrics"]["roc_auc"] for r in out["results"])
    assert out["champion"]["metrics"]["roc_auc"] == best

    # 3. Registry : version 1 en Production avec description QA et alias
    name = cfg["mlflow"]["registered_model_name"]
    prod = get_production_version(name, client)
    assert prod is not None and str(prod.version) == "1"
    assert "QA" in prod.description
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"

    # 4. Export : modele, metadonnees et scaler presents et coherents
    model = joblib.load(cfg["models"]["model_path"])
    with open(cfg["models"]["metadata_path"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["version"] == "1"
    assert meta["run_name"] == out["champion"]["run_name"]
    assert len(meta["features"]["model_input"]) == 11
    assert Path(cfg["models"]["scaler_path"]).exists()
    assert hasattr(model, "predict_proba")
    with open(cfg["models"]["metrics_path"], encoding="utf-8") as f:
        assert len(json.load(f)["runs"]) == len(cfg["models"]["candidates"])


def test_second_training_archives_previous_version(ml_config_path: Path) -> None:
    """Un nouvel entrainement promeut v2 et archive automatiquement v1."""
    cfg = load_config(ml_config_path)
    run_training(str(ml_config_path))
    run_training(str(ml_config_path))
    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model_name"]
    stages = {
        str(v.version): v.current_stage
        for v in client.search_model_versions(f"name='{name}'")
    }
    assert stages == {"1": "Archived", "2": "Production"}


def test_training_without_registry(ml_config_path: Path) -> None:
    """--no-register : aucun modele enregistre, version 'local' dans les metadonnees."""
    cfg = load_config(ml_config_path)
    out = run_training(str(ml_config_path), register=False)
    assert out["metadata"]["version"] == "local"
    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    assert (
        get_production_version(cfg["mlflow"]["registered_model_name"], client) is None
    )


def test_production_model_inference(ml_config_path: Path) -> None:
    """models:/BankChurnModel/Production se charge et predit des 0/1."""
    import mlflow

    cfg = load_config(ml_config_path)
    run_training(str(ml_config_path))
    preds = predict(str(ml_config_path), n_customers=5)
    assert len(preds) == 5 and set(preds) <= {0, 1}

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    model = load_production_model(cfg["mlflow"]["registered_model_name"])
    assert hasattr(model, "predict")
