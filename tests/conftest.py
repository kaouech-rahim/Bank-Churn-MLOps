"""Fixtures partagees par tous les tests (donnees synthetiques et configuration)."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.config import load_config
from src.features.preprocess import (
    clean_raw_dataframe,
    encode_features,
    fit_scaler_params,
)


def make_churn_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fabrique un faux dataset churn avec le meme schema que churn.csv."""
    rng = np.random.default_rng(seed)
    age = rng.normal(39, 10, n).round().clip(18, 92).astype(int)
    balance = np.where(rng.random(n) < 0.36, 0.0, rng.normal(120000, 30000, n).clip(1))
    geography = rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25])
    churn_proba = 0.10 + 0.01 * (age - 30).clip(0) + 0.10 * (geography == "Germany")
    return pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": 15600000 + np.arange(n),
            "Surname": "Test",
            "CreditScore": rng.normal(650, 90, n).round().clip(350, 850).astype(int),
            "Geography": geography,
            "Gender": rng.choice(["Female", "Male"], n),
            "Age": age,
            "Tenure": rng.integers(0, 11, n),
            "Balance": balance,
            "NumOfProducts": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01]),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(1000, 200000, n),
            "Exited": (rng.random(n) < churn_proba).astype(int),
        }
    )


@pytest.fixture
def churn_df() -> pd.DataFrame:
    """Dataset synthetique de 2000 lignes."""
    return make_churn_df(2000, seed=0)


@pytest.fixture
def cfg() -> Dict[str, Any]:
    """Vraie configuration du projet (verifie aussi que config.yaml est coherent)."""
    return load_config("configs/config.yaml")


@pytest.fixture
def tmp_config_path(tmp_path: Path, cfg: Dict[str, Any]) -> Path:
    """Config dont tous les chemins pointent vers un dossier temporaire.

    Ecrit aussi un faux churn.csv de 2000 lignes pour tester les pipelines complets.
    """
    local = dict(cfg)
    local["data"] = dict(cfg["data"])
    local["data"]["reference_size"] = 1000
    for key in [
        "raw_path",
        "reference_path",
        "production_pool_path",
        "train_path",
        "test_path",
        "scaler_path",
    ]:
        local["data"][key] = str(tmp_path / Path(cfg["data"][key]).name)

    local["drift"] = dict(cfg["drift"])
    local["drift"]["report_path"] = str(tmp_path / "reports" / "drift_report.html")

    make_churn_df(2000, seed=1).to_csv(local["data"]["raw_path"], index=False)
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f)
    return config_file


# Candidats tres legers : les tests MLflow restent rapides (quelques secondes)
FAST_CANDIDATES = {
    "LogReg-Baseline": {
        "algorithm": "logistic_regression",
        "params": {"max_iter": 500},
    },
    "RandomForest-Tuned": {
        "algorithm": "random_forest",
        "params": {"n_estimators": 20, "max_depth": 4, "random_state": 0},
    },
    "XGBoost-Baseline": {
        "algorithm": "xgboost",
        "params": {"n_estimators": 20, "max_depth": 3, "random_state": 0},
    },
}


def write_ml_config(tmp_path: Path, config_file: Path) -> Path:
    """Complete une config temporaire avec MLflow, modeles, API et Prefect isoles.

    Lance aussi split + preprocess sur le faux churn.csv : train.csv, test.csv et
    scaler_params.json existent donc dans tmp_path (aucune donnee reelle utilisee).

    Args:
        tmp_path: Dossier temporaire du test.
        config_file: Config de base (sortie de la fixture tmp_config_path).

    Returns:
        Path: Chemin de la nouvelle config.
    """
    from src.data.split_data import run_split
    from src.features.preprocess import run_preprocessing

    with open(config_file, "r", encoding="utf-8") as f:
        local = yaml.safe_load(f)

    local["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    local["mlflow"]["artifact_root"] = str(tmp_path / "mlruns")
    local["models"]["candidates"] = FAST_CANDIDATES
    for key in ["model_path", "metadata_path", "scaler_path"]:
        local["models"][key] = str(
            tmp_path / "models" / Path(local["models"][key]).name
        )
    local["models"]["metrics_path"] = str(tmp_path / "reports" / "metrics.json")
    local["prefect"]["retrain_dir"] = str(tmp_path / "retrain")
    local["prefect"]["batch_size"] = 600
    local["prefect"]["retries"] = 0
    local["monitoring"]["drift_report_json"] = str(
        tmp_path / "reports" / "drift_report.json"
    )

    ml_config = tmp_path / "ml_config.yaml"
    with open(ml_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f)

    run_split(str(ml_config))
    run_preprocessing(str(ml_config))
    return ml_config


@pytest.fixture
def ml_config_path(tmp_path: Path, tmp_config_path: Path) -> Path:
    """Config complete et isolee (donnees, MLflow, modeles) dans tmp_path."""
    return write_ml_config(tmp_path, tmp_config_path)


@pytest.fixture
def model_files(tmp_path: Path) -> Dict[str, Path]:
    """Entraine un petit RandomForest et ecrit modele, metadonnees et scaler."""
    cfg = load_config("configs/config.yaml")
    feats = cfg["features"]
    df = clean_raw_dataframe(
        make_churn_df(1000, seed=3), feats["id_columns"], feats["target_column"]
    )
    X, y = df.drop(columns=[feats["target_column"]]), df[feats["target_column"]]
    scaler = fit_scaler_params(X, feats["numerical_columns"])
    X_enc = encode_features(
        X,
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler,
    )
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_enc, y)

    paths = {
        "model": tmp_path / "model.joblib",
        "metadata": tmp_path / "metadata.json",
        "scaler": tmp_path / "scaler_params.json",
    }
    joblib.dump(model, paths["model"])
    metadata = {
        "model_name": "BankChurnModel",
        "version": "7",
        "run_name": "RandomForest-Test",
        "algorithm": "RandomForestClassifier",
        "decision_threshold": 0.5,
        "metrics": {"roc_auc": 0.8, "f1": 0.6},
        "features": {
            "numeric": feats["numerical_columns"],
            "categorical": ["Gender", "Geography"],
            "model_input": list(X_enc.columns),
        },
    }
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    paths["scaler"].write_text(json.dumps(scaler), encoding="utf-8")
    return paths
