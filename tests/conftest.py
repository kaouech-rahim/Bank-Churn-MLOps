"""Fixtures partagees par tous les tests (donnees synthetiques et configuration)."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config


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
