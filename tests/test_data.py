"""Tests unitaires du module d'ingestion et du decoupage reference / production."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.config import load_config
from src.data.load_data import load_raw_data
from src.data.split_data import run_split, split_reference_production


def test_load_raw_data_valid(tmp_path: Path) -> None:
    """Verifie le chargement reussi d'un CSV valide via un fichier temporaire."""
    dummy_csv = tmp_path / "valid.csv"
    dummy_csv.write_text("col_a,col_b\n10,20\n30,40", encoding="utf-8")

    df = load_raw_data(dummy_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col_a", "col_b"]


def test_load_raw_data_file_not_found() -> None:
    """Verifie qu'un chemin inexistant leve formellement FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("chemin_totalement_inexistant_vers_donnees.csv")


def test_load_raw_data_empty_file(tmp_path: Path) -> None:
    """Verifie qu'un fichier vide declenche une erreur explicite."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_raw_data(empty_csv)


def test_load_config_file_not_found() -> None:
    """Verifie qu'une config introuvable leve FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/inexistant.yaml")


def test_split_sizes_and_no_overlap(churn_df: pd.DataFrame) -> None:
    """Les deux jeux ont la bonne taille et aucun client en commun."""
    ref, prod = split_reference_production(churn_df, 1000, "Exited", 42)
    assert len(ref) == 1000
    assert len(prod) == 1000
    assert set(ref["CustomerId"]).isdisjoint(set(prod["CustomerId"]))
    assert len(set(ref["CustomerId"]) | set(prod["CustomerId"])) == len(churn_df)


def test_split_is_stratified(churn_df: pd.DataFrame) -> None:
    """Le taux de churn est quasi identique dans les deux jeux."""
    ref, prod = split_reference_production(churn_df, 1000, "Exited", 42)
    assert abs(ref["Exited"].mean() - prod["Exited"].mean()) < 0.005


def test_split_is_reproducible(churn_df: pd.DataFrame) -> None:
    """Meme graine = meme decoupage ; graine differente = decoupage different."""
    ref_a, _ = split_reference_production(churn_df, 1000, "Exited", 42)
    ref_b, _ = split_reference_production(churn_df, 1000, "Exited", 42)
    ref_c, _ = split_reference_production(churn_df, 1000, "Exited", 7)
    assert ref_a.equals(ref_b)
    assert not ref_a["CustomerId"].equals(ref_c["CustomerId"])


@pytest.mark.parametrize("size", [0, 2000, 5000, -1])
def test_split_invalid_reference_size(churn_df: pd.DataFrame, size: int) -> None:
    """Une taille de reference hors de ]0, n[ leve ValueError."""
    with pytest.raises(ValueError):
        split_reference_production(churn_df, size, "Exited", 42)


def test_split_missing_target(churn_df: pd.DataFrame) -> None:
    """Une colonne cible absente leve ValueError."""
    with pytest.raises(ValueError):
        split_reference_production(churn_df, 1000, "Cible_Inconnue", 42)


def test_run_split_writes_both_files(tmp_config_path: Path) -> None:
    """Le pipeline complet ecrit les 2 CSV demandes par la configuration."""
    ref, prod = run_split(str(tmp_config_path))
    cfg: Dict[str, Any] = load_config(tmp_config_path)
    saved_ref = pd.read_csv(cfg["data"]["reference_path"])
    saved_prod = pd.read_csv(cfg["data"]["production_pool_path"])
    assert len(saved_ref) == len(ref) == 1000
    assert len(saved_prod) == len(prod) == 1000
    assert list(saved_ref.columns) == list(saved_prod.columns)
