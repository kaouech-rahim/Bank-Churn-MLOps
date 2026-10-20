"""Tests unitaires du moteur de drift : PSI, KS, chi2 et detect_drift."""

import copy
import json
import math
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    chi2_test,
    compute_psi,
    compute_psi_categorical,
    detect_drift,
    get_monitored_columns,
    ks_test,
)
from src.monitoring.drift_simulator import DriftSimulator
from tests.conftest import make_churn_df


# ---------------------------------------------------------------- PSI numerique
def test_psi_matches_hand_computed_value() -> None:
    """Le PSI correspond au calcul fait a la main avec la formule officielle."""
    reference = pd.Series(np.arange(1, 101))  # 4 tranches de 25 % : <=25, 26-50, ...
    current = pd.Series([10] * 10 + [30] * 20 + [60] * 30 + [90] * 40)
    p_ref = [0.25, 0.25, 0.25, 0.25]
    p_cur = [0.10, 0.20, 0.30, 0.40]
    expected = sum((c - r) * math.log(c / r) for r, c in zip(p_ref, p_cur))
    assert compute_psi(reference, current, bins=4) == pytest.approx(expected)


def test_psi_identical_distributions_is_zero() -> None:
    """Deux echantillons identiques donnent un PSI nul."""
    values = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert compute_psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shift() -> None:
    """Un decalage de 2 ecarts-types donne un PSI tres au-dessus de 0.2."""
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(2, 1, 5000))
    assert compute_psi(ref, cur) > 0.2


def test_psi_handles_many_zeros_like_balance() -> None:
    """36 % de zeros (comme Balance) : pas d'erreur, PSI proche de 0 si identique."""
    rng = np.random.default_rng(1)
    balance = np.where(rng.random(5000) < 0.36, 0.0, rng.normal(120000, 30000, 5000))
    other = np.where(rng.random(5000) < 0.36, 0.0, rng.normal(120000, 30000, 5000))
    assert compute_psi(pd.Series(balance), pd.Series(other)) < 0.05


def test_psi_constant_reference() -> None:
    """Reference constante : PSI nul si le courant est identique, grand sinon."""
    ref = pd.Series([5] * 100)
    assert compute_psi(ref, pd.Series([5] * 100)) == pytest.approx(0.0)
    assert compute_psi(ref, pd.Series([3] * 100)) > 1.0


def test_psi_empty_series_raises() -> None:
    """Une serie vide leve ValueError."""
    with pytest.raises(ValueError):
        compute_psi(pd.Series([], dtype=float), pd.Series([1.0, 2.0]))


# --------------------------------------------------------------- PSI categoriel
def test_psi_categorical_matches_hand_computed_value() -> None:
    """Meme verification a la main pour une variable categorielle."""
    ref = pd.Series(["A"] * 50 + ["B"] * 50)
    cur = pd.Series(["A"] * 80 + ["B"] * 20)
    expected = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert compute_psi_categorical(ref, cur) == pytest.approx(expected)


def test_psi_categorical_new_category_is_finite_and_large() -> None:
    """Une modalite apparue en production donne un PSI fini et eleve."""
    ref = pd.Series(["A"] * 50 + ["B"] * 50)
    cur = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 20)
    psi = compute_psi_categorical(ref, cur)
    assert math.isfinite(psi)
    assert psi > 0.2


def test_psi_categorical_empty_raises() -> None:
    """Une serie vide leve ValueError."""
    with pytest.raises(ValueError):
        compute_psi_categorical(pd.Series([], dtype=object), pd.Series(["A"]))


# ------------------------------------------------------------------- KS et chi2
def test_ks_test_same_vs_shifted_distribution() -> None:
    """KS : pas significatif sur la meme loi, significatif apres decalage."""
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    same = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(0.5, 1, 2000))
    assert ks_test(ref, same)["significant"] is False
    result = ks_test(ref, shifted)
    assert result["significant"] is True
    assert result["p_value"] < 0.001
    assert 0 < result["statistic"] <= 1


def test_ks_test_respects_alpha() -> None:
    """Un alpha tres petit rend le test moins sensible."""
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(0.15, 1, 500))
    p_value = ks_test(ref, cur)["p_value"]
    assert ks_test(ref, cur, alpha=p_value * 2)["significant"] is True
    assert ks_test(ref, cur, alpha=p_value / 2)["significant"] is False


def test_chi2_test_same_vs_different_proportions() -> None:
    """Chi2 : memes proportions non significatif, proportions differentes oui."""
    ref = pd.Series(["France"] * 500 + ["Germany"] * 250 + ["Spain"] * 250)
    same = pd.Series(["France"] * 250 + ["Germany"] * 125 + ["Spain"] * 125)
    different = pd.Series(["France"] * 150 + ["Germany"] * 300 + ["Spain"] * 50)
    assert chi2_test(ref, same)["significant"] is False
    assert chi2_test(ref, different)["significant"] is True


def test_chi2_test_single_category_and_new_category() -> None:
    """Une seule modalite : p = 1. Une nouvelle modalite est bien detectee."""
    only_a = pd.Series(["A"] * 100)
    assert chi2_test(only_a, only_a)["p_value"] == 1.0
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 300 + ["B"] * 300 + ["C"] * 400)
    assert chi2_test(ref, cur)["significant"] is True


# ----------------------------------------------------------------- detect_drift
@pytest.fixture
def datasets() -> Dict[str, pd.DataFrame]:
    """Reference de 5000 lignes et pool de production independant de 5000 lignes."""
    return {
        "reference": make_churn_df(5000, seed=10),
        "pool": make_churn_df(5000, seed=11),
    }


def test_get_monitored_columns(cfg: Dict[str, Any]) -> None:
    """6 numeriques et 2 categorielles (Gender, Geography) sont surveillees."""
    numerical, categorical = get_monitored_columns(cfg)
    assert numerical == [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    assert sorted(categorical) == ["Gender", "Geography"]


def test_detect_drift_no_drift(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Deux echantillons de la meme loi : aucune alerte critique."""
    report = detect_drift(datasets["reference"], datasets["pool"], cfg)
    assert report["critical"] is False
    assert report["drift_share"] < cfg["drift"]["drift_share_threshold"]
    assert report["n_features"] == 8
    assert report["drifted_features"] == []


def test_detect_drift_with_strong_scenario(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Scenario fort : Age, Balance et Geography derivent, alerte critique."""
    simulator = DriftSimulator.from_config(cfg)
    drifted = simulator.apply(datasets["pool"], "strong", seed=42)
    report = detect_drift(datasets["reference"], drifted, cfg)
    assert report["critical"] is True
    assert sorted(report["drifted_features"]) == ["Age", "Balance", "Geography"]
    assert report["drift_share"] == pytest.approx(3 / 8)
    assert report["n_drifted"] == 3
    assert report["features"]["Age"]["test"] == "ks"
    assert report["features"]["Geography"]["test"] == "chi2"


def test_detect_drift_report_is_json_serializable(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Le rapport ne contient que des types Python simples (contrat avec Prometheus)."""
    report = detect_drift(datasets["reference"], datasets["pool"], cfg)
    assert json.loads(json.dumps(report)) == report


def test_detect_drift_threshold_decides_critical(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Avec 3 variables sur 8 en drift, le seuil 0.5 ne declenche pas l'alerte."""
    drifted = DriftSimulator.from_config(cfg).apply(datasets["pool"], "strong", 42)
    strict_cfg = copy.deepcopy(cfg)
    strict_cfg["drift"]["drift_share_threshold"] = 0.5
    report = detect_drift(datasets["reference"], drifted, strict_cfg)
    assert report["n_drifted"] == 3
    assert report["critical"] is False


def test_detect_drift_critical_when_share_equals_threshold(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Cas limite : drift_share == seuil (3/8 = 0.375) declenche bien l'alerte."""
    drifted = DriftSimulator.from_config(cfg).apply(datasets["pool"], "strong", 42)
    edge_cfg = copy.deepcopy(cfg)
    edge_cfg["drift"]["drift_share_threshold"] = 0.375
    report = detect_drift(datasets["reference"], drifted, edge_cfg)
    assert report["drift_share"] == 0.375
    assert report["critical"] is True


def test_detect_drift_needs_both_significance_and_psi(
    cfg: Dict[str, Any],
) -> None:
    """Ecart significatif mais minuscule (PSI faible) : pas de drift."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0.00, 1, 50000)})
    cur = pd.DataFrame({"x": rng.normal(0.05, 1, 50000)})
    small_cfg = copy.deepcopy(cfg)
    small_cfg["features"]["numerical_columns"] = ["x"]
    small_cfg["features"]["binary_mapping"] = {}
    small_cfg["features"]["category_values"] = {}

    result = detect_drift(ref, cur, small_cfg)["features"]["x"]
    assert result["p_value"] < small_cfg["drift"]["alpha"]  # significatif...
    assert result["psi"] < small_cfg["drift"]["psi_threshold"]  # ...mais negligeable
    assert result["drifted"] is False


def test_detect_drift_ignores_missing_values(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Quelques NaN dans le batch courant ne font pas planter l'analyse."""
    current = datasets["pool"].copy()
    current.loc[:49, "Age"] = np.nan
    report = detect_drift(datasets["reference"], current, cfg)
    assert report["n_features"] == 8


def test_detect_drift_missing_column_raises(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Une colonne surveillee absente leve ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        detect_drift(datasets["reference"], datasets["pool"].drop(columns=["Age"]), cfg)


def test_detect_drift_small_batch_raises(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Un batch plus petit que min_rows est refuse."""
    with pytest.raises(ValueError, match="trop petit"):
        detect_drift(datasets["reference"], datasets["pool"].head(10), cfg)


def test_detect_drift_all_missing_column_raises(
    datasets: Dict[str, pd.DataFrame], cfg: Dict[str, Any]
) -> None:
    """Une colonne entierement vide est refusee avec un message clair."""
    current = datasets["pool"].copy()
    current["Age"] = np.nan
    with pytest.raises(ValueError, match="Age"):
        detect_drift(datasets["reference"], current, cfg)
