"""Tests unitaires du simulateur de drift."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_simulator import DriftSimulator
from tests.conftest import make_churn_df


@pytest.fixture
def production() -> pd.DataFrame:
    """Pool de production synthetique de 5000 lignes."""
    return make_churn_df(5000, seed=5)


def _simulator(scenario_params: Dict[str, Any]) -> DriftSimulator:
    return DriftSimulator({"test": scenario_params})


def test_scenario_none_returns_identical_copy(
    production: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Le scenario "none" renvoie des donnees identiques, mais une copie."""
    result = DriftSimulator.from_config(cfg).apply(production, "none")
    assert result is not production
    assert result.equals(production)


def test_apply_does_not_mutate_input(
    production: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Regle de purete : le DataFrame d'entree n'est jamais modifie."""
    before = production.copy()
    DriftSimulator.from_config(cfg).apply(production, "strong", seed=1)
    assert production.equals(before)


def test_strong_scenario_keeps_size_and_columns(
    production: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Le scenario fort garde le nombre de lignes et les colonnes."""
    result = DriftSimulator.from_config(cfg).apply(production, "strong", seed=1)
    assert len(result) == len(production)
    assert list(result.columns) == list(production.columns)


def test_age_shift_moves_mean_by_the_requested_amount(
    production: pd.DataFrame,
) -> None:
    """Un decalage de +8 ans augmente l'age moyen d'environ 8 ans."""
    result = _simulator({"age_shift": 8}).apply(production, "test", seed=1)
    gap = result["Age"].mean() - production["Age"].mean()
    assert 7.5 < gap < 8.5


def test_age_is_clipped_to_bounds(production: pd.DataFrame) -> None:
    """Les ages restent dans les bornes (18-100 par defaut)."""
    up = _simulator({"age_shift": 200}).apply(production, "test", seed=1)
    down = _simulator({"age_shift": -200}).apply(production, "test", seed=1)
    assert (up["Age"] == 100).all()
    assert (down["Age"] == 18).all()


def test_balance_factor_scales_positive_balances_only(
    production: pd.DataFrame,
) -> None:
    """Les soldes positifs sont multiplies, les comptes a 0 restent a 0."""
    result = _simulator({"balance_factor": 2.0}).apply(production, "test")
    positive = production["Balance"] > 0
    assert (result.loc[~positive, "Balance"] == 0).all()
    np.testing.assert_allclose(
        result.loc[positive, "Balance"], production.loc[positive, "Balance"] * 2.0
    )


def test_country_share_reaches_the_target(production: pd.DataFrame) -> None:
    """La part de l'Allemagne passe d'environ 25 % a environ 50 %."""
    before = (production["Geography"] == "Germany").mean()
    params = {"target_country": "Germany", "target_share": 0.5}
    result = _simulator(params).apply(production, "test", seed=1)
    after = (result["Geography"] == "Germany").mean()
    assert 0.2 < before < 0.3
    assert 0.46 < after < 0.54


def test_same_seed_same_result_different_seed_different_result(
    production: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Reproductibilite : meme graine = meme resultat."""
    sim = DriftSimulator.from_config(cfg)
    a = sim.apply(production, "strong", seed=7)
    b = sim.apply(production, "strong", seed=7)
    c = sim.apply(production, "strong", seed=8)
    assert a.equals(b)
    assert not a.equals(c)


def test_unknown_scenario_raises(production: pd.DataFrame) -> None:
    """Un scenario inconnu leve ValueError."""
    with pytest.raises(ValueError, match="inconnu"):
        _simulator({}).apply(production, "n_existe_pas")


@pytest.mark.parametrize("share", [0.0, 1.0, -0.2, 1.5])
def test_invalid_target_share_raises(production: pd.DataFrame, share: float) -> None:
    """target_share doit etre strictement entre 0 et 1."""
    params = {"target_country": "Germany", "target_share": share}
    with pytest.raises(ValueError):
        _simulator(params).apply(production, "test")


def test_missing_target_country_raises(production: pd.DataFrame) -> None:
    """target_share sans target_country leve ValueError."""
    with pytest.raises(ValueError, match="target_country"):
        _simulator({"target_share": 0.5}).apply(production, "test")


def test_country_absent_from_data_raises(production: pd.DataFrame) -> None:
    """Un pays absent des donnees (part de 0) leve ValueError."""
    params = {"target_country": "Tunisia", "target_share": 0.5}
    with pytest.raises(ValueError):
        _simulator(params).apply(production, "test")


def test_from_config_uses_age_bounds_from_schema(cfg: Dict[str, Any]) -> None:
    """Les bornes d'age du simulateur viennent du contrat de donnees."""
    assert DriftSimulator.from_config(cfg).age_bounds == (18, 100)
