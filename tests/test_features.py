"""Tests unitaires du nettoyage, de la validation de schema et de l'encodage."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.config import load_config
from src.features.preprocess import (
    clean_raw_dataframe,
    encode_features,
    fit_scaler_params,
    run_preprocessing,
    split_data,
    validate_schema,
)

ID_COLS = ["RowNumber", "CustomerId", "Surname"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_raw_dataframe(df, ID_COLS, "Exited")


def test_clean_raw_dataframe_drops_ids_and_keeps_target(
    churn_df: pd.DataFrame,
) -> None:
    """Les identifiants disparaissent, la cible reste intacte."""
    cleaned = _clean(churn_df)
    for col in ID_COLS:
        assert col not in cleaned.columns
    assert cleaned["Exited"].tolist() == churn_df["Exited"].tolist()
    assert len(cleaned) == len(churn_df)


def test_clean_raw_dataframe_does_not_mutate_input(churn_df: pd.DataFrame) -> None:
    """Regle de purete : le DataFrame d'entree n'est pas modifie."""
    before = churn_df.copy()
    _clean(churn_df)
    assert churn_df.equals(before)


def test_clean_raw_dataframe_ignores_missing_id_column(
    churn_df: pd.DataFrame,
) -> None:
    """Une colonne identifiant deja absente ne provoque pas d'erreur."""
    cleaned = clean_raw_dataframe(churn_df.drop(columns=["Surname"]), ID_COLS, "Exited")
    assert "Surname" not in cleaned.columns


def test_validate_schema_accepts_valid_data(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Un dataset conforme passe sans erreur."""
    validate_schema(_clean(churn_df), cfg["schema"], "Exited")


def test_validate_schema_missing_column(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Une colonne obligatoire absente leve ValueError."""
    with pytest.raises(ValueError, match="colonnes manquantes"):
        validate_schema(_clean(churn_df).drop(columns=["Age"]), cfg["schema"])


def test_validate_schema_unknown_category(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Une modalite inconnue (ex : pays) leve ValueError."""
    df = _clean(churn_df)
    df.loc[0, "Geography"] = "Tunisia"
    with pytest.raises(ValueError, match="Tunisia"):
        validate_schema(df, cfg["schema"])


@pytest.mark.parametrize(
    "column, bad_value",
    [("Age", 150), ("Age", 5), ("CreditScore", 900), ("Balance", -10.0)],
)
def test_validate_schema_out_of_range(
    churn_df: pd.DataFrame, cfg: Dict[str, Any], column: str, bad_value: float
) -> None:
    """Une valeur hors bornes leve ValueError."""
    df = _clean(churn_df)
    df[column] = df[column].astype(float)
    df.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=column):
        validate_schema(df, cfg["schema"])


def test_validate_schema_missing_values(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Une valeur manquante leve ValueError."""
    df = _clean(churn_df)
    df["Tenure"] = df["Tenure"].astype(float)
    df.loc[3, "Tenure"] = None
    with pytest.raises(ValueError, match="manquante"):
        validate_schema(df, cfg["schema"])


def test_validate_schema_non_binary_target(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Une cible autre que 0/1 leve ValueError."""
    df = _clean(churn_df)
    df.loc[0, "Exited"] = 2
    with pytest.raises(ValueError, match="Exited"):
        validate_schema(df, cfg["schema"], "Exited")


def test_validate_schema_reports_all_errors_at_once(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Toutes les erreurs sont listees dans un seul message."""
    df = _clean(churn_df)
    df.loc[0, "Geography"] = "Tunisia"
    df.loc[1, "Age"] = 150
    with pytest.raises(ValueError) as excinfo:
        validate_schema(df, cfg["schema"])
    assert "Tunisia" in str(excinfo.value)
    assert "Age" in str(excinfo.value)


def test_fit_scaler_params() -> None:
    """Le scaler memorise le min et le max de chaque colonne."""
    df = pd.DataFrame({"a": [1.0, 5.0, 3.0], "b": [10, 20, 30]})
    params = fit_scaler_params(df, ["a", "b"])
    assert params == {
        "a": {"min": 1.0, "max": 5.0},
        "b": {"min": 10.0, "max": 30.0},
    }


def _encode(df: pd.DataFrame, cfg: Dict[str, Any], params=None) -> pd.DataFrame:
    feats = cfg["features"]
    params = params or fit_scaler_params(df, feats["numerical_columns"])
    return encode_features(
        df,
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        params,
    )


def test_encode_features_columns_and_bounds(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Gender binaire, One-Hot sans France, numeriques dans [0, 1]."""
    encoded = _encode(_clean(churn_df), cfg)
    assert "Geography" not in encoded.columns
    assert "Geography_France" not in encoded.columns  # modalite de reference
    assert {"Geography_Germany", "Geography_Spain"} <= set(encoded.columns)
    assert set(encoded["Gender"].unique()) <= {0, 1}
    for col in cfg["features"]["numerical_columns"]:
        assert encoded[col].min() >= 0.0
        assert encoded[col].max() <= 1.0
    assert not encoded.isnull().values.any()


def test_encode_features_gender_mapping(cfg: Dict[str, Any]) -> None:
    """Male = 1 et Female = 0, comme defini dans la configuration."""
    df = _clean(make_df_two_rows())
    encoded = _encode(df, cfg)
    assert encoded["Gender"].tolist() == [1, 0]


def make_df_two_rows() -> pd.DataFrame:
    """Petit DataFrame de 2 clients (Male puis Female)."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [1, 2],
            "Surname": ["a", "b"],
            "CreditScore": [500, 700],
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
            "Age": [30, 50],
            "Tenure": [1, 9],
            "Balance": [0.0, 100000.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [0, 1],
            "EstimatedSalary": [50000.0, 90000.0],
            "Exited": [0, 1],
        }
    )


def test_encode_features_same_columns_for_partial_batch(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Un batch ne contenant qu'un pays produit les MEMES colonnes que le train."""
    clean = _clean(churn_df)
    params = fit_scaler_params(clean, cfg["features"]["numerical_columns"])
    full = _encode(clean, cfg, params)
    only_france = _encode(clean[clean["Geography"] == "France"], cfg, params)
    assert list(only_france.columns) == list(full.columns)
    assert only_france["Geography_Germany"].sum() == 0


def test_encode_features_uses_train_params_not_batch_params(
    cfg: Dict[str, Any],
) -> None:
    """Une valeur hors plage du train sort de [0, 1] (pas de fuite de donnees)."""
    clean = _clean(make_df_two_rows())
    params = fit_scaler_params(clean, cfg["features"]["numerical_columns"])
    new_batch = clean.copy()
    new_batch.loc[0, "Age"] = 90  # le train allait de 30 a 50
    encoded = _encode(new_batch, cfg, params)
    assert encoded.loc[0, "Age"] == pytest.approx((90 - 30) / (50 - 30))
    assert encoded.loc[0, "Age"] > 1.0


def test_encode_features_constant_column_becomes_zero(cfg: Dict[str, Any]) -> None:
    """Une colonne constante (max == min) est mise a 0 sans division par zero."""
    clean = _clean(make_df_two_rows())
    clean["Tenure"] = 5
    encoded = _encode(clean, cfg)
    assert encoded["Tenure"].tolist() == [0.0, 0.0]


def test_encode_features_unknown_values_raise(cfg: Dict[str, Any]) -> None:
    """Un pays ou un genre inconnu leve ValueError au lieu d'etre ignore."""
    clean = _clean(make_df_two_rows())
    bad_country = clean.copy()
    bad_country.loc[0, "Geography"] = "Tunisia"
    with pytest.raises(ValueError, match="Geography"):
        _encode(bad_country, cfg)
    bad_gender = clean.copy()
    bad_gender.loc[0, "Gender"] = "Autre"
    with pytest.raises(ValueError, match="Gender"):
        _encode(bad_gender, cfg)


def test_encode_features_does_not_mutate_input(
    churn_df: pd.DataFrame, cfg: Dict[str, Any]
) -> None:
    """Le DataFrame d'entree n'est pas modifie par l'encodage."""
    clean = _clean(churn_df)
    before = clean.copy()
    _encode(clean, cfg)
    assert clean.equals(before)


def test_split_data_proportions() -> None:
    """Verifie le respect strict du ratio de test et la stratification."""
    df = pd.DataFrame({"feat_1": range(100), "Churn": [0, 1] * 50})
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="Churn", test_size=0.2, random_state=42
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == y_test.mean() == 0.5


def test_run_preprocessing_end_to_end(tmp_config_path: Path) -> None:
    """Pipeline complet : split, fichiers ecrits, scaler appris sur le train seul."""
    run_split_first(tmp_config_path)
    result = run_preprocessing(str(tmp_config_path))
    cfg = load_config(tmp_config_path)

    train = pd.read_csv(cfg["data"]["train_path"])
    test = pd.read_csv(cfg["data"]["test_path"])
    assert len(train) == 800 and len(test) == 200  # 1000 lignes, test_size = 0.2
    assert "Exited" in train.columns
    assert not train.isnull().values.any()
    for col in cfg["features"]["numerical_columns"]:
        assert train[col].between(0, 1).all()  # le train est scale sur lui-meme
    assert Path(cfg["data"]["scaler_path"]).exists()
    assert set(result["scaler_params"]) == set(cfg["features"]["numerical_columns"])


def run_split_first(config_path: Path) -> None:
    """Ecrit churn_reference.csv (entree du pretraitement)."""
    from src.data.split_data import run_split

    run_split(str(config_path))
