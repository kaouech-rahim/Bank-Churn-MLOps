"""Module de nettoyage, validation et feature engineering des donnees."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.load_data import load_raw_data


def clean_raw_dataframe(
    df: pd.DataFrame, id_cols: List[str], target_col: str
) -> pd.DataFrame:
    """Supprime les colonnes d'identification technique.

    La cible Exited est deja en 0/1 dans ce dataset : aucune conversion.

    Args:
        df: DataFrame brut d'entree.
        id_cols: Colonnes identifiants a supprimer (RowNumber, CustomerId...).
        target_col: Nom de la colonne cible (conservee telle quelle).

    Returns:
        pd.DataFrame: DataFrame nettoye (copie, l'entree n'est pas modifiee).
    """
    data = df.copy()  # Regle de purete : aucune mutation de la variable d'entree
    to_drop = [c for c in id_cols if c in data.columns and c != target_col]
    return data.drop(columns=to_drop)


def validate_schema(
    df: pd.DataFrame,
    schema_cfg: Dict[str, Any],
    target_col: Optional[str] = None,
) -> None:
    """Verifie le contrat de donnees : colonnes, valeurs autorisees, bornes.

    Toutes les erreurs sont collectees puis levees en une seule fois pour
    faciliter le debogage.

    Args:
        df: DataFrame nettoye a verifier.
        schema_cfg: Section "schema" de la configuration (allowed_values, ranges).
        target_col: Si fourni, la cible doit exister et ne contenir que 0 ou 1.

    Raises:
        ValueError: Si une colonne manque, contient des valeurs manquantes, des
            modalites inconnues, des valeurs hors bornes ou une cible non binaire.
    """
    allowed_values = schema_cfg.get("allowed_values", {})
    ranges = schema_cfg.get("ranges", {})
    required = list(dict.fromkeys(list(allowed_values) + list(ranges)))

    errors: List[str] = []

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"colonnes manquantes : {missing}")

    for col in [c for c in required if c in df.columns]:
        n_null = int(df[col].isnull().sum())
        if n_null:
            errors.append(f"{col} : {n_null} valeur(s) manquante(s)")
            continue
        if col in allowed_values:
            unknown = sorted(set(df[col].unique()) - set(allowed_values[col]))
            if unknown:
                errors.append(f"{col} : valeurs inconnues {unknown}")
        if col in ranges:
            low, high = ranges[col]
            n_out = int(((df[col] < low) | (df[col] > high)).sum())
            if n_out:
                errors.append(f"{col} : {n_out} valeur(s) hors de [{low}, {high}]")

    if target_col is not None:
        if target_col not in df.columns:
            errors.append(f"cible absente : {target_col}")
        elif not set(df[target_col].unique()) <= {0, 1}:
            errors.append(f"{target_col} doit contenir uniquement 0 ou 1")

    if errors:
        raise ValueError("Schema invalide : " + " ; ".join(errors))


def fit_scaler_params(
    df: pd.DataFrame, numerical_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calcule le min et le max de chaque colonne numerique (sur le TRAIN seul).

    Args:
        df: DataFrame d'entrainement.
        numerical_cols: Colonnes numeriques a normaliser.

    Returns:
        Dict[str, Dict[str, float]]: {"Age": {"min": 18.0, "max": 92.0}, ...}
    """
    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in numerical_cols
    }


def encode_features(
    df: pd.DataFrame,
    binary_mapping: Dict[str, Dict[str, int]],
    category_values: Dict[str, List[str]],
    numerical_cols: List[str],
    scaler_params: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Applique l'encodage binaire, One-Hot et la normalisation Min-Max.

    Les parametres de normalisation viennent du jeu d'entrainement : on peut ainsi
    transformer un batch de production exactement comme le train (pas de fuite).
    Les valeurs hors de la plage du train sortent de [0, 1], ce qui est voulu.

    Args:
        df: DataFrame nettoye.
        binary_mapping: {colonne: {modalite: 0 ou 1}} (ex : Gender).
        category_values: {colonne: [modalites]} pour le One-Hot (ex : Geography).
        numerical_cols: Colonnes numeriques a normaliser.
        scaler_params: Sortie de fit_scaler_params.

    Returns:
        pd.DataFrame: Jeu de donnees entierement transforme et encode.

    Raises:
        ValueError: Si une modalite n'est pas prevue dans la configuration.
    """
    data = df.copy()

    # 1. Encodage binaire deterministe
    for col, mapping in binary_mapping.items():
        mapped = data[col].map(mapping)
        if mapped.isnull().any():
            raise ValueError(f"Valeurs inconnues dans la colonne binaire {col}")
        data[col] = mapped.astype(int)

    # 2. Encodage One-Hot avec modalites fixes (memes colonnes train / production)
    for col, categories in category_values.items():
        if not data[col].isin(categories).all():
            raise ValueError(f"Valeurs inconnues dans la colonne {col}")
        data[col] = pd.Categorical(data[col], categories=categories)
    if category_values:
        data = pd.get_dummies(
            data, columns=list(category_values), drop_first=True, dtype=int
        )

    # 3. Normalisation Min-Max avec les parametres du train
    for col in numerical_cols:
        c_min = scaler_params[col]["min"]
        c_max = scaler_params[col]["max"]
        if c_max > c_min:
            data[col] = (data[col] - c_min) / (c_max - c_min)
        else:
            data[col] = 0.0

    return data


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Decoupe le dataset en ensembles train / test avec stratification.

    Args:
        df: DataFrame pret pour la modelisation.
        target_col: Nom de la variable cible binaire.
        test_size: Proportion de l'echantillon de test.
        random_state: Graine aleatoire garantissant la stricte reproductibilite.

    Returns:
        Tuple : X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def run_preprocessing(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Pipeline complet : chargement, nettoyage, validation, split, encodage, ecriture.

    Etapes : reference.csv -> nettoyage -> validation du schema -> split train/test
    -> calcul du scaler sur le train -> encodage -> train.csv, test.csv, scaler.json

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Dict[str, Any]: Les DataFrames "train" et "test" encodes et les
        "scaler_params" appris sur le train.

    Raises:
        FileNotFoundError: Si la configuration ou le fichier de reference manque.
        ValueError: Si les donnees ne respectent pas le contrat de schema.
    """
    cfg = load_config(config_path)
    feats = cfg["features"]
    target = feats["target_column"]

    df_raw = load_raw_data(cfg["data"]["reference_path"])
    df_clean = clean_raw_dataframe(df_raw, feats["id_columns"], target)
    validate_schema(df_clean, cfg["schema"], target)

    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    scaler_params = fit_scaler_params(X_train, feats["numerical_columns"])
    outputs = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        X_enc = encode_features(
            X,
            feats["binary_mapping"],
            feats["category_values"],
            feats["numerical_columns"],
            scaler_params,
        )
        outputs[name] = pd.concat([X_enc, y], axis=1)

    for name, key in [("train", "train_path"), ("test", "test_path")]:
        out_path = Path(cfg["data"][key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        outputs[name].to_csv(out_path, index=False)

    scaler_path = Path(cfg["data"]["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)

    print("=" * 50)
    print("PRETRAITEMENT TERMINE")
    print("=" * 50)
    print(f"* Train : {outputs['train'].shape} | Test : {outputs['test'].shape}")
    print(f"* Colonnes : {list(outputs['train'].columns)}")
    print("=" * 50)
    return {**outputs, "scaler_params": scaler_params}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraitement des donnees churn")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Chemin vers le fichier YAML"
    )
    args = parser.parse_args()
    run_preprocessing(args.config)
