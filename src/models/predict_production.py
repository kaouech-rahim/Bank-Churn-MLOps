"""Client d'inference : charge models:/BankChurnModel/Production (TP3, etape 7).

Le modele est charge par son URI logique : aucun chemin de fichier en dur.
Les clients a evaluer sont lus dans le pool de production (lecture seule) puis
encodes avec le MEME pretraitement que l'entrainement (encode_features).
"""

import argparse
import json
from typing import Any, Dict, List

import mlflow
import pandas as pd

from src.config import load_config
from src.data.load_data import load_raw_data
from src.features.preprocess import clean_raw_dataframe, encode_features
from src.models.registry import load_production_model


def prepare_customers(
    raw_df: pd.DataFrame,
    cfg: Dict[str, Any],
    scaler_params: Dict[str, Dict[str, float]],
    feature_order: List[str],
) -> pd.DataFrame:
    """Transforme des clients bruts en matrice d'entree du modele.

    Args:
        raw_df: Lignes brutes (format churn.csv, cible eventuellement presente).
        cfg: Configuration complete.
        scaler_params: Parametres Min-Max appris sur le train.
        feature_order: Ordre des colonnes attendu par le modele.

    Returns:
        pd.DataFrame: Donnees encodees, colonnes dans l'ordre du modele.
    """
    feats = cfg["features"]
    clean = clean_raw_dataframe(raw_df, feats["id_columns"], feats["target_column"])
    clean = clean.drop(columns=[feats["target_column"]], errors="ignore")
    encoded = encode_features(
        clean,
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler_params,
    )
    return encoded[feature_order]


def predict(
    config_path: str = "configs/config.yaml", n_customers: int = 3
) -> List[int]:
    """Predit le churn de quelques clients avec le modele en Production.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        n_customers: Nombre de clients du pool de production a evaluer.

    Returns:
        List[int]: Predictions (0 = reste, 1 = part).
    """
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    model_name = cfg["mlflow"]["registered_model_name"]
    print(f"[CLIENT] Chargement de models:/{model_name}/Production ...")
    model = load_production_model(model_name)

    with open(cfg["models"]["scaler_path"], "r", encoding="utf-8") as f:
        scaler_params = json.load(f)
    with open(cfg["models"]["metadata_path"], "r", encoding="utf-8") as f:
        feature_order = json.load(f)["features"]["model_input"]

    sample = load_raw_data(cfg["data"]["production_pool_path"]).head(n_customers)
    X = prepare_customers(sample, cfg, scaler_params, feature_order)
    preds = [int(p) for p in model.predict(X)]

    print("=" * 55)
    print(" PREDICTIONS DU MODELE DE PRODUCTION")
    print("=" * 55)
    for idx, p in enumerate(preds, start=1):
        decision = "DEPART PROBABLE (alerte churn)" if p == 1 else "Client fidele"
        print(f" -> Client #{idx} : {decision} (prediction = {p})")
    print("=" * 55)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference depuis le Model Registry")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Chemin vers le fichier YAML"
    )
    parser.add_argument("--n", type=int, default=3, help="Nombre de clients")
    args = parser.parse_args()
    predict(args.config, args.n)
