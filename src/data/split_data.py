"""Separation du dataset complet en reference (entrainement) et production (drift)."""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.load_data import load_raw_data


def split_reference_production(
    df: pd.DataFrame,
    reference_size: int,
    target_col: str,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decoupe le dataset en un jeu de reference et un pool de production.

    Le decoupage est stratifie : les deux jeux gardent le meme taux de churn.
    Le jeu de reference sert a l'entrainement, le pool de production sert a
    simuler le flux reel (et le data drift).

    Args:
        df: DataFrame complet (ex : 10 000 lignes).
        reference_size: Nombre de lignes du jeu de reference (ex : 5000).
        target_col: Nom de la colonne cible utilisee pour la stratification.
        random_state: Graine aleatoire garantissant la reproductibilite.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (reference, production_pool).

    Raises:
        ValueError: Si la colonne cible est absente ou si reference_size n'est
            pas strictement compris entre 0 et le nombre de lignes du dataset.
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible absente du dataset : {target_col}")
    if not 0 < reference_size < len(df):
        raise ValueError(
            f"reference_size doit etre entre 1 et {len(df) - 1} "
            f"(recu : {reference_size})"
        )

    reference, production = train_test_split(
        df,
        train_size=reference_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return reference.reset_index(drop=True), production.reset_index(drop=True)


def run_split(
    config_path: str = "configs/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute le decoupage complet : lecture, split et ecriture des 2 CSV.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (reference, production_pool).

    Raises:
        FileNotFoundError: Si la configuration ou le fichier brut est introuvable.
        ValueError: Si le dataset est vide ou si reference_size est invalide.
    """
    cfg = load_config(config_path)
    target = cfg["features"]["target_column"]

    df = load_raw_data(cfg["data"]["raw_path"])
    reference, production = split_reference_production(
        df,
        reference_size=cfg["data"]["reference_size"],
        target_col=target,
        random_state=cfg["data"]["random_state"],
    )

    for df_part, key in [
        (reference, "reference_path"),
        (production, "production_pool_path"),
    ]:
        out_path = Path(cfg["data"][key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_part.to_csv(out_path, index=False)

    print("=" * 50)
    print("DECOUPAGE REFERENCE / PRODUCTION")
    print("=" * 50)
    print(
        f"* Reference  : {len(reference)} lignes "
        f"| churn = {reference[target].mean():.3f}"
    )
    print(
        f"* Production : {len(production)} lignes "
        f"| churn = {production[target].mean():.3f}"
    )
    print("=" * 50)
    return reference, production


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split reference / production")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Chemin vers le fichier YAML"
    )
    args = parser.parse_args()
    run_split(args.config)
