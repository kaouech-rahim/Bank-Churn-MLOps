"""Flow Prefect d'entrainement : ingestion -> validation -> pretraitement
-> entrainement -> evaluation -> comparaison au modele en service -> Registry
-> export (seulement si le champion est promu).

Chaque etape est une @task atomique et testable. Les taches d'entree/sortie
(lecture des donnees, Registry) ont des retries (nombre et delai dans la config).
Le flow lit churn_reference.csv sans jamais le modifier : les fichiers de travail
sont ecrits dans prefect.retrain_dir/training (data/retrain/training).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from src.config import load_config
from src.data.load_data import load_raw_data
from src.features.preprocess import (
    clean_raw_dataframe,
    encode_features,
    fit_scaler_params,
    split_data,
    validate_schema,
)
from src.models.promotion import decide_promotion
from src.models.registry import get_production_version, register_champion
from src.models.train import export_champion, select_champion, train_candidates


def retry_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Options de retry lues dans la config (section prefect)."""
    return {
        "retries": cfg["prefect"]["retries"],
        "retry_delay_seconds": cfg["prefect"]["retry_delay_seconds"],
    }


@task(name="ingestion", cache_policy=NO_CACHE)
def ingest_data(path: str) -> pd.DataFrame:
    """Lit un CSV de donnees brutes (lecture seule)."""
    return load_raw_data(path)


@task(name="validation-schema", cache_policy=NO_CACHE)
def validate_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Supprime les identifiants et verifie le contrat de donnees."""
    feats = cfg["features"]
    clean = clean_raw_dataframe(df, feats["id_columns"], feats["target_column"])
    validate_schema(clean, cfg["schema"], feats["target_column"])
    return clean


def encode_train_test(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    cfg: Dict[str, Any],
    work_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Apprend le scaler sur le train, encode train et test, ecrit le scaler.

    Args:
        train_clean: Donnees d'entrainement nettoyees (cible incluse).
        test_clean: Donnees de test nettoyees (cible incluse).
        cfg: Configuration complete.
        work_dir: Dossier de travail ou ecrire scaler_params.json.

    Returns:
        Tuple: (train encode, test encode, chemin du scaler).
    """
    feats = cfg["features"]
    target = feats["target_column"]
    X_train = train_clean.drop(columns=[target])
    X_test = test_clean.drop(columns=[target])
    scaler = fit_scaler_params(X_train, feats["numerical_columns"])
    args = (
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler,
    )
    train_df = encode_features(X_train, *args).assign(**{target: train_clean[target]})
    test_df = encode_features(X_test, *args).assign(**{target: test_clean[target]})

    work_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = work_dir / "scaler_params.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)
    return train_df, test_df, scaler_path


def preprocess_frames(
    clean: pd.DataFrame, cfg: Dict[str, Any], work_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, pd.DataFrame]:
    """Split train/test stratifie puis encodage (meme logique que preprocess.py).

    Args:
        clean: Donnees nettoyees et validees (cible incluse).
        cfg: Configuration complete.
        work_dir: Dossier de travail ou ecrire scaler_params.json.

    Returns:
        Tuple: (train encode, test encode, chemin du scaler, test BRUT avec cible).
    """
    target = cfg["features"]["target_column"]
    X_train, X_test, y_train, y_test = split_data(
        clean, target, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )
    raw_test = X_test.assign(**{target: y_test})
    train_df, test_df, scaler_path = encode_train_test(
        X_train.assign(**{target: y_train}), raw_test, cfg, work_dir
    )
    return train_df, test_df, scaler_path, raw_test


@task(name="pretraitement", cache_policy=NO_CACHE)
def preprocess_data(
    clean: pd.DataFrame, cfg: Dict[str, Any], work_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, pd.DataFrame]:
    """Tache Prefect autour de preprocess_frames."""
    return preprocess_frames(clean, cfg, Path(work_dir))


@task(name="entrainement", cache_policy=NO_CACHE)
def train_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict[str, Any], origin: str
) -> List[Dict[str, Any]]:
    """Entraine les 6 candidats (un run MLflow chacun)."""
    return train_candidates(train_df, test_df, cfg, tags={"origin": origin})


@task(name="evaluation", cache_policy=NO_CACHE)
def evaluate_models(
    results: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare les runs et renvoie le champion (regle : models.champion_metric)."""
    return select_champion(results, cfg["models"]["champion_metric"])


@task(name="enregistrement-registry", cache_policy=NO_CACHE)
def register_model(
    champion: Dict[str, Any], cfg: Dict[str, Any], reason: str, promote: bool = True
) -> str:
    """Enregistre le champion (Staging) et le promeut en Production si promote."""
    metric = cfg["models"]["champion_metric"]
    return register_champion(
        champion["model_uri"],
        cfg["mlflow"]["registered_model_name"],
        description=(
            f"{champion['run_name']} ({metric}={champion['metrics'][metric]}) - "
            f"{reason} - decision de recette QA."
        ),
        alias=cfg["mlflow"].get("champion_alias"),
        promote=promote,
    )


@task(name="comparaison-modele-actuel", cache_policy=NO_CACHE)
def compare_with_current(
    champion: Dict[str, Any], raw_test: pd.DataFrame, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare le candidat au modele en service sur les memes clients de test."""
    production = get_production_version(cfg["mlflow"]["registered_model_name"])
    return decide_promotion(
        champion["metrics"], raw_test, cfg, production_exists=production is not None
    )


@task(name="export-modele", cache_policy=NO_CACHE)
def export_model(
    champion: Dict[str, Any],
    cfg: Dict[str, Any],
    feature_names: List[str],
    scaler_path: str,
    version: Optional[str],
) -> Dict[str, Any]:
    """Exporte le champion (model.joblib, metadata.json, scaler) pour l'API."""
    return export_champion(champion, cfg, feature_names, scaler_path, version)


@flow(name="bank-churn-training", log_prints=True)
def training_flow(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """DAG d'entrainement complet, planifie chaque semaine (prefect.yaml).

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Dict[str, Any]: champion, version enregistree et metriques.
    """
    logger = get_run_logger()
    cfg = load_config(config_path)
    opts = retry_options(cfg)
    work_dir = str(Path(cfg["prefect"]["retrain_dir"]) / "training")

    raw = ingest_data.with_options(**opts)(cfg["data"]["reference_path"])
    clean = validate_data(raw, cfg)
    train_df, test_df, scaler_path, raw_test = preprocess_data(clean, cfg, work_dir)
    results = train_models(train_df, test_df, cfg, "prefect-training")
    champion = evaluate_models(results, cfg)
    decision = compare_with_current(champion, raw_test, cfg)
    version = register_model.with_options(**opts)(
        champion,
        cfg,
        f"entrainement planifie ({decision['reason']})",
        decision["promote"],
    )
    if decision["promote"]:
        target = cfg["features"]["target_column"]
        features = [c for c in train_df.columns if c != target]
        export_model(champion, cfg, features, str(scaler_path), version)

    logger.info(
        f"Champion {champion['run_name']} enregistre en version {version} - "
        f"decision : {decision}"
    )
    return {
        "champion": champion["run_name"],
        "version": version,
        "metrics": champion["metrics"],
        "decision": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Prefect d'entrainement")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    training_flow(args.config)
