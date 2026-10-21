"""Flow Prefect de surveillance du drift et boucle de retroaction.

1. Lot de production : echantillon du pool de production, coupe en deux parties
   disjointes ("fit" et "holdout"), puis drift simule sur chaque partie
   (copies en memoire : le pool sur disque n'est jamais modifie).
2. detect_drift() sur le lot complet : KS / chi2 / PSI -> rapport JSON
   (lu par Prometheus) + dashboard HTML Evidently.
3. Si drift critique (drift_share >= 0.30) : reentrainement sur reference + "fit".
4. Comparaison equitable sur le "holdout" (donnees de production jamais vues par
   l'ancien NI par le nouveau modele). Promotion SEULEMENT si
   ROC-AUC nouveau >= ROC-AUC actuel + min_improvement ; sinon on garde l'ancien.

Limite assumee sur les labels : on ne fabrique aucun label. Le pool contient les
vrais labels (Exited) historiques ; apres injection artificielle du drift, ces
labels sont conserves tels quels, ce qui est une approximation de simulation.
En production, il faudrait attendre les labels reels (departs constates). Si le
lot n'a pas de colonne cible, le reentrainement est saute.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pipelines.training_flow import (
    encode_train_test,
    evaluate_models,
    export_model,
    ingest_data,
    register_model,
    retry_options,
    train_models,
    validate_data,
)
from src.config import load_config
from src.features.preprocess import encode_features
from src.monitoring.dashboard import build_drift_report
from src.monitoring.drift import detect_drift
from src.monitoring.drift_simulator import DriftSimulator


def should_promote(
    new_score: float, current_score: Optional[float], min_improvement: float = 0.0
) -> bool:
    """Regle de promotion : le nouveau modele doit faire au moins aussi bien.

    Args:
        new_score: ROC-AUC du nouveau champion (sur le holdout).
        current_score: ROC-AUC du modele en service (None s'il n'y en a pas).
        min_improvement: Gain minimal exige (0.0 = meilleur ou egal).

    Returns:
        bool: True si le nouveau modele doit etre promu.
    """
    if current_score is None:
        return True
    return new_score >= current_score + min_improvement


@task(name="lot-de-production", cache_policy=NO_CACHE)
def build_production_batch(
    pool: pd.DataFrame, cfg: Dict[str, Any], scenario: str, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tire un lot du pool, le coupe en (fit, holdout) puis injecte le drift.

    La coupe a lieu AVANT la simulation : le simulateur re-echantillonne avec
    remise, et un meme client ne peut donc pas se retrouver des deux cotes.

    Args:
        pool: Pool de production brut.
        cfg: Configuration complete.
        scenario: Scenario de drift (none, strong...).
        seed: Graine aleatoire.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (partie fit, partie holdout).
    """
    size = min(cfg["prefect"]["batch_size"], len(pool))
    sample = pool.sample(n=size, random_state=seed)
    target = cfg["features"]["target_column"]
    stratify = sample[target] if target in sample.columns else None
    fit_raw, holdout_raw = train_test_split(
        sample,
        test_size=cfg["data"]["test_size"],
        random_state=seed,
        stratify=stratify,
    )
    simulator = DriftSimulator.from_config(cfg)
    return (
        simulator.apply(fit_raw, scenario, seed),
        simulator.apply(holdout_raw, scenario, seed + 1),
    )


@task(name="detection-drift", cache_policy=NO_CACHE)
def compute_drift(
    reference: pd.DataFrame, batch: pd.DataFrame, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Calcule KS / chi2 / PSI et la part de variables en drift."""
    return detect_drift(reference, batch, cfg)


@task(name="rapports-drift", cache_policy=NO_CACHE)
def write_reports(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    report: Dict[str, Any],
    cfg: Dict[str, Any],
    scenario: str,
) -> Dict[str, str]:
    """Ecrit le rapport JSON (Prometheus) et le dashboard HTML Evidently."""
    json_path = Path(cfg["monitoring"]["drift_report_json"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario, **report}, f, indent=2)
        f.write("\n")
    html_path = build_drift_report(
        reference, batch, cfg, json_path.with_suffix(".html")
    )
    return {"json": str(json_path), "html": str(html_path)}


def score_current_model(
    holdout_clean: pd.DataFrame, cfg: Dict[str, Any]
) -> Optional[float]:
    """ROC-AUC du modele en service sur le holdout (None s'il n'y a pas de modele).

    Le modele actuel est evalue avec SON scaler et SON ordre de colonnes.

    Args:
        holdout_clean: Clients du holdout nettoyes, avec la cible.
        cfg: Configuration complete.

    Returns:
        Optional[float]: ROC-AUC arrondi a 4 decimales, ou None.
    """
    models_cfg = cfg["models"]
    paths = [models_cfg[k] for k in ["model_path", "metadata_path", "scaler_path"]]
    if not all(Path(p).exists() for p in paths):
        return None
    model = joblib.load(paths[0])
    with open(paths[1], "r", encoding="utf-8") as f:
        feature_order = json.load(f)["features"]["model_input"]
    with open(paths[2], "r", encoding="utf-8") as f:
        scaler = json.load(f)

    feats = cfg["features"]
    target = feats["target_column"]
    X = encode_features(
        holdout_clean.drop(columns=[target]),
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler,
    )[feature_order]
    proba = model.predict_proba(X)[:, 1]
    return round(float(roc_auc_score(holdout_clean[target], proba)), 4)


@task(name="score-modele-actuel", cache_policy=NO_CACHE)
def evaluate_current_model(
    holdout_clean: pd.DataFrame, cfg: Dict[str, Any]
) -> Optional[float]:
    """Tache Prefect autour de score_current_model."""
    return score_current_model(holdout_clean, cfg)


@task(name="pretraitement-reentrainement", cache_policy=NO_CACHE)
def preprocess_retraining(
    train_clean: pd.DataFrame, holdout_clean: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Encode reference + fit (train) et holdout (test) avec un nouveau scaler."""
    work_dir = Path(cfg["prefect"]["retrain_dir"]) / "drift_retrain"
    return encode_train_test(train_clean, holdout_clean, cfg, work_dir)


def retrain_and_compare(
    reference: pd.DataFrame,
    batch_fit: pd.DataFrame,
    batch_holdout: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Reentraine sur reference + fit, compare sur holdout, promeut si autorise.

    Args:
        reference: Donnees de reference brutes (avec la cible).
        batch_fit: Partie du lot de production utilisee pour l'entrainement.
        batch_holdout: Partie du lot reservee a la comparaison des modeles.
        cfg: Configuration complete.

    Returns:
        Dict[str, Any]: scores, decision et version eventuellement promue.
    """
    logger = get_run_logger()
    opts = retry_options(cfg)
    train_clean = validate_data(
        pd.concat([reference, batch_fit], ignore_index=True), cfg
    )
    holdout_clean = validate_data(batch_holdout, cfg)
    train_df, test_df, scaler_path = preprocess_retraining(
        train_clean, holdout_clean, cfg
    )

    results = train_models(train_df, test_df, cfg, "prefect-retraining")
    champion = evaluate_models(results, cfg)
    new_score = champion["metrics"]["roc_auc"]
    current_score = evaluate_current_model(holdout_clean, cfg)
    promote = should_promote(
        new_score, current_score, cfg["prefect"]["min_improvement"]
    )
    logger.info(
        f"ROC-AUC holdout : nouveau={new_score} / actuel={current_score} "
        f"-> promotion={promote}"
    )

    version = None
    if promote:
        version = register_model.with_options(**opts)(
            champion, cfg, "reentrainement apres drift critique"
        )
        target = cfg["features"]["target_column"]
        features = [c for c in train_df.columns if c != target]
        export_model(champion, cfg, features, str(scaler_path), version)
    return {
        "new_champion": champion["run_name"],
        "new_roc_auc": new_score,
        "current_roc_auc": current_score,
        "promoted": promote,
        "version": version,
        "n_training_rows": len(train_df),
        "n_holdout_rows": len(test_df),
    }


@flow(name="bank-churn-drift-monitoring", log_prints=True)
def drift_flow(
    config_path: str = "configs/config.yaml",
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Surveillance quotidienne du drift et reentrainement automatique.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        scenario: Scenario de drift simule (defaut : prefect.drift_scenario).
        seed: Graine du lot (defaut : drift.simulation_seed).

    Returns:
        Dict[str, Any]: drift_share, critical, rapports et resultat du reentrainement.
    """
    logger = get_run_logger()
    cfg = load_config(config_path)
    opts = retry_options(cfg)
    scenario = scenario or cfg["prefect"]["drift_scenario"]
    seed = cfg["drift"]["simulation_seed"] if seed is None else seed

    reference = ingest_data.with_options(**opts)(cfg["data"]["reference_path"])
    pool = ingest_data.with_options(**opts)(cfg["data"]["production_pool_path"])
    batch_fit, batch_holdout = build_production_batch(pool, cfg, scenario, seed)
    batch = pd.concat([batch_fit, batch_holdout], ignore_index=True)
    report = compute_drift(reference, batch, cfg)
    paths = write_reports(reference, batch, report, cfg, scenario)
    logger.info(
        f"drift_share={report['drift_share']:.3f} critique={report['critical']} "
        f"variables={report['drifted_features']}"
    )

    retraining: Dict[str, Any] = {"triggered": False}
    target = cfg["features"]["target_column"]
    if report["critical"]:
        if target not in batch.columns:
            logger.warning(
                "Drift critique mais lot sans labels : reentrainement saute."
            )
            retraining = {"triggered": False, "reason": "labels_missing"}
        else:
            retraining = {
                "triggered": True,
                **retrain_and_compare(reference, batch_fit, batch_holdout, cfg),
            }

    return {
        "scenario": scenario,
        "drift_share": report["drift_share"],
        "critical": report["critical"],
        "drifted_features": report["drifted_features"],
        "reports": paths,
        "retraining": retraining,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Prefect de drift")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scenario", default=None, help="none, strong...")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = drift_flow(args.config, args.scenario, args.seed)
    print(
        f"scenario={result['scenario']} drift_share={result['drift_share']:.3f} "
        f"critique={result['critical']}"
    )
    print(f"variables en drift : {result['drifted_features']}")
    print(f"reentrainement : {result['retraining']}")
