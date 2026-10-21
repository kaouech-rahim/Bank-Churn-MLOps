"""Decision de promotion d'un nouveau champion (partagee par tous les flows).

Un candidat n'est promu en Production que si TOUTES les conditions sont reunies,
les deux modeles etant evalues sur les MEMES clients (le modele en service avec
SON scaler, SON ordre de colonnes et SON seuil de decision) :

1. metrique principale (models.champion_metric, ROC-AUC par defaut) :
   candidat >= modele en service + promotion.min_improvement ;
2. rappel au seuil de decision : candidat >= rappel en service
   - promotion.max_recall_drop (non-regression de la detection des departs) ;
3. si promotion.recall_floor est defini : rappel candidat >= recall_floor.

Cas particuliers :
- aucun modele en service -> promotion (seul le plancher 3 s'applique),
  SAUF si le Registry declare une version en Production alors que le bundle
  models/ est absent (bundle perdu) -> pas de promotion ;
- metrique absente, NaN ou infinie, modele illisible, colonnes incompatibles,
  une seule classe dans les donnees -> PAS de promotion (echec "ferme").
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.features.preprocess import encode_features
from src.models.evaluate import compute_metrics

# Metriques calculees par compute_metrics (utilisables comme critere principal)
SUPPORTED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def should_promote(
    new_score: float, current_score: Optional[float], min_improvement: float = 0.0
) -> bool:
    """Regle sur la metrique principale : le candidat doit faire au moins aussi bien.

    Args:
        new_score: Score du candidat.
        current_score: Score du modele en service (None s'il n'y en a pas).
        min_improvement: Gain minimal exige (0.0 = meilleur ou egal).

    Returns:
        bool: True si la condition est remplie.
    """
    if current_score is None:
        return True
    return new_score >= current_score + min_improvement


def promotion_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lit et valide les regles de promotion de la configuration.

    Args:
        cfg: Configuration complete (section models).

    Returns:
        Dict[str, Any]: metric, min_improvement, max_recall_drop, recall_floor.

    Raises:
        ValueError: Si la metrique principale n'est pas calculable.
    """
    models_cfg = cfg["models"]
    rules = models_cfg.get("promotion", {})
    metric = models_cfg["champion_metric"]
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"champion_metric '{metric}' non supportee pour la promotion "
            f"(choix : {SUPPORTED_METRICS})"
        )
    floor = rules.get("recall_floor")
    return {
        "metric": metric,
        "min_improvement": float(rules.get("min_improvement", 0.0)),
        "max_recall_drop": float(rules.get("max_recall_drop", 0.0)),
        "recall_floor": None if floor is None else float(floor),
    }


def score_current_model(
    eval_clean: pd.DataFrame, cfg: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Metriques du modele en service sur des clients bruts nettoyes.

    Args:
        eval_clean: Clients nettoyes (identifiants retires), avec la cible.
        cfg: Configuration complete.

    Returns:
        Optional[Dict[str, float]]: accuracy, precision, recall, f1, roc_auc
        (au seuil de decision du modele en service), ou None s'il n'y a pas de
        modele en service.

    Raises:
        Exception: Toute erreur de chargement ou de prediction (traitee par
            decide_promotion comme un echec d'evaluation).
    """
    models_cfg = cfg["models"]
    paths = [models_cfg[k] for k in ["model_path", "metadata_path", "scaler_path"]]
    if not all(Path(p).exists() for p in paths):
        return None
    model = joblib.load(paths[0])
    with open(paths[1], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(paths[2], "r", encoding="utf-8") as f:
        scaler = json.load(f)

    feats = cfg["features"]
    target = feats["target_column"]
    X = encode_features(
        eval_clean.drop(columns=[target]),
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler,
    )[metadata["features"]["model_input"]]
    proba = model.predict_proba(X)[:, 1]
    threshold = metadata.get("decision_threshold", models_cfg["decision_threshold"])
    return compute_metrics(eval_clean[target].to_numpy(), proba, threshold)


def _finite(metrics: Optional[Dict[str, Any]], key: str) -> bool:
    """Vrai si metrics[key] existe et est un nombre fini."""
    if not metrics or key not in metrics:
        return False
    try:
        return math.isfinite(float(metrics[key]))
    except (TypeError, ValueError):
        return False


def decide_promotion(
    candidate_metrics: Dict[str, Any],
    eval_clean: pd.DataFrame,
    cfg: Dict[str, Any],
    min_improvement: Optional[float] = None,
    production_exists: bool = False,
) -> Dict[str, Any]:
    """Compare le candidat au modele en service et renvoie la decision.

    Args:
        candidate_metrics: Metriques du candidat sur eval_clean (sortie de
            compute_metrics, avec au moins la metrique principale et recall).
        eval_clean: Clients d'evaluation nettoyes, avec la cible.
        cfg: Configuration complete.
        min_improvement: Gain minimal sur la metrique principale (si None :
            models.promotion.min_improvement).
        production_exists: True si le Registry a deja une version en
            Production (un bundle absent est alors une anomalie, pas un
            premier deploiement).

    Returns:
        Dict[str, Any]: promote, metric, candidate_score, current_score,
        candidate_recall, current_recall et reason (no_current_model,
        better_or_equal, worse, recall_regression, recall_below_floor,
        evaluation_failed: ...).
    """
    rules = promotion_settings(cfg)
    metric = rules["metric"]
    if min_improvement is None:
        min_improvement = rules["min_improvement"]
    decision: Dict[str, Any] = {
        "promote": False,
        "metric": metric,
        "candidate_score": candidate_metrics.get(metric),
        "current_score": None,
        "candidate_recall": candidate_metrics.get("recall"),
        "current_recall": None,
    }

    # dict.fromkeys garde l'ordre et evite de tester deux fois "recall"
    # quand la metrique principale est elle-meme le rappel.
    for key in dict.fromkeys([metric, "recall"]):
        if not _finite(candidate_metrics, key):
            return {**decision, "reason": f"evaluation_failed: candidate_{key}_invalid"}

    try:
        current = score_current_model(eval_clean, cfg)
    except Exception as err:  # echec ferme : on garde le modele en service
        return {
            **decision,
            "reason": f"evaluation_failed: {type(err).__name__}: {err}",
        }

    cand_score = float(candidate_metrics[metric])
    cand_recall = float(candidate_metrics["recall"])
    floor = rules["recall_floor"]

    if current is None:
        if production_exists:
            return {**decision, "reason": "evaluation_failed: current_bundle_missing"}
        if floor is not None and cand_recall < floor:
            return {**decision, "reason": "recall_below_floor"}
        return {**decision, "promote": True, "reason": "no_current_model"}

    for key in dict.fromkeys([metric, "recall"]):
        if not _finite(current, key):
            return {**decision, "reason": f"evaluation_failed: current_{key}_invalid"}
    decision["current_score"] = float(current[metric])
    decision["current_recall"] = float(current["recall"])

    if not should_promote(cand_score, decision["current_score"], min_improvement):
        reason = "worse"
    elif cand_recall < decision["current_recall"] - rules["max_recall_drop"]:
        reason = "recall_regression"
    elif floor is not None and cand_recall < floor:
        reason = "recall_below_floor"
    else:
        return {**decision, "promote": True, "reason": "better_or_equal"}
    return {**decision, "reason": reason}
