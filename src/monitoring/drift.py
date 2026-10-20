"""Detection de data drift : test KS, test du chi2 et PSI.

Regle de decision : une variable est "en drift" si le test statistique est
significatif (p < alpha) ET si son PSI est superieur ou egal a psi_threshold.
Le double critere evite les fausses alertes : sur de gros echantillons, un test
seul detecte des ecarts minuscules sans importance.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

# Evite ln(0) et les divisions par zero quand une tranche est vide
EPSILON = 1e-6


def get_monitored_columns(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Renvoie les colonnes surveillees : (numeriques, categorielles).

    Les categorielles regroupent les colonnes binaires et les colonnes One-Hot
    de la configuration (ex : Gender et Geography).

    Args:
        cfg: Configuration complete du projet.

    Returns:
        Tuple[List[str], List[str]]: (colonnes numeriques, colonnes categorielles).
    """
    feats = cfg["features"]
    numerical = list(feats["numerical_columns"])
    categorical = list(feats["binary_mapping"]) + list(feats["category_values"])
    return numerical, categorical


def _psi_from_proportions(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Formule du PSI : somme de (p_cur - p_ref) * ln(p_cur / p_ref).

    Args:
        p_ref: Proportions par tranche dans la population de reference.
        p_cur: Proportions par tranche dans la population courante.

    Returns:
        float: Valeur du PSI (les proportions nulles sont remplacees par EPSILON).
    """
    p_ref = np.clip(p_ref, EPSILON, None)
    p_cur = np.clip(p_cur, EPSILON, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcule le PSI (Population Stability Index) d'une variable numerique.

    Les tranches sont construites sur les quantiles de la REFERENCE.
    Lecture : < 0.1 stable, 0.1 a 0.2 modere, >= 0.2 drift important.

    Args:
        reference: Valeurs de la population de reference.
        current: Valeurs de la population courante (production).
        bins: Nombre de tranches (quantiles) pour la discretisation.

    Returns:
        float: Valeur du PSI (0 = distributions identiques).

    Raises:
        ValueError: Si l'une des deux series est vide.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Impossible de calculer le PSI sur une serie vide.")

    # Bornes internes des tranches (np.unique gere les valeurs repetees, ex : Balance=0)
    cut_points = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = cut_points.size + 1
    ref_counts = np.bincount(
        np.searchsorted(cut_points, ref, side="right"), minlength=n_bins
    )
    cur_counts = np.bincount(
        np.searchsorted(cut_points, cur, side="right"), minlength=n_bins
    )
    return _psi_from_proportions(ref_counts / ref.size, cur_counts / cur.size)


def compute_psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """Calcule le PSI d'une variable categorielle (une "tranche" par modalite).

    Args:
        reference: Modalites de la population de reference.
        current: Modalites de la population courante.

    Returns:
        float: Valeur du PSI (0 = memes proportions).

    Raises:
        ValueError: Si l'une des deux series est vide.
    """
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Impossible de calculer le PSI sur une serie vide.")

    categories = list(dict.fromkeys(list(reference.unique()) + list(current.unique())))
    p_ref = reference.value_counts(normalize=True).reindex(categories).fillna(0.0)
    p_cur = current.value_counts(normalize=True).reindex(categories).fillna(0.0)
    return _psi_from_proportions(p_ref.to_numpy(), p_cur.to_numpy())


def ks_test(
    reference: pd.Series, current: pd.Series, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test de Kolmogorov-Smirnov a 2 echantillons (variables numeriques).

    Args:
        reference: Valeurs de reference.
        current: Valeurs courantes.
        alpha: Seuil de significativite.

    Returns:
        Dict[str, Any]: statistic, p_value et significant (True si p < alpha).
    """
    result = ks_2samp(reference, current)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
    }


def chi2_test(
    reference: pd.Series, current: pd.Series, alpha: float = 0.05
) -> Dict[str, Any]:
    """Test du chi2 d'homogeneite (variables categorielles).

    Args:
        reference: Modalites de reference.
        current: Modalites courantes.
        alpha: Seuil de significativite.

    Returns:
        Dict[str, Any]: statistic, p_value et significant (True si p < alpha).
    """
    categories = list(dict.fromkeys(list(reference.unique()) + list(current.unique())))
    if len(categories) < 2:  # une seule modalite : rien a comparer
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    table = np.array(
        [
            [int((reference == c).sum()) for c in categories],
            [int((current == c).sum()) for c in categories],
        ]
    )
    statistic, p_value, _, _ = chi2_contingency(table)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def detect_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare un batch courant a la reference et mesure le data drift.

    Args:
        reference_df: Donnees de reference (celles de l'entrainement).
        current_df: Batch de production a analyser.
        cfg: Configuration complete (sections features et drift).

    Returns:
        Dict[str, Any]: Rapport avec les cles n_reference, n_current, n_features,
        n_drifted, drift_share, critical, drifted_features et features
        (detail par variable : test, statistic, p_value, psi, drifted).

    Raises:
        ValueError: Si une colonne surveillee manque ou si un echantillon est
            trop petit (min_rows) ou vide apres suppression des valeurs manquantes.
    """
    drift_cfg = cfg["drift"]
    numerical, categorical = get_monitored_columns(cfg)

    missing = [
        c
        for c in numerical + categorical
        if c not in reference_df.columns or c not in current_df.columns
    ]
    if missing:
        raise ValueError(f"Colonnes surveillees manquantes : {missing}")
    for name, frame in [("reference", reference_df), ("courant", current_df)]:
        if len(frame) < drift_cfg["min_rows"]:
            raise ValueError(
                f"Echantillon {name} trop petit : "
                f"{len(frame)} < {drift_cfg['min_rows']}"
            )

    features: Dict[str, Dict[str, Any]] = {}
    for col in numerical + categorical:
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()
        if ref.empty or cur.empty:
            raise ValueError(
                f"Colonne {col} vide apres suppression des valeurs manquantes."
            )

        if col in numerical:
            test_name = "ks"
            test = ks_test(ref, cur, drift_cfg["alpha"])
            psi = compute_psi(ref, cur, drift_cfg["bins"])
        else:
            test_name = "chi2"
            test = chi2_test(ref, cur, drift_cfg["alpha"])
            psi = compute_psi_categorical(ref, cur)

        features[col] = {
            "test": test_name,
            "statistic": test["statistic"],
            "p_value": test["p_value"],
            "psi": psi,
            "drifted": bool(test["significant"] and psi >= drift_cfg["psi_threshold"]),
        }

    drifted = [col for col, res in features.items() if res["drifted"]]
    drift_share = len(drifted) / len(features)
    return {
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "n_features": len(features),
        "n_drifted": len(drifted),
        "drift_share": float(drift_share),
        "critical": bool(drift_share >= drift_cfg["drift_share_threshold"]),
        "drifted_features": drifted,
        "features": features,
    }
