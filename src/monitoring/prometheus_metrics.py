"""Metriques Prometheus exposees par l'API sur /metrics.

- churn_predictions_total : nombre de clients predits (par endpoint)
- churn_prediction_latency_seconds : latence d'inference (histogramme)
- churn_model_info : version et algorithme du modele servi (valeur 1)
- churn_drift_* : valeurs lues dans le dernier rapport de drift (JSON)

Un registre dedie (et non le registre global) evite les doublons quand
l'application est importee plusieurs fois (tests).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

PREDICTIONS = Counter(
    "churn_predictions",
    "Nombre de clients evalues par le modele",
    ["endpoint"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latence d'inference (pretraitement + modele) en secondes",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "churn_model_info",
    "Modele servi (la valeur vaut 1, les infos sont dans les labels)",
    ["model_name", "version", "algorithm"],
    registry=REGISTRY,
)
DRIFT_SHARE = Gauge(
    "churn_drift_share", "Part des variables en drift (0 a 1)", registry=REGISTRY
)
DRIFT_CRITICAL = Gauge(
    "churn_drift_critical", "1 si le drift est critique, 0 sinon", registry=REGISTRY
)
DRIFT_FEATURE_PSI = Gauge(
    "churn_drift_feature_psi", "PSI de chaque variable", ["feature"], registry=REGISTRY
)
DRIFT_REPORT_AVAILABLE = Gauge(
    "churn_drift_report_available",
    "1 si le rapport de drift a pu etre lu, 0 sinon",
    registry=REGISTRY,
)


def set_model_info(metadata: Dict[str, Any]) -> None:
    """Publie la version et l'algorithme du modele servi.

    Args:
        metadata: Contenu de models/metadata.json.
    """
    MODEL_INFO.clear()
    MODEL_INFO.labels(
        model_name=str(metadata.get("model_name", "inconnu")),
        version=str(metadata.get("version", "inconnue")),
        algorithm=str(metadata.get("algorithm", "inconnu")),
    ).set(1)


def read_drift_report(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Lit le rapport JSON de drift sans jamais lever d'exception.

    Args:
        path: Chemin du rapport (ex : reports/drift_report.json).

    Returns:
        Optional[Dict[str, Any]]: Le rapport, ou None s'il est absent ou invalide.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        float(report["drift_share"])  # verifie la presence des cles utiles
        bool(report["critical"])
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def update_drift_metrics(path: Union[str, Path]) -> bool:
    """Met a jour les jauges de drift a partir du rapport JSON.

    Si le rapport est absent ou invalide, seule churn_drift_report_available
    passe a 0 : l'API continue de fonctionner normalement.

    Args:
        path: Chemin du rapport de drift.

    Returns:
        bool: True si le rapport a ete lu.
    """
    report = read_drift_report(path)
    if report is None:
        DRIFT_REPORT_AVAILABLE.set(0)
        return False

    DRIFT_REPORT_AVAILABLE.set(1)
    DRIFT_SHARE.set(float(report["drift_share"]))
    DRIFT_CRITICAL.set(1 if report["critical"] else 0)
    DRIFT_FEATURE_PSI.clear()
    for feature, details in report.get("features", {}).items():
        if isinstance(details, dict) and "psi" in details:
            DRIFT_FEATURE_PSI.labels(feature=feature).set(float(details["psi"]))
    return True
