"""Evaluation d'un classifieur : metriques industrielles et graphiques (TP3)."""

from pathlib import Path
from typing import Dict, Union

import matplotlib

matplotlib.use("Agg")  # backend non interactif : generation de PNG sans ecran
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_with_threshold(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Transforme des probabilites de churn en classes 0/1.

    Args:
        y_proba: Probabilites de la classe 1 (churn).
        threshold: Seuil de decision (proba >= seuil -> 1).

    Returns:
        np.ndarray: Classes predites (0 ou 1).

    Raises:
        ValueError: Si le seuil n'est pas dans ]0, 1[.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"Le seuil doit etre dans ]0, 1[ (recu : {threshold})")
    return (np.asarray(y_proba) >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calcule accuracy, precision, recall, f1 et roc_auc (arrondis a 4 decimales).

    Args:
        y_true: Vraies classes (0/1).
        y_proba: Probabilites predites pour la classe 1.
        threshold: Seuil de decision.

    Returns:
        Dict[str, float]: Les 5 metriques.

    Raises:
        ValueError: Si y_true ne contient qu'une seule classe (ROC-AUC indefini).
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true doit contenir les deux classes pour le ROC-AUC.")
    y_pred = predict_with_threshold(y_proba, threshold)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Union[str, Path]
) -> Path:
    """Dessine la matrice de confusion et l'enregistre en PNG.

    Args:
        y_true: Vraies classes.
        y_pred: Classes predites.
        title: Titre du graphique (ex : nom du run).
        path: Fichier PNG de sortie.

    Returns:
        Path: Chemin du PNG ecrit.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Matrice de confusion -- {title}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, path: Union[str, Path]
) -> Path:
    """Dessine la courbe ROC et l'enregistre en PNG.

    Args:
        y_true: Vraies classes.
        y_proba: Probabilites de la classe 1.
        title: Titre du graphique.
        path: Fichier PNG de sortie.

    Returns:
        Path: Chemin du PNG ecrit.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"Courbe ROC -- {title}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
