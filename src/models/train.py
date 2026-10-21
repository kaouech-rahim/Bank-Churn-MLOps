"""Entrainement multi-modeles avec tracking MLflow et Model Registry (TP3).

Etapes : train.csv / test.csv -> 6 runs MLflow (3 algorithmes x baseline/tuned)
-> choix du champion (meilleur ROC-AUC) -> Registry (Staging -> Production)
-> export models/model.joblib + metadata.json + scaler_params.json pour l'API.
"""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import load_config
from src.data.load_data import load_raw_data
from src.models.evaluate import (
    compute_metrics,
    predict_with_threshold,
    save_confusion_matrix,
    save_roc_curve,
)
from src.models.registry import register_champion

# Correspondance "nom dans la config" -> classe scikit-learn compatible
ALGORITHMS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

# MLflow 3 serialise les modeles avec skops (format sur, sans pickle arbitraire) :
# il faut declarer les types internes de nos modeles comme "de confiance".
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def build_model(algorithm: str, params: Dict[str, Any]) -> Any:
    """Instancie un classifieur a partir de son nom et de ses hyperparametres.

    Args:
        algorithm: Cle de ALGORITHMS (logistic_regression, random_forest, xgboost).
        params: Hyperparametres passes au constructeur.

    Returns:
        Any: Estimateur non entraine (API scikit-learn).

    Raises:
        ValueError: Si l'algorithme est inconnu.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Algorithme inconnu : {algorithm} (disponibles : {list(ALGORITHMS)})"
        )
    return ALGORITHMS[algorithm](**params)


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separe les variables explicatives et la cible.

    Args:
        df: DataFrame encode (sortie du pretraitement).
        target_col: Nom de la cible.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y).

    Raises:
        ValueError: Si la cible est absente.
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible absente : {target_col}")
    return df.drop(columns=[target_col]), df[target_col]


def setup_mlflow(cfg: Dict[str, Any]) -> str:
    """Connecte MLflow au backend SQLite et selectionne l'experience.

    L'experience est creee avec l'artifact store de la config (./mlruns par
    defaut) : les tests peuvent ainsi tout ecrire dans un dossier temporaire.

    Args:
        cfg: Configuration complete (section mlflow).

    Returns:
        str: Identifiant de l'experience MLflow.
    """
    ml_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(ml_cfg["experiment_name"])
    if experiment is None:
        artifact_uri = Path(ml_cfg["artifact_root"]).resolve().as_uri()
        experiment_id = mlflow.create_experiment(
            ml_cfg["experiment_name"], artifact_location=artifact_uri
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def run_experiment(
    run_name: str,
    spec: Dict[str, Any],
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    threshold: float,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Entraine un modele dans un run MLflow et journalise tout (TP3, etape 3).

    Sont logues : hyperparametres, 5 metriques, tags, matrice de confusion,
    courbe ROC, signature, exemple d'entree et le modele lui-meme.

    Args:
        run_name: Nom du run (ex : XGBoost-Tuned).
        spec: {"algorithm": ..., "params": {...}} venant de la config.
        data: (X_train, X_test, y_train, y_test).
        threshold: Seuil de decision utilise pour les metriques.
        tags: Tags supplementaires (ex : origine du run).

    Returns:
        Dict[str, Any]: run_name, run_id, model_uri, algorithm, params, metrics, model.
    """
    X_train, X_test, y_train, y_test = data
    print(f"\n[RUN] {run_name}")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"algorithm": spec["algorithm"], **spec["params"]})
        mlflow.log_param("decision_threshold", threshold)
        mlflow.set_tags(
            {
                "framework": (
                    "scikit-learn" if spec["algorithm"] != "xgboost" else "xgboost"
                ),
                "n_train": str(len(X_train)),
                **(tags or {}),
            }
        )

        model = build_model(spec["algorithm"], spec["params"])
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = predict_with_threshold(y_proba, threshold)

        metrics = compute_metrics(y_test, y_proba, threshold)
        mlflow.log_metrics(metrics)
        print("   -> " + " | ".join(f"{k}={v}" for k, v in metrics.items()))

        # Graphiques de diagnostic dans un dossier temporaire puis envoyes a MLflow
        with tempfile.TemporaryDirectory() as tmp:
            cm = save_confusion_matrix(
                y_test, y_pred, run_name, Path(tmp) / f"confusion_matrix_{run_name}.png"
            )
            roc = save_roc_curve(
                y_test, y_proba, run_name, Path(tmp) / f"roc_curve_{run_name}.png"
            )
            mlflow.log_artifact(str(cm), artifact_path="diagnostic_plots")
            mlflow.log_artifact(str(roc), artifact_path="diagnostic_plots")

        # Signature formelle (contrat d'interface) + exemple d'entree
        signature = infer_signature(X_test, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.iloc[:2],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return {
        "run_name": run_name,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "algorithm": spec["algorithm"],
        "params": spec["params"],
        "metrics": metrics,
        "model": model,
    }


def train_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Entraine tous les candidats de la config dans l'experience MLflow.

    Args:
        train_df: Jeu d'entrainement encode (avec la cible).
        test_df: Jeu de test encode (avec la cible).
        cfg: Configuration complete (sections features, mlflow, models).
        tags: Tags ajoutes a chaque run.

    Returns:
        List[Dict[str, Any]]: Un resultat par candidat (voir run_experiment).
    """
    target = cfg["features"]["target_column"]
    X_train, y_train = split_xy(train_df, target)
    X_test, y_test = split_xy(test_df, target)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Les colonnes de train et test ne correspondent pas.")

    setup_mlflow(cfg)
    threshold = cfg["models"]["decision_threshold"]
    return [
        run_experiment(name, spec, (X_train, X_test, y_train, y_test), threshold, tags)
        for name, spec in cfg["models"]["candidates"].items()
    ]


def select_champion(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """Choisit le meilleur run selon une metrique (le premier en cas d'egalite).

    Args:
        results: Sorties de train_candidates.
        metric: Metrique a maximiser (ex : roc_auc).

    Returns:
        Dict[str, Any]: Le resultat champion.

    Raises:
        ValueError: Si la liste est vide ou si la metrique est absente.
    """
    if not results:
        raise ValueError("Aucun run a comparer.")
    if any(metric not in r["metrics"] for r in results):
        raise ValueError(f"Metrique absente de certains runs : {metric}")
    best = results[0]
    for result in results[1:]:
        if result["metrics"][metric] > best["metrics"][metric]:
            best = result
    return best


def export_champion(
    champion: Dict[str, Any],
    cfg: Dict[str, Any],
    feature_names: List[str],
    scaler_source: str,
    registry_version: Optional[str],
) -> Dict[str, Any]:
    """Exporte le champion pour l'API : model.joblib, metadata.json, scaler.

    Args:
        champion: Resultat champion (contient le modele entraine).
        cfg: Configuration complete.
        feature_names: Ordre exact des colonnes attendues par le modele.
        scaler_source: Fichier scaler_params.json utilise a l'entrainement.
        registry_version: Version dans le Registry (None si non enregistree).

    Returns:
        Dict[str, Any]: Les metadonnees ecrites.
    """
    models_cfg = cfg["models"]
    feats = cfg["features"]
    model_path = Path(models_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["model"], model_path)
    shutil.copyfile(scaler_source, models_cfg["scaler_path"])

    metadata = {
        "model_name": cfg["mlflow"]["registered_model_name"],
        "version": registry_version or "local",
        "run_name": champion["run_name"],
        "run_id": champion["run_id"],
        "algorithm": type(champion["model"]).__name__,
        "params": champion["params"],
        "metrics": champion["metrics"],
        "decision_threshold": models_cfg["decision_threshold"],
        "features": {
            "numeric": list(feats["numerical_columns"]),
            "categorical": list(feats["binary_mapping"])
            + list(feats["category_values"]),
            "model_input": feature_names,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(models_cfg["metadata_path"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def run_training(
    config_path: str = "configs/config.yaml", register: bool = True
) -> Dict[str, Any]:
    """Pipeline complet d'entrainement (etape DVC "train").

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        register: Si True, enregistre et promeut le champion dans le Registry.

    Returns:
        Dict[str, Any]: "results" (6 runs), "champion" et "metadata".

    Raises:
        FileNotFoundError: Si la configuration ou les donnees pretraitees manquent.
    """
    cfg = load_config(config_path)
    train_df = load_raw_data(cfg["data"]["train_path"])
    test_df = load_raw_data(cfg["data"]["test_path"])

    results = train_candidates(train_df, test_df, cfg, tags={"origin": "training"})
    metric = cfg["models"]["champion_metric"]
    champion = select_champion(results, metric)
    print(
        f"\n[CHAMPION] {champion['run_name']} ({metric}={champion['metrics'][metric]})"
    )

    version = None
    if register:
        version = register_champion(
            champion["model_uri"],
            cfg["mlflow"]["registered_model_name"],
            description=(
                f"Champion {champion['run_name']} choisi sur le {metric} "
                f"({champion['metrics'][metric]}) : valide pour la recette QA."
            ),
            alias=cfg["mlflow"].get("champion_alias"),
        )

    target = cfg["features"]["target_column"]
    feature_names = [c for c in train_df.columns if c != target]
    metadata = export_champion(
        champion, cfg, feature_names, cfg["data"]["scaler_path"], version
    )

    metrics_path = Path(cfg["models"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "champion": champion["run_name"],
                "champion_metric": metric,
                "runs": {r["run_name"]: r["metrics"] for r in results},
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"[EXPORT] {cfg['models']['model_path']} (version {metadata['version']})")
    return {"results": results, "champion": champion, "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement et tracking MLflow")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Chemin vers le fichier YAML"
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Ne pas enregistrer le champion dans le Model Registry",
    )
    args = parser.parse_args()
    run_training(args.config, register=not args.no_register)
