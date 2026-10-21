"""Rapport CML publie en commentaire de Pull Request (GitHub Actions).

La CI ne doit dependre ni de DVC/S3 ni du vrai churn.csv : le rapport est donc
calcule sur des donnees SYNTHETIQUES de meme schema (generateur des tests).
Il sert a verifier qu'une modification du code ne casse pas l'entrainement et
a comparer les candidats ; ce ne sont PAS les performances sur les vraies donnees.

Sorties (dossier --out, par defaut cml_report/) :
metrics.json, confusion_matrix.png, roc_curve.png, report.md
"""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.features.preprocess import (
    clean_raw_dataframe,
    encode_features,
    fit_scaler_params,
    split_data,
)
from src.models.evaluate import (
    predict_with_threshold,
    save_confusion_matrix,
    save_roc_curve,
)
from src.models.train import select_champion, split_xy, train_candidates
from tests.conftest import make_churn_df


def build_report(config_path: str, out_dir: str, n_rows: int = 3000) -> Dict[str, Any]:
    """Entraine les candidats sur donnees synthetiques et ecrit le rapport.

    Args:
        config_path: Configuration du projet (candidats, seuils...).
        out_dir: Dossier de sortie du rapport.
        n_rows: Nombre de lignes synthetiques.

    Returns:
        Dict[str, Any]: Contenu de metrics.json.
    """
    cfg = copy.deepcopy(load_config(config_path))
    feats = cfg["features"]
    target = feats["target_column"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Donnees synthetiques -> meme pretraitement que le vrai pipeline
    df = clean_raw_dataframe(make_churn_df(n_rows, seed=7), feats["id_columns"], target)
    X_train, X_test, y_train, y_test = split_data(df, target, 0.2, 42)
    scaler = fit_scaler_params(X_train, feats["numerical_columns"])
    encode_args = (
        feats["binary_mapping"],
        feats["category_values"],
        feats["numerical_columns"],
        scaler,
    )
    train_df = encode_features(X_train, *encode_args).assign(**{target: y_train})
    test_df = encode_features(X_test, *encode_args).assign(**{target: y_test})

    # 2. MLflow dans un dossier temporaire (aucun serveur dans la CI).
    # ignore_cleanup_errors : sous Windows, mlflow.db reste ouvert jusqu'a la fin.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg["mlflow"]["tracking_uri"] = f"sqlite:///{Path(tmp, 'mlflow.db').as_posix()}"
        cfg["mlflow"]["artifact_root"] = str(Path(tmp, "mlruns"))
        results = train_candidates(train_df, test_df, cfg, tags={"origin": "ci-cml"})

    metric = cfg["models"]["champion_metric"]
    champion = select_champion(results, metric)

    # 3. Graphiques du champion
    X_test_enc, y_true = split_xy(test_df, target)
    y_proba = champion["model"].predict_proba(X_test_enc)[:, 1]
    y_pred = predict_with_threshold(y_proba, cfg["models"]["decision_threshold"])
    save_confusion_matrix(
        y_true, y_pred, champion["run_name"], out / "confusion_matrix.png"
    )
    save_roc_curve(y_true, y_proba, champion["run_name"], out / "roc_curve.png")

    summary = {
        "data": f"synthetique ({n_rows} lignes)",
        "champion": champion["run_name"],
        "champion_metric": metric,
        "runs": {r["run_name"]: r["metrics"] for r in results},
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # 4. Rapport Markdown (publie par "cml comment create")
    lines = [
        "## Rapport CML - entrainement de controle",
        "",
        f"Donnees : **{summary['data']}** (la CI n'accede pas au vrai dataset).",
        f"Champion ({metric}) : **{champion['run_name']}**",
        "",
        "| Run | accuracy | precision | recall | f1 | roc_auc |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in summary["runs"].items():
        lines.append(
            f"| {name} | {m['accuracy']} | {m['precision']} | {m['recall']} "
            f"| {m['f1']} | {m['roc_auc']} |"
        )
    lines += [
        "",
        "### Matrice de confusion du champion",
        "![](./confusion_matrix.png)",
        "",
        "### Courbe ROC du champion",
        "![](./roc_curve.png)",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[CML] Rapport ecrit dans {out} (champion : {champion['run_name']})")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapport CML sur donnees synthetiques")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="cml_report")
    parser.add_argument("--rows", type=int, default=3000)
    args = parser.parse_args()
    build_report(args.config, args.out, args.rows)
