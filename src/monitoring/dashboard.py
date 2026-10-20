"""Dashboard HTML de drift (Evidently) et resume JSON du rapport de drift."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import load_config
from src.data.load_data import load_raw_data
from src.monitoring.drift import detect_drift, get_monitored_columns
from src.monitoring.drift_simulator import DriftSimulator


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    cfg: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Genere le rapport HTML interactif Evidently (reference vs courant).

    Le verdict d'Evidently est regle sur le PSI (meme seuil que notre moteur) pour
    que le dashboard et detect_drift() designent les memes variables. Les valeurs
    de PSI different legerement des notres (decoupage en tranches different) :
    la decision officielle reste celle de detect_drift().

    Args:
        reference_df: Donnees de reference.
        current_df: Batch courant (eventuellement avec drift simule).
        cfg: Configuration complete du projet.
        output_path: Chemin du fichier HTML a ecrire.

    Returns:
        Path: Chemin du fichier HTML genere.
    """
    numerical, categorical = get_monitored_columns(cfg)
    columns = numerical + categorical
    definition = DataDefinition(
        numerical_columns=numerical, categorical_columns=categorical
    )
    reference = Dataset.from_pandas(reference_df[columns], data_definition=definition)
    current = Dataset.from_pandas(current_df[columns], data_definition=definition)

    report = Report(
        [
            DataDriftPreset(
                columns=columns,
                drift_share=cfg["drift"]["drift_share_threshold"],
                method="psi",
                threshold=cfg["drift"]["psi_threshold"],
            )
        ]
    )
    snapshot = report.run(current_data=current, reference_data=reference)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path


def run_dashboard(
    config_path: str = "configs/config.yaml", scenario: str = "strong"
) -> Dict[str, Any]:
    """Charge les donnees, simule le drift, ecrit le HTML et le resume JSON.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        scenario: Nom du scenario de drift a simuler (none, strong...).

    Returns:
        Dict[str, Any]: Rapport de drift renvoye par detect_drift().

    Raises:
        FileNotFoundError: Si la configuration ou un fichier de donnees manque.
        ValueError: Si le scenario est inconnu ou les echantillons invalides.
    """
    cfg = load_config(config_path)
    reference = load_raw_data(cfg["data"]["reference_path"])
    production = load_raw_data(cfg["data"]["production_pool_path"])

    simulator = DriftSimulator.from_config(cfg)
    current = simulator.apply(production, scenario, cfg["drift"]["simulation_seed"])

    html_path = build_drift_report(reference, current, cfg, cfg["drift"]["report_path"])
    report = detect_drift(reference, current, cfg)

    json_path = html_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario, **report}, f, indent=2)
        f.write("\n")  # saut de ligne final (exige par le hook end-of-file-fixer)

    print("=" * 50)
    print(f"RAPPORT DE DRIFT - scenario : {scenario}")
    print("=" * 50)
    print(
        f"* drift_share : {report['drift_share']:.3f} | critique : {report['critical']}"
    )
    print(f"* variables en drift : {report['drifted_features']}")
    print(f"* HTML : {html_path}")
    print(f"* JSON : {json_path}")
    print("=" * 50)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard de drift Evidently")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Chemin vers le fichier YAML"
    )
    parser.add_argument(
        "--scenario", default="strong", help="Scenario de drift (none, strong...)"
    )
    args = parser.parse_args()
    run_dashboard(args.config, args.scenario)
