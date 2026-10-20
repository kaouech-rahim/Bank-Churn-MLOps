"""Chargement de la configuration YAML centrale du projet."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(
    config_path: Union[str, Path] = "configs/config.yaml"
) -> Dict[str, Any]:
    """Lit le fichier de configuration YAML et le renvoie sous forme de dictionnaire.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Dict[str, Any]: Contenu du fichier (sections data, features, schema, drift).

    Raises:
        FileNotFoundError: Si le fichier de configuration est introuvable.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
