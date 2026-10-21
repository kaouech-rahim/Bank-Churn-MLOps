"""Gouvernance du modele : Model Registry MLflow (TP3, etape 6).

On suit le cycle de vie du TP3 : enregistrement -> Staging (description QA)
-> Production, avec archivage automatique de l'ancienne version de production.
MLflow 3 marque les "stages" comme deprecies au profit des alias : on garde
les stages (demandes par le TP et lus par models:/BankChurnModel/Production)
et on pose EN PLUS l'alias "champion" pour etre compatible avec les futures versions.
"""

from typing import Any, Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def register_champion(
    model_uri: str,
    model_name: str,
    description: str,
    alias: Optional[str] = "champion",
    client: Optional[MlflowClient] = None,
    promote: bool = True,
) -> str:
    """Enregistre le champion, le passe en Staging et, si promote, en Production.

    Args:
        model_uri: URI du modele logue (ex : runs:/<run_id>/model).
        model_name: Nom du modele dans le Registry (ex : BankChurnModel).
        description: Description QA ajoutee a la version.
        alias: Alias a poser sur la version (None pour ne pas en poser).
        client: Client MLflow (cree automatiquement si absent).
        promote: Si False, la version reste en Staging (tracee mais non servie).

    Returns:
        str: Numero de la version enregistree.
    """
    client = client or MlflowClient()
    print(f"[REGISTRY] Enregistrement de {model_uri} sous '{model_name}'...")
    details = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(details.version)

    # 1. Staging : validation QA
    # archive_existing_versions : une seule version a la fois en Staging
    client.transition_model_version_stage(
        model_name, version, stage="Staging", archive_existing_versions=True
    )
    client.update_model_version(model_name, version, description=description)
    if not promote:
        print(f"[REGISTRY] '{model_name}' v{version} reste en STAGING (non promu).")
        return version

    # 2. Production : les versions precedentes en production sont archivees
    client.transition_model_version_stage(
        model_name, version, stage="Production", archive_existing_versions=True
    )
    if alias:
        client.set_registered_model_alias(model_name, alias, version)
    print(f"[REGISTRY] '{model_name}' v{version} est en PRODUCTION.")
    return version


def get_production_version(
    model_name: str, client: Optional[MlflowClient] = None
) -> Optional[ModelVersion]:
    """Renvoie la version actuellement en Production (ou None s'il n'y en a pas).

    Args:
        model_name: Nom du modele dans le Registry.
        client: Client MLflow (cree automatiquement si absent).

    Returns:
        Optional[ModelVersion]: La version en Production, ou None.
    """
    client = client or MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    in_prod = [v for v in versions if v.current_stage == "Production"]
    if not in_prod:
        return None
    return max(in_prod, key=lambda v: int(v.version))


def load_production_model(model_name: str, stage: str = "Production") -> Any:
    """Charge le modele via son URI logique models:/<nom>/<stage>.

    Args:
        model_name: Nom du modele dans le Registry.
        stage: Stage a charger (Production par defaut).

    Returns:
        Any: Modele pyfunc MLflow (methode predict).
    """
    return mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
