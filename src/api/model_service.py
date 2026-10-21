"""Service d'inference : charge le modele et applique le pretraitement (TP4, ex. 1).

Le ModelService ne connait rien de HTTP : il est testable sans FastAPI.
Le pretraitement n'est PAS reecrit : on reutilise encode_features() avec le
scaler appris a l'entrainement (models/scaler_params.json).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd

from src.api.schemas import CustomerFeatures, PredictionResponse, RiskTier
from src.features.preprocess import encode_features


def risk_tier(
    probability: float, moderate: float = 0.35, high: float = 0.70
) -> RiskTier:
    """Classe une probabilite de depart en niveau de risque.

    Args:
        probability: Probabilite de churn entre 0 et 1.
        moderate: Seuil a partir duquel le risque est "Modere".
        high: Seuil a partir duquel le risque est "Eleve".

    Returns:
        RiskTier: "Faible" (< moderate), "Modere" ([moderate, high[) ou "Eleve".
    """
    if probability >= high:
        return "Eleve"
    if probability >= moderate:
        return "Modere"
    return "Faible"


class ModelService:
    """Encapsule le modele, ses metadonnees et le scaler."""

    def __init__(
        self,
        model_path: Union[str, Path],
        metadata_path: Union[str, Path],
        scaler_path: Union[str, Path],
        cfg: Dict[str, Any],
    ) -> None:
        """Initialise puis charge le service.

        Args:
            model_path: Fichier model.joblib du champion.
            metadata_path: Fichier metadata.json (version, features...).
            scaler_path: Fichier scaler_params.json appris sur le train.
            cfg: Configuration complete (sections features et api).

        Raises:
            FileNotFoundError: Si l'un des trois fichiers est absent.
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.scaler_path = Path(scaler_path)
        self.cfg = cfg
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        self.load()

    def load(self) -> None:
        """Charge le modele, les metadonnees et les parametres du scaler.

        Raises:
            FileNotFoundError: Si l'un des fichiers est absent.
        """
        for path in [self.model_path, self.metadata_path, self.scaler_path]:
            if not path.exists():
                raise FileNotFoundError(f"Fichier introuvable : {path}")
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(self.scaler_path, "r", encoding="utf-8") as f:
            self.scaler_params = json.load(f)

    def is_ready(self) -> bool:
        """Indique si le modele est charge en memoire."""
        return self.model is not None

    @property
    def version(self) -> str:
        """Version du modele servi (numero de version du Registry)."""
        return str(self.metadata.get("version", "inconnue"))

    def _to_features(self, customers: List[CustomerFeatures]) -> pd.DataFrame:
        """Transforme des clients valides en matrice d'entree du modele."""
        feats = self.cfg["features"]
        raw = pd.DataFrame([c.model_dump() for c in customers])
        encoded = encode_features(
            raw,
            feats["binary_mapping"],
            feats["category_values"],
            feats["numerical_columns"],
            self.scaler_params,
        )
        # Meme ordre de colonnes qu'a l'entrainement
        return encoded[self.metadata["features"]["model_input"]]

    def _build_response(self, probability: float) -> PredictionResponse:
        """Construit la reponse a partir d'une probabilite."""
        thresholds = self.cfg["api"]["risk_thresholds"]
        decision = self.metadata.get("decision_threshold", 0.5)
        return PredictionResponse(
            churn=probability >= decision,
            churn_probability=round(probability, 4),
            risk_tier=risk_tier(
                probability, thresholds["moderate"], thresholds["high"]
            ),
            model_version=self.version,
        )

    def predict_single(self, customer: CustomerFeatures) -> PredictionResponse:
        """Prediction pour un client.

        Args:
            customer: Client valide par Pydantic.

        Returns:
            PredictionResponse: churn, probabilite, niveau de risque, version.
        """
        results, _ = self.predict_batch([customer])
        return results[0]

    def predict_batch(
        self, customers: List[CustomerFeatures]
    ) -> Tuple[List[PredictionResponse], float]:
        """Prediction vectorisee pour une liste de clients.

        Args:
            customers: Clients valides par Pydantic.

        Returns:
            Tuple[List[PredictionResponse], float]: reponses et probabilite moyenne.

        Raises:
            RuntimeError: Si le modele n'est pas charge.
        """
        if self.model is None:
            raise RuntimeError("Le modele n'est pas charge.")
        if not customers:
            return [], 0.0
        probas = self.model.predict_proba(self._to_features(customers))[:, 1]
        results = [self._build_response(float(p)) for p in probas]
        return results, round(float(probas.mean()), 4)
