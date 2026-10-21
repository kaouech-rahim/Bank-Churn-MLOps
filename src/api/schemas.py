"""Schemas Pydantic v2 des entrees et sorties de l'API (TP4, exercice 2).

Les bornes reprennent la section "schema" de configs/config.yaml (contrat de
donnees du pretraitement) ; le test test_schema_bounds_match_config verifie que
les deux restent synchronises. Tout payload non conforme est rejete en HTTP 422.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

# Taille maximale d'un batch (identique a api.max_batch_size dans la config)
MAX_BATCH_SIZE = 1000

RiskTier = Literal["Faible", "Modere", "Eleve"]


class CustomerFeatures(BaseModel):
    """Caracteristiques d'un client bancaire (format brut de churn.csv)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        }
    )

    CreditScore: int = Field(..., ge=300, le=850, description="Score de credit")
    Geography: Literal["France", "Germany", "Spain"] = Field(
        ..., description="Pays du client"
    )
    Gender: Literal["Female", "Male"] = Field(..., description="Genre du client")
    Age: int = Field(..., ge=18, le=100, description="Age en annees")
    Tenure: int = Field(..., ge=0, le=50, description="Anciennete en annees")
    Balance: float = Field(..., ge=0, le=1_000_000, description="Solde du compte")
    NumOfProducts: int = Field(
        ..., ge=1, le=10, description="Nombre de produits bancaires"
    )
    HasCrCard: int = Field(..., ge=0, le=1, description="Possede une carte (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Client actif (0/1)")
    EstimatedSalary: float = Field(
        ..., ge=0, le=1_000_000, description="Salaire annuel estime"
    )


class PredictionResponse(BaseModel):
    """Resultat de l'inference pour un client."""

    churn: bool = Field(..., description="Vrai si le client risque de partir")
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilite estimee de depart"
    )
    risk_tier: RiskTier = Field(..., description="Niveau de risque")
    model_version: str = Field(..., description="Version du modele (Registry)")


class BatchCustomerRequest(BaseModel):
    """Requete d'inference par lot."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE, description="Clients a evaluer"
    )


class BatchPredictionResponse(BaseModel):
    """Reponse agregee d'une requete batch."""

    predictions: List[PredictionResponse]
    count: int = Field(..., description="Nombre de clients evalues")
    average_churn_probability: float = Field(
        ..., description="Probabilite moyenne de depart"
    )


class HealthResponse(BaseModel):
    """Etat de sante du service (sonde Docker / Kubernetes)."""

    status: str = "healthy"
    model_loaded: bool = True
    timestamp: str
    version: str


class ModelInfoResponse(BaseModel):
    """Informations de gouvernance sur le modele servi."""

    model_name: str
    version: str
    algorithm: str
    run_name: str
    decision_threshold: float
    metrics: Dict[str, float]
    features: Dict[str, List[str]]
