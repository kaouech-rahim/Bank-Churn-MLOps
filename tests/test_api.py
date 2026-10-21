"""Tests de l'API FastAPI avec TestClient (TP4, exercice 5).

Un mini-modele est entraine dans la fixture sur des donnees synthetiques :
aucun test ne depend des fichiers du dossier models/.
"""

from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_service import ModelService, risk_tier
from src.api.schemas import MAX_BATCH_SIZE, CustomerFeatures
from src.config import load_config

VALID_CUSTOMER: Dict[str, Any] = {
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


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, model_files: Dict[str, Path], tmp_path: Path
) -> Iterator[TestClient]:
    """Client HTTP de test branche sur le mini-modele."""
    monkeypatch.setenv("MODEL_PATH", str(model_files["model"]))
    monkeypatch.setenv("METADATA_PATH", str(model_files["metadata"]))
    monkeypatch.setenv("SCALER_PATH", str(model_files["scaler"]))
    monkeypatch.setenv("DRIFT_REPORT_PATH", str(tmp_path / "absent.json"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    """Client HTTP dont le modele est introuvable."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setenv("SCALER_PATH", str(tmp_path / "nope_scaler.json"))
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------ endpoints systeme
def test_health_endpoint(client: TestClient) -> None:
    """/health renvoie 200 et model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "7"
    assert "timestamp" in data


def test_info_endpoint(client: TestClient) -> None:
    """/info renvoie les metadonnees de gouvernance."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "BankChurnModel"
    assert data["algorithm"] == "RandomForestClassifier"
    assert "roc_auc" in data["metrics"]
    assert len(data["features"]["model_input"]) == 11


@pytest.mark.parametrize(
    "method, url", [("get", "/health"), ("get", "/info"), ("post", "/predict")]
)
def test_503_when_model_missing(
    client_without_model: TestClient, method: str, url: str
) -> None:
    """Sans modele, l'API repond 503 au lieu de planter."""
    kwargs = {"json": VALID_CUSTOMER} if method == "post" else {}
    response = getattr(client_without_model, method)(url, **kwargs)
    assert response.status_code == 503


# ------------------------------------------------------------ prediction unitaire
def test_predict_single_valid(client: TestClient) -> None:
    """Un client valide renvoie une reponse complete et coherente."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["churn"], bool)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["Faible", "Modere", "Eleve"]
    assert data["model_version"] == "7"
    assert data["churn"] == (data["churn_probability"] >= 0.5)


@pytest.mark.parametrize(
    "field, value",
    [
        ("Geography", "Italy"),  # pays inconnu
        ("Gender", "Other"),  # modalite inconnue
        ("Age", 12),  # sous la borne 18
        ("CreditScore", 900),  # au-dessus de 850
        ("Balance", -5.0),  # solde negatif
        ("HasCrCard", 2),  # binaire hors {0, 1}
        ("Tenure", "abc"),  # mauvais type
    ],
)
def test_predict_invalid_payload_422(
    client: TestClient, field: str, value: Any
) -> None:
    """Pydantic rejette les valeurs invalides avec un code 422."""
    response = client.post("/predict", json={**VALID_CUSTOMER, field: value})
    assert response.status_code == 422


def test_predict_missing_field_422(client: TestClient) -> None:
    """Un champ manquant est rejete."""
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "Age"}
    assert client.post("/predict", json=payload).status_code == 422


# ------------------------------------------------------------ prediction batch
def test_predict_batch(client: TestClient) -> None:
    """Le batch traite plusieurs clients et calcule la moyenne."""
    other = {**VALID_CUSTOMER, "Geography": "Germany", "Age": 60, "Balance": 120000}
    response = client.post(
        "/predict/batch", json={"customers": [VALID_CUSTOMER, other]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    probas = [p["churn_probability"] for p in data["predictions"]]
    assert data["average_churn_probability"] == pytest.approx(sum(probas) / 2, abs=1e-3)


def test_batch_limits(client: TestClient) -> None:
    """0 client ou plus de 1 000 clients : 422 ; exactement 1 000 : 200."""
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    too_many = {"customers": [VALID_CUSTOMER] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422
    full = {"customers": [VALID_CUSTOMER] * MAX_BATCH_SIZE}
    response = client.post("/predict/batch", json=full)
    assert response.status_code == 200
    assert response.json()["count"] == MAX_BATCH_SIZE


# ------------------------------------------------------------ logique metier
@pytest.mark.parametrize(
    "proba, expected",
    [
        (0.0, "Faible"),
        (0.3499, "Faible"),
        (0.35, "Modere"),
        (0.6999, "Modere"),
        (0.70, "Eleve"),
        (1.0, "Eleve"),
    ],
)
def test_risk_tier_boundaries(proba: float, expected: str) -> None:
    """Seuils exacts : Faible < 0.35 <= Modere < 0.70 <= Eleve."""
    assert risk_tier(proba, 0.35, 0.70) == expected


def test_service_reuses_training_preprocessing(model_files: Dict[str, Path]) -> None:
    """Le service donne la meme proba que le modele applique a encode_features."""
    cfg = load_config("configs/config.yaml")
    service = ModelService(
        model_files["model"], model_files["metadata"], model_files["scaler"], cfg
    )
    customer = CustomerFeatures(**VALID_CUSTOMER)
    expected = service.model.predict_proba(service._to_features([customer]))[0, 1]
    assert service.predict_single(customer).churn_probability == pytest.approx(
        round(float(expected), 4)
    )
    assert service.predict_batch([]) == ([], 0.0)


def test_service_missing_file_raises(
    model_files: Dict[str, Path], tmp_path: Path
) -> None:
    """Un fichier manquant leve FileNotFoundError au chargement."""
    cfg = load_config("configs/config.yaml")
    with pytest.raises(FileNotFoundError):
        ModelService(
            model_files["model"], tmp_path / "x.json", model_files["scaler"], cfg
        )


def test_service_not_loaded_raises(model_files: Dict[str, Path]) -> None:
    """predict_batch refuse de tourner si le modele a ete decharge."""
    cfg = load_config("configs/config.yaml")
    service = ModelService(
        model_files["model"], model_files["metadata"], model_files["scaler"], cfg
    )
    service.model = None
    assert service.is_ready() is False
    with pytest.raises(RuntimeError):
        service.predict_batch([CustomerFeatures(**VALID_CUSTOMER)])


def test_schema_bounds_match_config() -> None:
    """Les bornes Pydantic restent synchronisees avec le contrat de config.yaml."""
    cfg = load_config("configs/config.yaml")
    fields = CustomerFeatures.model_fields
    for col, (low, high) in cfg["schema"]["ranges"].items():
        bounds = {type(m).__name__: m for m in fields[col].metadata}
        assert bounds["Ge"].ge == low, col
        assert bounds["Le"].le == high, col
    for col, values in cfg["schema"]["allowed_values"].items():
        assert set(fields[col].annotation.__args__) == set(values), col
    assert MAX_BATCH_SIZE == cfg["api"]["max_batch_size"]
