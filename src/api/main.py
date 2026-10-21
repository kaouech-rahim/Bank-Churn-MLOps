"""API FastAPI de prediction du churn bancaire (TP4).

Le modele est charge une seule fois au demarrage (lifespan). S'il est absent,
l'API demarre quand meme mais /health renvoie 503 (le conteneur est "unhealthy")
et les endpoints de prediction renvoient 503.

Chemins configurables par variables d'environnement (valeurs par defaut : config) :
CONFIG_PATH, MODEL_PATH, METADATA_PATH, SCALER_PATH, DRIFT_REPORT_PATH.
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.model_service import ModelService
from src.api.schemas import (
    BatchCustomerRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import load_config
from src.monitoring import prometheus_metrics as prom


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Cycle de vie : chargement unique du modele au demarrage."""
    cfg = load_config(os.getenv("CONFIG_PATH", "configs/config.yaml"))
    models_cfg = cfg["models"]
    app.state.cfg = cfg
    app.state.drift_report_path = os.getenv(
        "DRIFT_REPORT_PATH", cfg["monitoring"]["drift_report_json"]
    )
    app.state.model_service = None
    print("[INIT] Chargement du modele de churn...")
    try:
        service = ModelService(
            model_path=os.getenv("MODEL_PATH", models_cfg["model_path"]),
            metadata_path=os.getenv("METADATA_PATH", models_cfg["metadata_path"]),
            scaler_path=os.getenv("SCALER_PATH", models_cfg["scaler_path"]),
            cfg=cfg,
        )
        app.state.model_service = service
        prom.set_model_info(service.metadata)
        print(f"[OK] Modele charge (version {service.version})")
    except FileNotFoundError as err:
        # On ne plante pas : /health signalera 503 tant que le modele manque
        print(f"[ERREUR] Modele non charge : {err}")
    yield
    print("[SHUTDOWN] Liberation des ressources.")


app = FastAPI(
    title="Bank Churn Inference API",
    description=(
        "Microservice REST de prediction du depart des clients bancaires.\n\n"
        "* **/health** : sonde de sante Docker\n"
        "* **/info** : metadonnees et metriques du modele\n"
        "* **/predict** : prediction unitaire\n"
        "* **/predict/batch** : prediction par lot (1 000 clients max)\n"
        "* **/metrics** : metriques Prometheus"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ModelService:
    """Renvoie le service d'inference ou leve une erreur 503 s'il n'est pas pret."""
    service: Optional[ModelService] = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modele n'est pas pret.",
        )
    return service


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(request: Request) -> HealthResponse:
    """Sonde de sante : 200 si le modele est charge, 503 sinon."""
    service = get_service(request)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=service.version,
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Gouvernance"])
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Metadonnees du modele : version, algorithme, metriques, features."""
    meta = get_service(request).metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "BankChurnModel"),
        version=str(meta.get("version", "inconnue")),
        algorithm=meta.get("algorithm", "inconnu"),
        run_name=meta.get("run_name", "inconnu"),
        decision_threshold=meta.get("decision_threshold", 0.5),
        metrics=meta.get("metrics", {}),
        features=meta.get("features", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(
    customer: CustomerFeatures, request: Request
) -> PredictionResponse:
    """Prediction pour un client (payload invalide -> 422 automatique)."""
    service = get_service(request)
    start = time.perf_counter()
    try:
        result = service.predict_single(customer)
    except Exception as err:  # erreur inattendue du modele : 500 explicite
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'inference : {err}",
        )
    prom.LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
    prom.PREDICTIONS.labels(endpoint="predict").inc()
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    payload: BatchCustomerRequest, request: Request
) -> BatchPredictionResponse:
    """Prediction vectorisee pour une liste de clients (1 a 1 000)."""
    service = get_service(request)
    start = time.perf_counter()
    try:
        predictions, avg_proba = service.predict_batch(payload.customers)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur batch : {err}",
        )
    prom.LATENCY.labels(endpoint="predict_batch").observe(time.perf_counter() - start)
    prom.PREDICTIONS.labels(endpoint="predict_batch").inc(len(predictions))
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        average_churn_probability=avg_proba,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics(request: Request) -> Response:
    """Metriques au format texte Prometheus (drift relu a chaque collecte)."""
    path = getattr(request.app.state, "drift_report_path", None)
    if path:
        prom.update_drift_metrics(path)
    return Response(generate_latest(prom.REGISTRY), media_type=CONTENT_TYPE_LATEST)
