# ============================================================
# Image de l'API d'inference churn : multi-stage, legere, non-root (TP4)
# ============================================================

# ---------- Etape 1 : construction de l'environnement Python ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Un venv isole que l'on copiera tel quel dans l'image finale
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt

# ---------- Etape 2 : image d'execution minimale ----------
FROM python:3.11-slim AS runtime

LABEL maintainer="Groupe Bank-Churn-MLOps (Rjeb, Kaouech, Ayadi)"
LABEL description="Microservice FastAPI de prediction du churn bancaire"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CONFIG_PATH=configs/config.yaml

# Utilisateur non-root (UID 1000) : le conteneur ne tourne jamais en root
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/service

# Seul le venv construit est recupere (pas de compilateur, pas de cache pip)
COPY --from=builder /opt/venv /opt/venv

# Code, configuration et modele exporte (aucune donnee, aucun secret)
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/
RUN mkdir -p reports && chown appuser:appuser reports

USER appuser
EXPOSE 8000

# Sonde de sante en Python pur (pas besoin d'installer curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
