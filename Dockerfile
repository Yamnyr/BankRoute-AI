# BankRoute AI Dockerfile de Déploiement en Production

FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_ARTIFACT_DIR=/app/model_artifact \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copie du code source et des artefacts de modèle
COPY app.py .
COPY test_api.py .
COPY model_artifact/ ./model_artifact/

# Exposition du port HTTP
EXPOSE 8000

# Healthcheck Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lancement du service FastAPI avec Uvicorn (port dynamique pour compatibilité Render/HuggingFace/Local)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
