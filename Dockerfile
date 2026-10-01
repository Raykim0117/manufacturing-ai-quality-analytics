FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/selected-model-v1/selected.pkl \
    APP_ENV=local \
    PERSIST_PREDICTIONS=false \
    PORT=8080

WORKDIR /app

# XGBoost's CPU wheel requires the OpenMP runtime; no compiler is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Explicit copies keep training, evaluation, SHAP, and local data out of the image.
COPY src/quality_analytics/api.py \
     src/quality_analytics/config.py \
     src/quality_analytics/predict.py \
     src/quality_analytics/preprocessing.py \
     src/quality_analytics/warehouse.py \
     ./src/quality_analytics/
COPY artifacts/secom-01b3bed261fc1223/selected-model-v1/selected.pkl \
     ./artifacts/selected-model-v1/selected.pkl

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + (os.getenv('PORT') or '8080') + '/health', timeout=3).close()"]

# The shell expands PORT; exec makes Uvicorn receive shutdown signals directly.
CMD ["sh", "-c", "exec uvicorn quality_analytics.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
