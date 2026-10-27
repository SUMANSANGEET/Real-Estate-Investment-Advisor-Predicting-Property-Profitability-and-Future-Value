# ---- Real Estate Investment Advisor : serving image (Streamlit + tuned/calibrated XGBoost) ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MODEL_DIR=models_v2 \
    INFERENCE_LOG_PATH=/tmp/inference_log.csv

WORKDIR /app

# libgomp1 = OpenMP runtime that XGBoost needs;  curl = for the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# dependency layer first -> rebuilds are fast when only app code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY housing_utils.py real_data.py modeling_v2.py monitoring.py app_v2.py ./
COPY .streamlit ./.streamlit
COPY models_v2 ./models_v2
COPY data/market_sample.csv ./data/market_sample.csv

# run as an unprivileged user
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/_stcore/health || exit 1

# Cloud Run injects $PORT; locally it defaults to 8080
CMD ["sh", "-c", "streamlit run app_v2.py --server.port=${PORT} --server.address=0.0.0.0"]
