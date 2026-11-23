FROM python:3.11-slim

WORKDIR /app

# Slimmer runtime deps (no Locust / notebook stack) → faster builds, less RAM.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements-docker.txt

COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY models/ ./models/
COPY data/ ./data/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

WORKDIR /app/api

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
# Model and training data paths inside the container.
ENV BEAN_MODEL_DIR=/app/models
ENV BEAN_TRAIN_DIR=/app/data/train

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
