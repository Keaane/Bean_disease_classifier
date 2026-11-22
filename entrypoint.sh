#!/bin/sh
set -e

PORT="${PORT:-8000}"

echo "Starting API on 0.0.0.0:${PORT} (model_dir=${BEAN_MODEL_DIR:-/app/models})"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 300
