#!/usr/bin/env bash
set -e

echo "[INFO] Checking for GPU availability..."

# Detect GPU at runtime
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "[INFO] GPU detected -> Installing faiss-gpu..."
    pip install --no-cache-dir faiss-gpu==1.8.0 || true
else
    echo "[INFO] No GPU detected -> Installing faiss-cpu..."
    pip install --no-cache-dir faiss-cpu==1.8.0 || true
fi

echo "[INFO] Starting FastAPI backend..."
exec uvicorn Server.server:app --host 0.0.0.0 --port 8000
