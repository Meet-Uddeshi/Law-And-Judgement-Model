# ===============================================================
# Dockerfile: Backend (FastAPI + ML pipeline) — CPU-only
# Debian Trixie compatible; CRLF-safe; no GPU / no runtime FAISS
# ===============================================================

FROM python:3.11-slim AS backend

WORKDIR /app

# System dependencies:
# - libgl1: minimal GL libs (Pillow/OpenCV backends if needed)
# - poppler-utils: pdfplumber PDF support (pdftotext, etc.)
# - build-essential: wheels compile fallback
# - dos2unix: normalize CRLF if any text copied with Windows line endings
RUN apt-get update && apt-get install -y \
    libgl1 poppler-utils build-essential dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY Common ./Common
COPY Core ./Core
COPY Server ./Server

# Ensure packages are importable (handles missing __init__.py)
RUN find Core Common -type d -exec sh -c 'touch {}/__init__.py' \;

# Normalize potential CRLF line endings in py files (safe no-ops on LF)
RUN find . -type f -name "*.py" -exec dos2unix {} + || true

# Install Python deps (CPU-only; includes faiss-cpu from Core/requirements.txt)
RUN pip install --no-cache-dir \
      -r Common/requirements.txt \
      -r Core/requirements.txt \
      -r Server/requirements.txt

# Expose API port
EXPOSE 8000

# Runtime env
ENV PYTHONPATH=/app \
    API_SECRET_KEY=SuperSecureSecretKey123

# Start FastAPI directly (no entrypoint scripts)
CMD ["uvicorn", "Server.server:app", "--host", "0.0.0.0", "--port", "8000"]
