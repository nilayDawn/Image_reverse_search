FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies (libgomp1 required for ONNX Runtime CPU multi-threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests
COPY pyproject.toml uv.lock* ./

# Install project dependencies without cache
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source and artifacts
COPY app/ ./app/
COPY artifacts/resnet50_extractor.onnx ./artifacts/
COPY artifacts/pca_1024.pkl ./artifacts/

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]