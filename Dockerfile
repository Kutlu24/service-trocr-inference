# ================================
# Stage 1: Builder
# ================================
FROM python:3.12-slim-bookworm AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

# Torch variant: "cpu" default, or CUDA variants like "cu128", "cu126", "cu124".
# Example:
#   docker compose build --build-arg TORCH_VARIANT=cu128
ARG TORCH_VARIANT=cpu

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Install dependencies only, not the local package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Optionally replace torch/torchvision with CUDA wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${TORCH_VARIANT}" != "cpu" ]; then \
        echo "Installing GPU torch variant: ${TORCH_VARIANT}" && \
        uv pip install \
            "torch==2.7.1+${TORCH_VARIANT}" \
            "torchvision==0.22.1+${TORCH_VARIANT}" \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
            --no-deps; \
    fi


# ================================
# Stage 2: Runtime
# ================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

# Runtime dependencies.
# libgl/libglib etc. are useful for image/PIL/OpenCV-style dependencies.
# curl is needed for Docker HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY ./src /app/src

EXPOSE 8000

# Requires /health endpoint in app.main
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for the FastAPI API container.
# The Celery worker container overrides this in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]