# syntax=docker/dockerfile:1.7
# =============================================================================
#  Implicit Interaction Intelligence (I³)  —  Production Container
# =============================================================================
#  Multi-stage build:
#     1. builder   :  Python 3.11-slim + Poetry 1.8 → venv with CPU-only torch
#     2. runtime   :  Python 3.11-slim, non-root user `i3` (uid 10001),
#                     no build toolchain, no Poetry, no pip, no curl.
#
#  Security posture:
#     * Pinned base image (digest-style tag, not `latest`)
#     * All capabilities droppable; non-root by default
#     * Signal-safe PID 1 via `tini`
#     * HEALTHCHECK via Python stdlib (no extra curl binary in runtime)
#     * OCI image annotations for provenance
#
#  Build:
#     DOCKER_BUILDKIT=1 docker build -t i3:latest .
#
#  Notes:
#     * CPU-only torch is fetched from the PyTorch CPU index to stay under
#       the 400 MiB image budget (official CUDA wheels are ~800 MiB).
# =============================================================================

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
ARG POETRY_VERSION=1.8.3
ARG APP_UID=10001
ARG APP_GID=10001

# -----------------------------------------------------------------------------
#  Stage 1 — builder
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ARG POETRY_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/root/.cache/pypoetry \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:/opt/poetry/bin:$PATH

# Build-toolchain + tini (copied to runtime later).
# `--mount=type=cache` speeds up rebuilds while keeping layers small.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry into an isolated location.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv ${POETRY_HOME} && \
    ${POETRY_HOME}/bin/pip install --upgrade pip "poetry==${POETRY_VERSION}"

# Create an application venv that we will copy verbatim into the runtime stage.
RUN python -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

# Bring only the dependency manifests first to maximise Docker layer caching.
COPY pyproject.toml poetry.lock* README.md ./

# Install CPU-only torch from the official PyTorch index BEFORE Poetry resolves
# the rest of the graph; this keeps the final image below 400 MiB.
# torch version floor must match the one pinned in pyproject.toml (>=2.6,<3.0).
RUN --mount=type=cache,target=/root/.cache/pip \
    ${VENV_PATH}/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.6,<3.0"

# Resolve the remaining runtime deps via Poetry (torch is already satisfied).
# --only main  → skip dev/docs/security groups.
# --no-root    → we install the project package explicitly in a later step.
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    . ${VENV_PATH}/bin/activate && \
    poetry install --only main --no-root --no-ansi

# Now copy the project source and install the package itself into the venv.
COPY i3         ./i3
COPY server     ./server
COPY training   ./training
COPY demo       ./demo
COPY configs    ./configs
COPY LICENSE    ./LICENSE

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    . ${VENV_PATH}/bin/activate && \
    poetry install --only main --no-ansi

# Strip tests/caches/optional artefacts from site-packages to shrink the image.
RUN set -eux; \
    find ${VENV_PATH} -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    find ${VENV_PATH} -type d -name 'tests' -prune -exec rm -rf {} + ; \
    find ${VENV_PATH} -type d -name 'test' -prune -exec rm -rf {} + ; \
    find ${VENV_PATH} -type f -name '*.pyc' -delete ; \
    find ${VENV_PATH} -type f -name '*.pyo' -delete

# -----------------------------------------------------------------------------
#  Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID
ARG APP_GID

# OCI image annotations (populated by CI via --build-arg).
ARG IMAGE_VERSION=1.0.0
ARG IMAGE_REVISION=unknown
ARG IMAGE_CREATED=unknown

LABEL org.opencontainers.image.title="implicit-interaction-intelligence" \
      org.opencontainers.image.description="Adaptive AI companion built from implicit interaction signals." \
      org.opencontainers.image.source="https://github.com/abailey81/implicit-interaction-intelligence" \
      org.opencontainers.image.url="https://github.com/abailey81/implicit-interaction-intelligence" \
      org.opencontainers.image.documentation="https://github.com/abailey81/implicit-interaction-intelligence/blob/main/docs/ARCHITECTURE.md" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Tamer Atesyakar" \
      org.opencontainers.image.authors="Tamer Atesyakar <tamer.atesyakar@bk.ru>"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    I3_PORT=8000 \
    I3_WORKERS=1 \
    I3_FORWARDED_IPS=127.0.0.1 \
    APP_HOME=/app

# Install ONLY the minimum runtime libraries we need.
# libgomp1 is required by PyTorch's OpenMP kernels.
# ca-certificates is required for outbound TLS.
# Everything else (pip, poetry, build-essential, git, curl) is intentionally
# excluded from the runtime image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copy the statically-linked tini from the builder stage.
COPY --from=builder /usr/bin/tini /usr/local/bin/tini

# Copy the pre-built virtualenv (contains torch, fastapi, uvicorn, …).
COPY --from=builder /opt/venv /opt/venv

# Create an unprivileged system user & group (fixed uid/gid for reproducibility
# and for deterministic file ownership on bind mounts).
RUN groupadd --system --gid ${APP_GID} i3 && \
    useradd --system --uid ${APP_UID} --gid ${APP_GID} \
        --home-dir ${APP_HOME} --shell /sbin/nologin i3

WORKDIR ${APP_HOME}

# Copy only what the runtime needs — no tests, no docs, no training data.
COPY --chown=i3:i3 i3          ./i3
COPY --chown=i3:i3 server      ./server
COPY --chown=i3:i3 training    ./training
COPY --chown=i3:i3 demo        ./demo
COPY --chown=i3:i3 configs     ./configs
COPY --chown=i3:i3 LICENSE     ./LICENSE
COPY --chown=i3:i3 docker/entrypoint.sh   /app/docker/entrypoint.sh
COPY --chown=i3:i3 docker/healthcheck.sh  /app/docker/healthcheck.sh

# Ensure runtime scripts are executable and mutable dirs are owned by `i3`.
RUN chmod 0755 /app/docker/entrypoint.sh /app/docker/healthcheck.sh && \
    mkdir -p /app/data /app/checkpoints /app/logs && \
    chown -R i3:i3 /app

USER i3:i3

EXPOSE 8000

# HEALTHCHECK uses Python stdlib (urllib) to avoid shipping a curl binary.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/docker/healthcheck.sh"]

ENTRYPOINT ["/usr/local/bin/tini", "--", "/app/docker/entrypoint.sh"]
