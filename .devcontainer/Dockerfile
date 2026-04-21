# syntax=docker/dockerfile:1.7
# =============================================================================
#  .devcontainer/Dockerfile  —  VSCode dev container image
# -----------------------------------------------------------------------------
#  Based on the Microsoft Python 3.11 dev-container image, which ships with a
#  pre-configured `vscode` user, shell niceties, and sensible defaults. On top
#  of that we add:
#
#     * OS-level build toolchain (for any C extensions rebuilt from source)
#     * `tini` for clean signal handling during long-running dev sessions
#     * `libgomp1` so torch's OpenMP kernels work
#
#  Poetry + the project dependencies are installed by `post-create.sh` so that
#  the in-project `.venv` ends up on a named docker volume (fast rebuilds).
# =============================================================================

ARG PYTHON_IMAGE=mcr.microsoft.com/devcontainers/python:3.11-bookworm

FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Additional packages we want available during dev sessions.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        tini \
        libgomp1 \
        libffi-dev \
        libssl-dev \
        make \
        jq \
        unzip \
        less \
        htop \
    && rm -rf /var/lib/apt/lists/*

# The `vscode` user already exists in the base image with uid 1000. We simply
# make sure the cache directories we named-volume-mount in devcontainer.json
# are owned by them.
RUN install -d -o vscode -g vscode \
        /home/vscode/.cache/pip \
        /home/vscode/.cache/pypoetry \
        /home/vscode/.cache/pre-commit

USER vscode
