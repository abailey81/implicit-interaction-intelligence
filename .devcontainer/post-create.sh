#!/usr/bin/env bash
# =============================================================================
#  .devcontainer/post-create.sh
# -----------------------------------------------------------------------------
#  Runs exactly once, on first container creation (via devcontainer.json's
#  `onCreateCommand`). Keep it idempotent — it may re-run if the workspace is
#  rebuilt.
# =============================================================================
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspaces/implicit-interaction-intelligence}"
POETRY_VERSION="${POETRY_VERSION:-1.8.3}"

cd "${WORKSPACE_DIR}"

echo "────────────────────────────────────────────────────────────────────"
echo "  I³ dev container  —  post-create bootstrap"
echo "  workspace : ${WORKSPACE_DIR}"
echo "  poetry    : ${POETRY_VERSION}"
echo "────────────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
#  Make sure git trusts the bind-mounted repo (otherwise git refuses to
#  operate because the uid of the repo on disk differs from uid 1000).
# ---------------------------------------------------------------------------
git config --global --add safe.directory "${WORKSPACE_DIR}" || true

# ---------------------------------------------------------------------------
#  Install Poetry into the user site so we don't clash with the system python.
# ---------------------------------------------------------------------------
if ! command -v poetry >/dev/null 2>&1; then
    python -m pip install --user "poetry==${POETRY_VERSION}"
fi
export PATH="${HOME}/.local/bin:${PATH}"

poetry config virtualenvs.in-project true
poetry config virtualenvs.create true

# ---------------------------------------------------------------------------
#  Install CPU-only torch first so Poetry does not pull the ~800 MiB CUDA
#  wheel when it resolves the dependency graph. This mirrors the prod image.
# ---------------------------------------------------------------------------
if [ ! -d "${WORKSPACE_DIR}/.venv" ]; then
    python -m venv "${WORKSPACE_DIR}/.venv"
fi
# shellcheck disable=SC1091
. "${WORKSPACE_DIR}/.venv/bin/activate"

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            "torch>=2.6,<3.0"

# ---------------------------------------------------------------------------
#  Install the full dev/docs/security toolchain.
# ---------------------------------------------------------------------------
poetry install --with dev,docs,security

# ---------------------------------------------------------------------------
#  Pre-commit hooks.
# ---------------------------------------------------------------------------
if [ -f "${WORKSPACE_DIR}/.pre-commit-config.yaml" ]; then
    poetry run pre-commit install --install-hooks || true
fi

# ---------------------------------------------------------------------------
#  Mutable working dirs.
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE_DIR}/data" \
         "${WORKSPACE_DIR}/checkpoints" \
         "${WORKSPACE_DIR}/logs"

echo
echo "Dev container bootstrap complete. Run:  poetry run i3-serve"
