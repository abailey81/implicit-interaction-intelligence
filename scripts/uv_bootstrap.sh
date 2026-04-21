#!/usr/bin/env sh
# -----------------------------------------------------------------------------
# scripts/uv_bootstrap.sh
#
# POSIX-compatible bootstrap for the Astral uv toolchain on a fresh clone of
# the Implicit Interaction Intelligence (I3) repository.
#
# What it does, in order:
#   1. Installs `uv` via the Astral one-line installer (if not already present).
#   2. Ensures the pinned Python interpreter is managed by uv.
#   3. Runs `uv sync --all-extras --all-groups` to materialise the venv.
#   4. Installs the `pre-commit` hooks.
#   5. Prints a short success banner with next-step hints.
#
# Usage:
#   sh scripts/uv_bootstrap.sh
#
# This script is intentionally POSIX sh (not bash) so it runs inside the
# Chainguard Wolfi builder image and minimal Alpine containers.
# -----------------------------------------------------------------------------

set -eu

REPO_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"

log() {
    printf '\033[1;34m[uv-bootstrap]\033[0m %s\n' "$*"
}

ok() {
    printf '\033[1;32m[uv-bootstrap]\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m[uv-bootstrap]\033[0m %s\n' "$*" >&2
}

# ----------------------------------------------------------------------------
# 1. Install uv if it's missing.
# ----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found on PATH — installing via https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # The installer drops uv into ~/.local/bin on Linux/macOS.
    # Make sure this shell sees it for the rest of the script.
    if [ -d "$HOME/.local/bin" ]; then
        PATH="$HOME/.local/bin:$PATH"
        export PATH
    fi
else
    log "uv already installed: $(uv --version)"
fi

# ----------------------------------------------------------------------------
# 2. Pin the Python interpreter.
# ----------------------------------------------------------------------------
PY_VERSION="$(cat "$REPO_ROOT/.python-version" 2>/dev/null || echo '3.11')"
log "ensuring Python $PY_VERSION is available (managed by uv)"
uv python install "$PY_VERSION" || warn "could not install Python $PY_VERSION via uv — falling back to system"

# ----------------------------------------------------------------------------
# 3. Sync the project environment.
# ----------------------------------------------------------------------------
log "running: uv sync --all-extras --all-groups"
uv sync --all-extras --all-groups

# ----------------------------------------------------------------------------
# 4. Pre-commit hooks.
# ----------------------------------------------------------------------------
if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
    log "installing pre-commit hooks"
    uv run pre-commit install --install-hooks || warn "pre-commit install failed — continuing"
else
    warn "no .pre-commit-config.yaml found — skipping hook install"
fi

# ----------------------------------------------------------------------------
# 5. Success banner.
# ----------------------------------------------------------------------------
cat <<'EOF'

  +-------------------------------------------------------------+
  |                                                             |
  |   I3 :: uv bootstrap complete.                              |
  |                                                             |
  |   Next steps:                                               |
  |     uv run pytest            # run tests                    |
  |     uv run ruff check        # lint                         |
  |     uv run mkdocs serve      # docs                         |
  |     just                     # list all task recipes        |
  |                                                             |
  |   Poetry still works untouched:                             |
  |     poetry install --with dev                               |
  |                                                             |
  +-------------------------------------------------------------+

EOF

ok "done."
