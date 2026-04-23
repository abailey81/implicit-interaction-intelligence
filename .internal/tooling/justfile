# justfile — command runner recipes for Implicit Interaction Intelligence (I3)
#
# This file is parallel to the existing Makefile and is the preferred entry
# point for the uv-based toolchain. Run `just` with no arguments to see the
# full list of recipes.
#
# https://just.systems/

# list all recipes when no target is given
default:
    @just --list

# -----------------------------------------------------------------------------
# core lifecycle
# -----------------------------------------------------------------------------

# sync the project environment from uv.lock (all groups + extras)
install:
    @uv sync --all-extras --all-groups

# run the full pytest suite
test:
    @uv run pytest -q

# lint the repo with ruff (check only, no autofix)
lint:
    @uv run ruff check .

# format the repo with ruff (in-place)
format:
    @uv run ruff format .

# serve the MkDocs site locally with live reload
docs:
    @uv run mkdocs serve

# run the benchmark suite and print a summary
bench:
    @uv run pytest benchmarks/ --benchmark-only --benchmark-columns=mean,min,max,stddev

# remove every derived artifact (venv, caches, build outputs)
clean:
    @rm -rf .venv .uv-cache .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info coverage.xml

# -----------------------------------------------------------------------------
# dual-lockfile maintenance during the Poetry → uv migration window
# -----------------------------------------------------------------------------

# regenerate BOTH poetry.lock and uv.lock from pyproject.toml
lock:
    @poetry lock --no-update
    @uv lock

# fail-fast check that both lockfiles are in sync with pyproject.toml
lock-check:
    @poetry lock --check
    @uv lock --check

# -----------------------------------------------------------------------------
# containers
# -----------------------------------------------------------------------------

# build the default Debian-slim production image
docker-build:
    @docker build -f Dockerfile -t i3:latest .

# build the Chainguard Wolfi zero-CVE production image
docker-build-wolfi:
    @docker build -f Dockerfile.wolfi -t i3:wolfi .

# scan the Wolfi image with trivy (high + critical only)
docker-scan-wolfi:
    @trivy image i3:wolfi --severity HIGH,CRITICAL

# -----------------------------------------------------------------------------
# reproducibility
# -----------------------------------------------------------------------------

# run the full reproducibility check (lockfiles + docker digest)
verify-repro:
    @sh scripts/verify_reproducibility.sh

# enter the Nix flake dev shell
nix-shell:
    @nix develop

# build the I3 wheel hermetically via the Nix flake
nix-build:
    @nix build .#default

# -----------------------------------------------------------------------------
# meta
# -----------------------------------------------------------------------------

# print every tool version that CI pins
versions:
    @python --version
    @uv --version
    @poetry --version
    @ruff --version
    @mypy --version 2>/dev/null || echo "mypy: not installed in current shell"
