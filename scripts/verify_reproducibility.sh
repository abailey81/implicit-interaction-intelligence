#!/usr/bin/env sh
# -----------------------------------------------------------------------------
# scripts/verify_reproducibility.sh
#
# End-to-end reproducibility verification for Implicit Interaction
# Intelligence (I3). Runs three independent checks:
#
#   1. `uv lock --check`      — uv.lock matches pyproject.toml
#   2. `poetry lock --check`  — poetry.lock matches pyproject.toml
#   3. Double-build the Wolfi container and compare SHA256 digests.
#
# Exits with a non-zero status if any check fails. Prints a coloured
# summary at the end.
# -----------------------------------------------------------------------------
set -eu

REPO_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"

GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

uv_status="SKIP"
poetry_status="SKIP"
docker_status="SKIP"
uv_detail=""
poetry_detail=""
docker_detail=""
digest_a=""
digest_b=""

# ------------------------------------------------------------------
# 1. uv lock check
# ------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    printf "%s[1/3]%s uv lock --check\n" "$YELLOW" "$RESET"
    if uv lock --check 2>&1; then
        uv_status="PASS"
    else
        uv_status="FAIL"
        uv_detail="uv.lock is stale — run 'uv lock' to regenerate"
    fi
else
    uv_detail="uv not installed"
fi

# ------------------------------------------------------------------
# 2. poetry lock check
# ------------------------------------------------------------------
if command -v poetry >/dev/null 2>&1; then
    printf "%s[2/3]%s poetry lock --check\n" "$YELLOW" "$RESET"
    if poetry lock --check 2>&1; then
        poetry_status="PASS"
    else
        poetry_status="FAIL"
        poetry_detail="poetry.lock is stale — run 'poetry lock --no-update'"
    fi
else
    poetry_detail="poetry not installed"
fi

# ------------------------------------------------------------------
# 3. Docker double-build digest compare
# ------------------------------------------------------------------
if command -v docker >/dev/null 2>&1 && [ -f "$REPO_ROOT/Dockerfile.wolfi" ]; then
    printf "%s[3/3]%s double-build Dockerfile.wolfi and compare digests\n" "$YELLOW" "$RESET"

    tag_a="i3:repro-a-$$"
    tag_b="i3:repro-b-$$"

    # Build A
    docker build \
        --no-cache \
        --pull \
        -f Dockerfile.wolfi \
        -t "$tag_a" . >/tmp/repro-build-a.log 2>&1

    # Build B (no cache again — must match byte-for-byte)
    docker build \
        --no-cache \
        --pull \
        -f Dockerfile.wolfi \
        -t "$tag_b" . >/tmp/repro-build-b.log 2>&1

    digest_a="$(docker image inspect "$tag_a" --format '{{.Id}}')"
    digest_b="$(docker image inspect "$tag_b" --format '{{.Id}}')"

    if [ "$digest_a" = "$digest_b" ]; then
        docker_status="PASS"
        docker_detail="$digest_a"
    else
        docker_status="FAIL"
        docker_detail="digests differ: A=$digest_a  B=$digest_b"
    fi

    # Tidy up
    docker image rm -f "$tag_a" "$tag_b" >/dev/null 2>&1 || true
else
    docker_detail="docker or Dockerfile.wolfi unavailable"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
colour_for() {
    case "$1" in
        PASS) printf "%s" "$GREEN" ;;
        FAIL) printf "%s" "$RED" ;;
        *)    printf "%s" "$YELLOW" ;;
    esac
}

echo ""
echo "======================================================================"
echo "  I3 reproducibility verification"
echo "======================================================================"
printf "  uv lock --check         : %s%s%s   %s\n"     "$(colour_for "$uv_status")"     "$uv_status"     "$RESET" "$uv_detail"
printf "  poetry lock --check     : %s%s%s   %s\n"     "$(colour_for "$poetry_status")" "$poetry_status" "$RESET" "$poetry_detail"
printf "  docker double-build     : %s%s%s   %s\n"     "$(colour_for "$docker_status")" "$docker_status" "$RESET" "$docker_detail"
echo "======================================================================"

if [ "$uv_status" = "FAIL" ] || [ "$poetry_status" = "FAIL" ] || [ "$docker_status" = "FAIL" ]; then
    printf "%sreproducibility check FAILED%s\n" "$RED" "$RESET"
    exit 1
fi

printf "%sall reproducibility checks passed%s\n" "$GREEN" "$RESET"
exit 0
