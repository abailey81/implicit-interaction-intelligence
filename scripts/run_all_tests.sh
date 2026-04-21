#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#
#   run_all_tests.sh — run every CI gate in order, fail fast
#
#   Runs:  lint  ->  type  ->  unit  ->  property  ->  contract
#                                  ->  integration  ->  security
#
#   Excludes:
#     - load tests (tests/load/)         — opt-in, much slower
#     - fuzz harnesses (tests/fuzz/)     — run separately under atheris
#     - mutation tests                    — run separately under mutmut
#     - benchmarks                        — run separately under /benchmarks/
#
# ════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Colours (only if stdout is a TTY) ─────────────────────────────────
if [ -t 1 ]; then
    _RED=$'\033[0;31m'
    _GREEN=$'\033[0;32m'
    _YELLOW=$'\033[0;33m'
    _BLUE=$'\033[0;34m'
    _BOLD=$'\033[1m'
    _DIM=$'\033[2m'
    _RESET=$'\033[0m'
else
    _RED=""; _GREEN=""; _YELLOW=""; _BLUE=""; _BOLD=""; _DIM=""; _RESET=""
fi

banner() {
    local title="$1"
    local line="────────────────────────────────────────────────────────────────"
    printf "\n%s%s%s\n" "${_BLUE}${_BOLD}" "${line}" "${_RESET}"
    printf "%s  %s%s\n"  "${_BLUE}${_BOLD}" "${title}" "${_RESET}"
    printf "%s%s%s\n\n"  "${_BLUE}${_BOLD}" "${line}" "${_RESET}"
}

ok()   { printf "%s[OK]%s    %s\n" "${_GREEN}" "${_RESET}" "$1"; }
fail() { printf "%s[FAIL]%s  %s\n" "${_RED}"   "${_RESET}" "$1"; }
info() { printf "%s[..]%s    %s\n" "${_DIM}"   "${_RESET}" "$1"; }

# ─── Runner discovery ──────────────────────────────────────────────────
# Prefer poetry if available; fall back to whatever python is on PATH.
if command -v poetry >/dev/null 2>&1; then
    RUN="poetry run"
else
    RUN=""
    printf "%s%s%s\n" "${_YELLOW}" "Warning: poetry not found; using bare python." "${_RESET}"
fi

# ─── Gate execution helpers ────────────────────────────────────────────
START_TIME=$(date +%s)
STEP_COUNT=0

run_step() {
    local title="$1"; shift
    STEP_COUNT=$((STEP_COUNT + 1))
    banner "[$STEP_COUNT] ${title}"
    info "running: $*"
    if "$@"; then
        ok "${title}"
    else
        fail "${title}"
        exit 1
    fi
}

# ─── Gates ─────────────────────────────────────────────────────────────

# 1. Lint (ruff)
if command -v ruff >/dev/null 2>&1 || [ -n "$RUN" ]; then
    run_step "lint (ruff)" $RUN ruff check i3 server tests
else
    printf "%sskipping lint — ruff not installed%s\n" "${_YELLOW}" "${_RESET}"
fi

# 2. Type-check (mypy)
if command -v mypy >/dev/null 2>&1 || [ -n "$RUN" ]; then
    run_step "type-check (mypy)" $RUN mypy i3 server
else
    printf "%sskipping type-check — mypy not installed%s\n" "${_YELLOW}" "${_RESET}"
fi

# 3. Unit tests (exclude slow / load / integration so this stays hot-path)
run_step "unit tests" \
    $RUN pytest tests/ \
        --ignore=tests/property \
        --ignore=tests/contract \
        --ignore=tests/fuzz \
        --ignore=tests/load \
        --ignore=tests/chaos \
        --ignore=tests/snapshot \
        --ignore=tests/benchmarks \
        -m "not slow and not load and not integration"

# 4. Property tests (Hypothesis)
run_step "property tests" $RUN pytest tests/property/ -q

# 5. Contract tests (schemathesis, WS protocol)
run_step "contract tests" $RUN pytest tests/contract/ -q

# 6. Snapshot tests (syrupy)
run_step "snapshot tests" $RUN pytest tests/snapshot/ -q

# 7. Chaos tests (failure injection)
run_step "chaos tests" $RUN pytest tests/chaos/ -q

# 8. Integration tests (marker-based, multi-subsystem)
run_step "integration tests" $RUN pytest tests/ -m integration -q

# 9. Security tests (marker-based)
run_step "security tests" $RUN pytest tests/ -m security -q

# ─── Summary ───────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

banner "ALL GATES PASSED"
printf "%s%s%s  %d gate(s) in %ds\n" "${_GREEN}${_BOLD}" "[OK]" "${_RESET}" "$STEP_COUNT" "$ELAPSED"
