#!/usr/bin/env bash
# =============================================================================
# I3 pre-flight checklist runner
# -----------------------------------------------------------------------------
# Programmatic version of the 5-minute pre-flight list documented in
# docs/DEMO_SCRIPT.md. Run this 15 minutes before the interview to catch
# every obvious failure mode in one place.
#
# Exits non-zero on any failure so the script can be wired into CI or a
# desktop notification hook.
# =============================================================================

set -u
set -o pipefail

# -------- ANSI helpers ------------------------------------------------------
RESET='\033[0m'
BOLD='\033[1m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'

fails=0
warnings=0

print_banner() {
    printf "\n${BOLD}==============================================================${RESET}\n"
    printf "${BOLD} I3 PRE-FLIGHT CHECKLIST${RESET}\n"
    printf "${BOLD}==============================================================${RESET}\n"
}

pass() {
    printf "  ${GREEN}[ OK ]${RESET} %s\n" "$1"
}

warn() {
    printf "  ${YELLOW}[WARN]${RESET} %s\n" "$1"
    warnings=$((warnings + 1))
}

fail() {
    printf "  ${RED}[FAIL]${RESET} %s\n" "$1"
    fails=$((fails + 1))
}

# -------- Checks ------------------------------------------------------------

check_env_var() {
    # Args: $1 = env var name, $2 = human label
    local name="$1"
    local label="$2"
    local val="${!name:-}"
    if [ -z "$val" ]; then
        fail "$label ($name) is not set"
        return 1
    fi
    # Print only the first 8 chars so we never leak the full token.
    local prefix
    prefix="${val:0:8}"
    pass "$label ($name) present: ${prefix}..."
    return 0
}

check_network() {
    if command -v ping >/dev/null 2>&1; then
        if ping -c 2 -W 3 1.1.1.1 >/dev/null 2>&1; then
            pass "Network reachable (ping 1.1.1.1)"
            return 0
        fi
    fi
    fail "No network (ping 1.1.1.1 failed)"
    return 1
}

check_server() {
    local url="${I3_PREFLIGHT_URL:-http://localhost:8000/api/ready}"
    if ! command -v curl >/dev/null 2>&1; then
        fail "curl not installed — cannot check server"
        return 1
    fi
    if curl -sf --max-time 5 "$url" >/dev/null; then
        pass "Demo server responded ($url)"
        return 0
    fi
    fail "Demo server did NOT respond ($url)"
    return 1
}

check_optional_anthropic_ping() {
    # Informational only. Do NOT fail the pre-flight if the cloud is
    # unreachable — the brief's recovery playbook covers that case.
    if ! command -v curl >/dev/null 2>&1; then
        return 0
    fi
    if curl -sf --max-time 3 https://api.anthropic.com >/dev/null 2>&1; then
        pass "api.anthropic.com reachable"
    else
        warn "api.anthropic.com NOT reachable (will fall back to local)"
    fi
}

check_python_module() {
    local mod="$1"
    local label="$2"
    if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
        fail "python / python3 not installed"
        return 1
    fi
    local py
    if command -v python3 >/dev/null 2>&1; then
        py=python3
    else
        py=python
    fi
    if "$py" -c "import $mod" >/dev/null 2>&1; then
        pass "$label importable"
        return 0
    fi
    warn "$label not importable — may impact demo"
    return 1
}

# -------- Run ---------------------------------------------------------------

print_banner

check_env_var "ANTHROPIC_API_KEY" "Anthropic API key"
check_env_var "I3_ENCRYPTION_KEY" "Fernet encryption key"
check_network
check_optional_anthropic_ping
check_server
check_python_module "i3.pipeline.engine" "i3.pipeline.engine"
check_python_module "websockets"         "websockets"

printf "\n"
if [ "$fails" -gt 0 ]; then
    printf "${RED}${BOLD}PRE-FLIGHT FAILED: %d failure(s), %d warning(s)${RESET}\n" "$fails" "$warnings"
    exit 1
fi

if [ "$warnings" -gt 0 ]; then
    printf "${YELLOW}${BOLD}PRE-FLIGHT PASSED WITH WARNINGS: %d warning(s)${RESET}\n" "$warnings"
    exit 0
fi

printf "${GREEN}${BOLD}PRE-FLIGHT PASSED${RESET}\n"
exit 0
