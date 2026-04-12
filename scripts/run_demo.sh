#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  I³ Demo Launcher
#  Verifies environment, seeds demo data, and starts the FastAPI server.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
BLUE='\033[34m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENCODER_CKPT_DEFAULT="${PROJECT_ROOT}/checkpoints/encoder/best.pt"
SLM_CKPT_DEFAULT="${PROJECT_ROOT}/checkpoints/slm/best.pt"

# ── Helpers ─────────────────────────────────────────────────────────────
step() { printf "${CYAN}▶${RESET} ${BOLD}%s${RESET}\n" "$1"; }
success() { printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
warn() { printf "  ${YELLOW}⚠${RESET} %s\n" "$1"; }
fail() { printf "  ${RED}✗${RESET} %s\n" "$1" >&2; exit 1; }

# ── Banner ──────────────────────────────────────────────────────────────
echo ""
printf "${BLUE}  ╔═══════════════════════════════════════════════════════════╗${RESET}\n"
printf "${BLUE}  ║${RESET}  ${BOLD}Implicit Interaction Intelligence (I³) — Demo${RESET}          ${BLUE}║${RESET}\n"
printf "${BLUE}  ╚═══════════════════════════════════════════════════════════╝${RESET}\n"
echo ""

cd "$PROJECT_ROOT"

# ── Step 1: Verify .env exists ──────────────────────────────────────────
step "Checking environment file"
if [ ! -f "$ENV_FILE" ]; then
  fail ".env not found. Run 'bash scripts/setup.sh' first."
fi
success ".env located"

# ── Step 2: Source .env ─────────────────────────────────────────────────
step "Loading environment variables"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
success "Environment loaded"

# ── Step 3: Check for checkpoints ───────────────────────────────────────
step "Checking model checkpoints"
ENCODER_CKPT="${I3_ENCODER_CHECKPOINT:-$ENCODER_CKPT_DEFAULT}"
SLM_CKPT="${I3_SLM_CHECKPOINT:-$SLM_CKPT_DEFAULT}"

# Resolve relative paths against project root
[[ "$ENCODER_CKPT" != /* ]] && ENCODER_CKPT="${PROJECT_ROOT}/${ENCODER_CKPT}"
[[ "$SLM_CKPT" != /* ]] && SLM_CKPT="${PROJECT_ROOT}/${SLM_CKPT}"

CKPT_WARNINGS=0
if [ ! -f "$ENCODER_CKPT" ]; then
  warn "Encoder checkpoint missing: ${ENCODER_CKPT}"
  warn "  → Run 'make train-encoder' (demo will fall back to random weights)"
  CKPT_WARNINGS=$((CKPT_WARNINGS + 1))
else
  success "Encoder checkpoint found"
fi

if [ ! -f "$SLM_CKPT" ]; then
  warn "SLM checkpoint missing: ${SLM_CKPT}"
  warn "  → Run 'make train-slm' (demo will fall back to cloud-only routing)"
  CKPT_WARNINGS=$((CKPT_WARNINGS + 1))
else
  success "SLM checkpoint found"
fi

if [ "$CKPT_WARNINGS" -gt 0 ]; then
  printf "  ${DIM}(continuing with ${CKPT_WARNINGS} missing checkpoint(s))${RESET}\n"
fi

# ── Step 4: Seed demo data ──────────────────────────────────────────────
step "Seeding demo data"
if python -m demo.seed 2>/dev/null; then
  success "Demo data seeded"
else
  warn "Seed step failed or not available — continuing"
fi

# ── Step 5: Launch the server ───────────────────────────────────────────
HOST="${I3_HOST:-0.0.0.0}"
PORT="${I3_PORT:-8000}"
# Pretty URL for user (prefer localhost for clickability)
DISPLAY_HOST="$HOST"
if [ "$HOST" = "0.0.0.0" ]; then
  DISPLAY_HOST="localhost"
fi
URL="http://${DISPLAY_HOST}:${PORT}"

echo ""
printf "${GREEN}  ╔═══════════════════════════════════════════════════════════╗${RESET}\n"
printf "${GREEN}  ║${RESET}                                                           ${GREEN}║${RESET}\n"
printf "${GREEN}  ║${RESET}     ${BOLD}I³ Demo is starting…${RESET}                                ${GREEN}║${RESET}\n"
printf "${GREEN}  ║${RESET}                                                           ${GREEN}║${RESET}\n"
printf "${GREEN}  ║${RESET}     Open in browser:  ${CYAN}${BOLD}%-30s${RESET}      ${GREEN}║${RESET}\n" "$URL"
printf "${GREEN}  ║${RESET}                                                           ${GREEN}║${RESET}\n"
printf "${GREEN}  ║${RESET}     ${DIM}Press Ctrl+C to stop the server${RESET}                     ${GREEN}║${RESET}\n"
printf "${GREEN}  ║${RESET}                                                           ${GREEN}║${RESET}\n"
printf "${GREEN}  ╚═══════════════════════════════════════════════════════════╝${RESET}\n"
echo ""

exec uvicorn server.main:app --host "$HOST" --port "$PORT" --reload
