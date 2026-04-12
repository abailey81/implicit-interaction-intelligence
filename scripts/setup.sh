#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  I³ Setup Script
#  Creates virtual environment, installs dependencies, generates key.
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
VENV_DIR="${PROJECT_ROOT}/.venv"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"

# ── Helpers ─────────────────────────────────────────────────────────────
banner() {
  echo ""
  printf "${BLUE}  ╔═══════════════════════════════════════════════════════════╗${RESET}\n"
  printf "${BLUE}  ║${RESET}  ${BOLD}Implicit Interaction Intelligence (I³) — Setup${RESET}         ${BLUE}║${RESET}\n"
  printf "${BLUE}  ╚═══════════════════════════════════════════════════════════╝${RESET}\n"
  echo ""
}

step() {
  printf "${CYAN}▶${RESET} ${BOLD}%s${RESET}\n" "$1"
}

success() {
  printf "  ${GREEN}✓${RESET} %s\n" "$1"
}

warn() {
  printf "  ${YELLOW}⚠${RESET} %s\n" "$1"
}

fail() {
  printf "  ${RED}✗${RESET} %s\n" "$1" >&2
  exit 1
}

# ── Step 1: Banner ──────────────────────────────────────────────────────
banner

# ── Step 2: Check Python version ────────────────────────────────────────
step "Checking Python version"
if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found. Please install Python 3.10 or later."
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  fail "Python 3.10+ required, found $PY_VERSION"
fi
success "Python $PY_VERSION detected"

# ── Step 3: Create virtual environment ──────────────────────────────────
step "Creating virtual environment"
if [ -d "$VENV_DIR" ]; then
  warn "Virtual environment already exists at .venv (skipping)"
else
  python3 -m venv "$VENV_DIR"
  success "Created .venv"
fi

# ── Step 4: Activate venv ───────────────────────────────────────────────
step "Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
success "Activated .venv"

# ── Step 5: Upgrade pip ─────────────────────────────────────────────────
step "Upgrading pip"
python -m pip install --quiet --upgrade pip
success "pip upgraded to $(pip --version | awk '{print $2}')"

# ── Step 6: Install package ─────────────────────────────────────────────
step "Installing i3 package (with dev dependencies)"
printf "  ${DIM}(this may take a few minutes)${RESET}\n"
cd "$PROJECT_ROOT"
pip install --quiet -e ".[dev]" || fail "pip install failed"
success "Package installed in editable mode"

# ── Step 7: Create .env from template ───────────────────────────────────
step "Configuring environment variables"
if [ -f "$ENV_FILE" ]; then
  warn ".env already exists (skipping copy)"
else
  if [ ! -f "$ENV_EXAMPLE" ]; then
    fail ".env.example not found at $ENV_EXAMPLE"
  fi
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  success "Copied .env.example → .env"
fi

# ── Step 8: Generate Fernet key ─────────────────────────────────────────
step "Generating Fernet encryption key"
if grep -q '^I3_ENCRYPTION_KEY=.\+' "$ENV_FILE" 2>/dev/null; then
  warn "I3_ENCRYPTION_KEY already set in .env (skipping)"
else
  FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' 2>/dev/null || echo "")
  if [ -z "$FERNET_KEY" ]; then
    warn "Could not generate Fernet key (cryptography not installed yet?)"
    warn "Run 'python scripts/generate_encryption_key.py --update-env' later"
  else
    printf "  ${DIM}Generated key:${RESET} ${CYAN}%s${RESET}\n" "$FERNET_KEY"
    printf "  ${YELLOW}?${RESET} Add this key to .env? [Y/n] "
    read -r answer
    answer="${answer:-Y}"
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      # Portable sed in-place replacement
      if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^I3_ENCRYPTION_KEY=.*|I3_ENCRYPTION_KEY=${FERNET_KEY}|" "$ENV_FILE"
      else
        sed -i "s|^I3_ENCRYPTION_KEY=.*|I3_ENCRYPTION_KEY=${FERNET_KEY}|" "$ENV_FILE"
      fi
      success "Wrote I3_ENCRYPTION_KEY to .env"
    else
      warn "Skipped — add the key manually to .env"
    fi
  fi
fi

# ── Step 9: Create required directories ─────────────────────────────────
step "Creating required directories"
DIRS=(
  "data/raw"
  "data/processed"
  "data/synthetic"
  "checkpoints/encoder"
  "checkpoints/slm"
)
for dir in "${DIRS[@]}"; do
  mkdir -p "${PROJECT_ROOT}/${dir}"
  touch "${PROJECT_ROOT}/${dir}/.gitkeep"
done
success "Created ${#DIRS[@]} directories"

# ── Step 10: Success banner ─────────────────────────────────────────────
echo ""
printf "${GREEN}  ╔═══════════════════════════════════════════════════════════╗${RESET}\n"
printf "${GREEN}  ║${RESET}  ${BOLD}✓ Setup complete!${RESET}                                         ${GREEN}║${RESET}\n"
printf "${GREEN}  ╚═══════════════════════════════════════════════════════════╝${RESET}\n"
echo ""
printf "${BOLD}  Next steps:${RESET}\n"
echo ""
printf "    ${DIM}1.${RESET} Activate the venv:       ${CYAN}source .venv/bin/activate${RESET}\n"
printf "    ${DIM}2.${RESET} Set your API key in:     ${CYAN}.env${RESET}\n"
printf "    ${DIM}3.${RESET} Generate synthetic data: ${CYAN}make generate-data${RESET}\n"
printf "    ${DIM}4.${RESET} Train the models:        ${CYAN}make train-all${RESET}\n"
printf "    ${DIM}5.${RESET} Run the demo:            ${CYAN}make demo${RESET}\n"
echo ""
printf "  See ${CYAN}make help${RESET} for all available commands.\n"
echo ""
