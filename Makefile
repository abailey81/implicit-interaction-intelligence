# ═════════════════════════════════════════════════════════════════════════
#
#   Implicit Interaction Intelligence (I³) — Build System
#
#   Professional Makefile with colored output and self-documenting targets.
#   Uses Poetry for dependency management and virtual environments.
#
# ═════════════════════════════════════════════════════════════════════════

# ── ANSI color codes ─────────────────────────────────────────────────────
BLUE   := \033[34m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
CYAN   := \033[36m
MAGENTA:= \033[35m
RESET  := \033[0m
BOLD   := \033[1m
DIM    := \033[2m

# ── Project configuration ────────────────────────────────────────────────
POETRY      := poetry
PY          := $(POETRY) run python
PYTEST      := $(POETRY) run pytest
RUFF        := $(POETRY) run ruff
MYPY        := $(POETRY) run mypy
BANDIT      := $(POETRY) run bandit
UVICORN     := $(POETRY) run uvicorn

PKG         := i3
SRC_DIRS    := i3 server training demo tests
LINT_DIRS   := i3 server training tests
DATA_DIRS   := data/raw data/processed data/synthetic
CKPT_DIRS   := checkpoints/encoder checkpoints/slm

.DEFAULT_GOAL := help

# ═════════════════════════════════════════════════════════════════════════
#  Help
# ═════════════════════════════════════════════════════════════════════════

.PHONY: help
help: ## Show this help message
	@echo ""
	@printf "$(BOLD)$(BLUE)  ╭────────────────────────────────────────────────────────────────╮$(RESET)\n"
	@printf "$(BOLD)$(BLUE)  │                                                                │$(RESET)\n"
	@printf "$(BOLD)$(BLUE)  │    Implicit Interaction Intelligence — Build System           │$(RESET)\n"
	@printf "$(BOLD)$(BLUE)  │                                                                │$(RESET)\n"
	@printf "$(BOLD)$(BLUE)  ╰────────────────────────────────────────────────────────────────╯$(RESET)\n"
	@echo ""
	@printf "$(BOLD)  Usage:$(RESET) make $(CYAN)<target>$(RESET)\n"
	@echo ""
	@printf "$(BOLD)  Environment:$(RESET) $(DIM)(managed by Poetry)$(RESET)\n"
	@echo ""
	@printf "$(BOLD)  ── Setup ───────────────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(install|install-.*|setup|clean.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@printf "$(BOLD)  ── Code Quality ────────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(lint|format|typecheck|check|security.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@printf "$(BOLD)  ── Testing ─────────────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(test.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@printf "$(BOLD)  ── Training & Data ─────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(generate-data|train.*|evaluate|profile):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@printf "$(BOLD)  ── Server & Demo ───────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(serve.*|demo.*|seed.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@printf "$(BOLD)  ── Build & Release ─────────────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(build|publish|docker.*|release.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ═════════════════════════════════════════════════════════════════════════
#  Setup & Installation
# ═════════════════════════════════════════════════════════════════════════

.PHONY: install install-prod install-all setup

install: ## Install package with main + dev dependencies
	@printf "$(BLUE)▶ Installing $(PKG) with Poetry (main + dev)...$(RESET)\n"
	$(POETRY) install --with dev
	@printf "$(GREEN)✓ Install complete$(RESET)\n"

install-prod: ## Install package without dev dependencies
	@printf "$(BLUE)▶ Installing $(PKG) (production)...$(RESET)\n"
	$(POETRY) install --only main
	@printf "$(GREEN)✓ Production install complete$(RESET)\n"

install-all: ## Install with all optional groups (dev + security + docs)
	@printf "$(BLUE)▶ Installing $(PKG) with all groups...$(RESET)\n"
	$(POETRY) install --with dev,security,docs
	@printf "$(GREEN)✓ Full install complete$(RESET)\n"

setup: install ## Alias for install
	@printf "$(GREEN)✓ Project is ready$(RESET)\n"

# ═════════════════════════════════════════════════════════════════════════
#  Cleaning
# ═════════════════════════════════════════════════════════════════════════

.PHONY: clean clean-data clean-checkpoints clean-all

clean: ## Remove build artifacts, caches, and compiled files
	@printf "$(YELLOW)▶ Cleaning build artifacts and caches...$(RESET)\n"
	@rm -rf build/ dist/ *.egg-info .eggs/
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov/ .tox/ .hypothesis/
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
	@printf "$(GREEN)✓ Clean complete$(RESET)\n"

clean-data: ## Remove generated synthetic data and processed datasets
	@printf "$(YELLOW)▶ Removing generated data...$(RESET)\n"
	@find data/synthetic data/processed -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	@printf "$(GREEN)✓ Data cleaned$(RESET)\n"

clean-checkpoints: ## Remove model checkpoints (with confirmation)
	@printf "$(RED)$(BOLD)⚠  This will delete all model checkpoints.$(RESET)\n"
	@printf "$(YELLOW)Are you sure? [y/N] $(RESET)" && read ans && [ $${ans:-N} = y ] && \
		find checkpoints -type f \( -name '*.pt' -o -name '*.pth' \) -delete 2>/dev/null || \
		printf "$(DIM)  (cancelled)$(RESET)\n"
	@printf "$(GREEN)✓ Checkpoints removed$(RESET)\n"

clean-all: clean clean-data ## Clean everything except checkpoints
	@printf "$(GREEN)✓ Full clean complete$(RESET)\n"

# ═════════════════════════════════════════════════════════════════════════
#  Code Quality
# ═════════════════════════════════════════════════════════════════════════

.PHONY: lint format typecheck check security-check audit

lint: ## Run ruff linting
	@printf "$(BLUE)▶ Running ruff lint...$(RESET)\n"
	$(RUFF) check $(LINT_DIRS)
	@printf "$(GREEN)✓ Lint passed$(RESET)\n"

format: ## Format code with ruff (autofix)
	@printf "$(BLUE)▶ Formatting code with ruff...$(RESET)\n"
	$(RUFF) format $(LINT_DIRS)
	$(RUFF) check --fix $(LINT_DIRS)
	@printf "$(GREEN)✓ Format complete$(RESET)\n"

typecheck: ## Run mypy type checker
	@printf "$(BLUE)▶ Running mypy...$(RESET)\n"
	$(MYPY) $(PKG)
	@printf "$(GREEN)✓ Type check passed$(RESET)\n"

check: lint typecheck test ## Run lint + typecheck + test (pre-commit gate)
	@printf "$(BOLD)$(GREEN)✓ All checks passed$(RESET)\n"

security-check: ## Run all security scanners (bandit + pip-audit)
	@printf "$(BLUE)▶ Running security scanners...$(RESET)\n"
	@printf "$(YELLOW)  → Bandit (static analysis)$(RESET)\n"
	@$(POETRY) run bandit -r i3/ server/ -ll -q || printf "$(RED)Bandit found issues$(RESET)\n"
	@printf "$(YELLOW)  → pip-audit (dependency vulnerabilities)$(RESET)\n"
	@$(POETRY) run pip-audit || printf "$(RED)pip-audit found issues$(RESET)\n"
	@printf "$(YELLOW)  → safety (CVE database)$(RESET)\n"
	@$(POETRY) run safety check || printf "$(RED)safety found issues$(RESET)\n"
	@printf "$(GREEN)✓ Security check complete$(RESET)\n"

audit: security-check ## Alias for security-check

# ═════════════════════════════════════════════════════════════════════════
#  Testing
# ═════════════════════════════════════════════════════════════════════════

.PHONY: test test-cov test-fast test-security test-parallel

test: ## Run pytest with verbose output
	@printf "$(BLUE)▶ Running test suite...$(RESET)\n"
	$(PYTEST) -v tests/
	@printf "$(GREEN)✓ Tests passed$(RESET)\n"

test-cov: ## Run pytest with coverage report
	@printf "$(BLUE)▶ Running tests with coverage...$(RESET)\n"
	$(PYTEST) --cov=$(PKG) --cov=server --cov-report=term-missing --cov-report=html --cov-report=xml tests/
	@printf "$(GREEN)✓ Coverage report written to htmlcov/index.html$(RESET)\n"

test-fast: ## Run only fast tests (skip slow markers)
	@printf "$(BLUE)▶ Running fast tests...$(RESET)\n"
	$(PYTEST) -v -m "not slow" tests/
	@printf "$(GREEN)✓ Fast tests passed$(RESET)\n"

test-security: ## Run only security-focused tests
	@printf "$(BLUE)▶ Running security tests...$(RESET)\n"
	$(PYTEST) -v -m security tests/ || $(PYTEST) -v tests/test_security.py
	@printf "$(GREEN)✓ Security tests passed$(RESET)\n"

test-parallel: ## Run tests in parallel with pytest-xdist
	@printf "$(BLUE)▶ Running tests in parallel...$(RESET)\n"
	$(PYTEST) -n auto tests/
	@printf "$(GREEN)✓ Parallel tests complete$(RESET)\n"

# ═════════════════════════════════════════════════════════════════════════
#  Data & Training
# ═════════════════════════════════════════════════════════════════════════

.PHONY: generate-data train-encoder train-slm train-all evaluate profile

generate-data: ## Generate synthetic training data
	@printf "$(BLUE)▶ Generating synthetic data...$(RESET)\n"
	$(PY) -m training.generate_synthetic --config configs/default.yaml
	@printf "$(GREEN)✓ Data generation complete$(RESET)\n"

train-encoder: ## Train the TCN encoder
	@printf "$(BLUE)▶ Training TCN encoder...$(RESET)\n"
	$(PY) -m training.train_encoder --config configs/default.yaml
	@printf "$(GREEN)✓ Encoder training complete$(RESET)\n"

train-slm: ## Train the SLM
	@printf "$(BLUE)▶ Training SLM...$(RESET)\n"
	$(PY) -m training.train_slm --config configs/default.yaml
	@printf "$(GREEN)✓ SLM training complete$(RESET)\n"

train-all: generate-data train-encoder train-slm ## Run full training pipeline
	@printf "$(BOLD)$(GREEN)✓ Full training pipeline complete$(RESET)\n"

evaluate: ## Run evaluation suite (perplexity, conditioning sensitivity)
	@printf "$(BLUE)▶ Running evaluation...$(RESET)\n"
	$(PY) -m training.evaluate --config configs/default.yaml
	@printf "$(GREEN)✓ Evaluation complete$(RESET)\n"

profile: ## Run edge-feasibility profiling
	@printf "$(BLUE)▶ Profiling edge feasibility...$(RESET)\n"
	$(PY) -c "from i3.profiling.report import EdgeProfiler; EdgeProfiler().profile_full_system(None, None, None, None)"
	@printf "$(GREEN)✓ Profile complete$(RESET)\n"

# ═════════════════════════════════════════════════════════════════════════
#  Server & Demo
# ═════════════════════════════════════════════════════════════════════════

.PHONY: seed-demo serve serve-dev demo

seed-demo: ## Seed the database with demo data
	@printf "$(BLUE)▶ Seeding demo data...$(RESET)\n"
	$(PY) -m demo.seed_data
	@printf "$(GREEN)✓ Demo data seeded$(RESET)\n"

serve: ## Run the FastAPI server in production mode
	@printf "$(BLUE)▶ Starting server (production)...$(RESET)\n"
	$(UVICORN) server.app:app --host 0.0.0.0 --port 8000

serve-dev: ## Run the FastAPI server with hot reload
	@printf "$(BLUE)▶ Starting server (dev, hot reload)...$(RESET)\n"
	$(UVICORN) server.app:app --host 127.0.0.1 --port 8000 --reload

demo: seed-demo serve-dev ## Full demo setup (seed + serve-dev)

# ═════════════════════════════════════════════════════════════════════════
#  Build & Release
# ═════════════════════════════════════════════════════════════════════════

.PHONY: build publish docker-build release-check

build: ## Build distribution packages (wheel + sdist)
	@printf "$(BLUE)▶ Building distribution packages...$(RESET)\n"
	$(POETRY) build
	@printf "$(GREEN)✓ Build complete. Artifacts in dist/$(RESET)\n"

publish: build ## Publish to PyPI (requires credentials)
	@printf "$(BLUE)▶ Publishing to PyPI...$(RESET)\n"
	$(POETRY) publish
	@printf "$(GREEN)✓ Published$(RESET)\n"

release-check: check security-check ## Run all checks before release
	@printf "$(BOLD)$(GREEN)✓ Release checks passed$(RESET)\n"

docker-build: ## Build Docker image (placeholder)
	@printf "$(YELLOW)▶ Docker build not yet implemented$(RESET)\n"
	@printf "$(DIM)  (placeholder — Dockerfile to be added)$(RESET)\n"
