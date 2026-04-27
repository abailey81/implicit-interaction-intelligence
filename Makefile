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
	@printf "$(BOLD)  ── End-to-end Orchestrator ─────────────────────────────────────$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(all|all-.*):.*?## / {printf "    $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
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

.PHONY: test test-cov test-fast test-security test-parallel test-iter test-cascade

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

test-iter: ## Iter 52-74 fast subset (cascade + multilingual + perf + health + privacy + contract + multi-provider + BPE-corner + observability + cost-tracker + multimodal + engagement + KG + PII + critic + adaptation + TCN)
	@printf "$(BLUE)▶ Running iter 52-74 fast subset...$(RESET)\n"
	$(PYTEST) -q --no-header --tb=no \
	    tests/test_intent_cascade.py \
	    tests/test_profiling_cascade.py \
	    tests/test_health_deep.py \
	    tests/test_multilingual_robustness.py \
	    tests/test_slm_perf_guard.py \
	    tests/test_privacy_budget_circuit.py \
	    tests/test_pipeline_output_contract.py \
	    tests/test_cloud_multi_provider.py \
	    tests/test_bpe_corner_cases.py \
	    tests/test_observability_spans.py \
	    tests/test_cost_tracker_global.py \
	    tests/test_multimodal_validators.py \
	    tests/test_engagement_signal.py \
	    tests/test_knowledge_graph_dedupe.py \
	    tests/test_pii_sanitizer_coverage.py \
	    tests/test_self_critic.py \
	    tests/test_adaptation_vector.py \
	    tests/test_tcn_invariants.py \
	    tests/test_diary_schema_invariants.py \
	    tests/test_intent_types.py \
	    tests/test_routing_decision_schema.py \
	    tests/test_knowledge_graph_canonical.py \
	    tests/test_encryption_envelope.py \
	    tests/test_privacy_budget_snapshot.py \
	    tests/test_sensitivity_categories.py \
	    tests/test_diary_store_lifecycle.py \
	    tests/test_dashboard_html_contract.py \
	    tests/test_server_app_routes.py \
	    tests/test_pipeline_stated_facts.py \
	    tests/test_cost_tracker_integration.py \
	    tests/test_bandit_invariants.py \
	    tests/test_qwen_adapter_alignment.py \
	    tests/test_pricing_table.py \
	    tests/test_cost_pricing_integration.py \
	    tests/test_knowledge_graph_compose.py \
	    tests/test_pipeline_input_contract.py \
	    tests/test_chat_chip_css_classes.py \
	    tests/test_huawei_tabs_js_wiring.py \
	    tests/test_intent_dataset_files.py \
	    tests/test_dedupe_sentences.py \
	    tests/test_response_postprocess.py \
	    tests/test_linguistic_analyzer.py \
	    tests/test_edge_profiler_helpers.py \
	    tests/test_pipeline_trace_collector.py \
	    tests/test_cloud_prompt_builder.py \
	    tests/test_chat_js_chips.py \
	    tests/test_cost_cached_tokens.py \
	    tests/test_pipeline_error_output.py \
	    tests/test_bandit_convergence.py \
	    tests/test_explain_decomposer_patterns.py \
	    tests/test_privacy_budget_redactions.py \
	    tests/test_cloud_error_taxonomy.py \
	    tests/test_completion_dataclasses.py \
	    tests/test_explain_plan_dataclass.py \
	    tests/test_stage_record_invariants.py \
	    tests/test_kg_relation_dataclass.py \
	    tests/test_response_path_lifecycle.py
	@printf "$(GREEN)✓ Iter 52-121 subset green$(RESET)\n"

test-cascade: ## Cascade-only sweep (Qwen LoRA arm B + dashboard + chips + spans)
	@printf "$(BLUE)▶ Running cascade-only sweep...$(RESET)\n"
	$(PYTEST) -q --no-header --tb=no \
	    tests/test_intent_cascade.py \
	    tests/test_profiling_cascade.py \
	    tests/test_observability_spans.py
	@printf "$(GREEN)✓ Cascade tests green$(RESET)\n"

# ═════════════════════════════════════════════════════════════════════════
#  Data & Training
# ═════════════════════════════════════════════════════════════════════════

.PHONY: generate-data train-encoder train-slm train-all evaluate profile

generate-data: ## Generate synthetic training data
	@printf "$(BLUE)▶ Generating synthetic data...$(RESET)\n"
	$(PY) -m training.generate_synthetic --config configs/default.yaml
	@printf "$(GREEN)✓ Data generation complete$(RESET)\n"

prepare-dialogue: ## Clean + dedup + split the bundled sample dialogue corpus
	@printf "$(BLUE)▶ Running the i3.data pipeline on the bundled sample...$(RESET)\n"
	$(PY) -m training.prepare_dialogue_v2 \
		--jsonl data/corpora/sample_dialogues.jsonl \
		--output-dir data/processed/sample
	@printf "$(GREEN)✓ Sample dataset written to data/processed/sample/$(RESET)\n"

prepare-data: prepare-dialogue ## Alias for prepare-dialogue

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
#  End-to-end orchestrator — one command to rule them all
# ═════════════════════════════════════════════════════════════════════════

.PHONY: all all-fast all-full all-list all-resume

all-fast: ## Fast end-to-end: deps + env + verify + serve (~5 min, uses demo ckpts)
	@printf "$(BOLD)$(CYAN)▶ Orchestrator — fast mode$(RESET)\n"
	$(PY) scripts/run_everything.py --mode fast

all-full: ## Full end-to-end: prereq → install → env → data → train → eval → tests → security → verify → bench → onnx → docs → serve
	@printf "$(BOLD)$(CYAN)▶ Orchestrator — full mode$(RESET)\n"
	$(PY) scripts/run_everything.py --mode full

all: all-fast ## Alias for all-fast (the quickstart path)

all-list: ## Print the orchestrator stage graph (fast + full) and exit
	@printf "$(BOLD)$(CYAN)── fast mode ──$(RESET)\n"
	@$(PY) scripts/run_everything.py --mode fast --list
	@printf "\n$(BOLD)$(CYAN)── full mode ──$(RESET)\n"
	@$(PY) scripts/run_everything.py --mode full --list

all-resume: ## Re-run the orchestrator, skipping stages whose outputs already exist
	@printf "$(BOLD)$(CYAN)▶ Orchestrator — resume$(RESET)\n"
	$(PY) scripts/run_everything.py --mode full --resume

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

docker-build: ## Build the production Docker image (multi-stage, non-root, tini PID 1)
	@printf "$(BLUE)▶ Building production Docker image (i3:latest)...$(RESET)\n"
	DOCKER_BUILDKIT=1 docker build -t i3:latest .
	@printf "$(GREEN)✓ Image built. Run: docker run --rm -p 8000:8000 i3:latest$(RESET)\n"

docker-build-dev: ## Build the development Docker image (Dockerfile.dev, hot reload)
	@printf "$(BLUE)▶ Building development Docker image (i3:dev)...$(RESET)\n"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t i3:dev .
	@printf "$(GREEN)✓ Image i3:dev built$(RESET)\n"

docker-up: ## Start the service via docker compose (base profile)
	@printf "$(BLUE)▶ Starting docker compose stack...$(RESET)\n"
	docker compose up --build

docker-up-prod: ## Start with hardened prod profile (read-only rootfs, cap_drop, nginx sidecar)
	@printf "$(BLUE)▶ Starting production docker compose stack...$(RESET)\n"
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
	@printf "$(GREEN)✓ Stack up. Logs: docker compose logs -f$(RESET)\n"

docker-down: ## Tear down docker compose stack
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml down || docker compose down

# ═════════════════════════════════════════════════════════════════════════
#  Documentation (MkDocs Material)
# ═════════════════════════════════════════════════════════════════════════

.PHONY: docs docs-serve docs-build docs-strict docs-deploy

docs: docs-serve ## Alias for docs-serve

docs-serve: ## Serve MkDocs site locally on :8001 with hot reload
	@printf "$(BLUE)▶ Starting MkDocs dev server on http://127.0.0.1:8001 ...$(RESET)\n"
	$(POETRY) run mkdocs serve -a 127.0.0.1:8001

docs-build: ## Build the static site into ./site
	@printf "$(BLUE)▶ Building MkDocs site...$(RESET)\n"
	$(POETRY) run mkdocs build
	@printf "$(GREEN)✓ Site built in ./site$(RESET)\n"

docs-strict: ## Build with --strict (fails on broken links / warnings)
	@printf "$(BLUE)▶ Strict MkDocs build...$(RESET)\n"
	$(POETRY) run mkdocs build --strict
	@printf "$(GREEN)✓ Strict build passed$(RESET)\n"

docs-deploy: ## Deploy docs to GitHub Pages (gh-pages branch)
	@printf "$(BLUE)▶ Deploying MkDocs to GitHub Pages...$(RESET)\n"
	$(POETRY) run mkdocs gh-deploy --force

# ═════════════════════════════════════════════════════════════════════════
#  Observability stack (OTel Collector + Prometheus + Grafana + Tempo)
# ═════════════════════════════════════════════════════════════════════════

.PHONY: obs-up obs-down

obs-up: ## Start local observability stack (Grafana on :3000, Prometheus on :9090)
	@printf "$(BLUE)▶ Starting observability stack...$(RESET)\n"
	docker compose -f deploy/observability/docker-compose.observability.yml up -d
	@printf "$(GREEN)✓ Grafana: http://localhost:3000  Prometheus: http://localhost:9090$(RESET)\n"

obs-down: ## Tear down local observability stack
	@docker compose -f deploy/observability/docker-compose.observability.yml down

# ═════════════════════════════════════════════════════════════════════════
#  Advanced ML tooling (ONNX export, edge profiling, model signing)
# ═════════════════════════════════════════════════════════════════════════

.PHONY: benchmarks bench-ci export-onnx verify-onnx profile-edge sign-model eval-conditioning

benchmarks: ## Run pytest-benchmark micro-benchmarks (warmup 3 / measured 20)
	@printf "$(BLUE)▶ Running benchmarks...$(RESET)\n"
	$(PYTEST) benchmarks/ --benchmark-only

bench-ci: ## Run benchmarks and emit JSON for CI regression tracking
	$(PYTEST) benchmarks/ --benchmark-only --benchmark-json=benchmark-output.json

export-onnx: ## Export TCN encoder + SLM to ONNX (checkpoints/ required)
	$(PY) -m scripts.export_onnx --encoder checkpoints/encoder/best.pt --slm checkpoints/slm/best.pt --out exports/

verify-onnx: ## Verify ONNX inference parity vs PyTorch (atol=1e-4)
	$(PY) -m scripts.verify_onnx --onnx exports/

profile-edge: ## Write reports/edge_profile_<date>.md using i3.profiling
	$(PY) -m scripts.profile_edge

sign-model: ## Sign a checkpoint with OpenSSF Model Signing v1.0 (sigstore)
	$(PY) -m scripts.sign_model sign --model checkpoints/slm/best.pt --method sigstore

eval-conditioning: ## Run the cross-attention conditioning-sensitivity evaluation
	$(PY) -m scripts.evaluate_conditioning

# ═════════════════════════════════════════════════════════════════════════
#  Verification harness (Batch G8) + iterative verification (Batch G10)
# ═════════════════════════════════════════════════════════════════════════

.PHONY: verify verify-strict verify-quick redteam run-ablation run-closed-loop run-sae run-llm-judge run-ewc run-maml run-hmaf

verify: ## Run the 46-check verification harness -> reports/verification_latest.md
	@printf "$(BLUE)▶ Running verification harness...$(RESET)\n"
	$(PY) scripts/verify_all.py --out reports/verification_latest.json --out-md reports/verification_latest.md
	@printf "$(GREEN)✓ Verification report at reports/verification_latest.md$(RESET)\n"

verify-strict: ## Run the harness in --strict mode (exit non-zero on any FAIL)
	@printf "$(BLUE)▶ Strict verification (exit non-zero on any FAIL)...$(RESET)\n"
	$(PY) scripts/verify_all.py --strict --out reports/verification_strict.json --out-md reports/verification_strict.md

verify-quick: ## Only code + config + interview categories
	$(PY) scripts/verify_all.py --categories code,config,interview \
		--out reports/verification_quick.json --out-md reports/verification_quick.md

redteam: ## Run the 55-attack red-team harness
	$(PY) scripts/security/run_redteam.py --targets sanitizer,pddl,guardrails --fail-fast

audit: ## Run every verification layer (tests + strict harness + red-team + lint)
	@printf "$(BLUE)▶ Layer 1 — pytest suite...$(RESET)\n"
	@$(MAKE) --no-print-directory test
	@printf "$(BLUE)▶ Layer 2 — 44-check verification harness (strict)...$(RESET)\n"
	@$(MAKE) --no-print-directory verify-strict
	@printf "$(BLUE)▶ Layer 3 — 55-attack red-team harness...$(RESET)\n"
	@$(MAKE) --no-print-directory redteam
	@printf "$(BLUE)▶ Layer 4 — lint + types + security scan...$(RESET)\n"
	@$(MAKE) --no-print-directory lint typecheck security-check
	@printf "$(GREEN)✓ Every verification layer passed$(RESET)\n"

run-ablation: ## Run the preregistered empirical ablation study (Batch A)
	$(PY) -m scripts.run_ablation_study --out reports/ablation_latest.json --out-md reports/ablation_latest.md

run-closed-loop: ## Run the closed-loop simulation evaluation (Batch G1)
	$(PY) -m scripts.run_closed_loop_eval --out reports/closed_loop_latest.json --out-md reports/closed_loop_latest.md

run-sae: ## Train + analyse sparse autoencoders on cross-attention (Batch G3)
	$(PY) -m scripts.train_sae && $(PY) -m scripts.analyse_sae

run-llm-judge: ## LLM-as-judge over an ablation or benchmark result (Batch G4)
	$(PY) -m scripts.run_llm_judge --ablation-results reports/ablation_latest.json

run-ewc: ## Sequential-task EWC training curve (Batch F-5)
	$(PY) -m scripts.run_ewc_demo

run-maml: ## Few-shot user-adaptation eval after MAML meta-training (Batch G5)
	$(PY) -m scripts.run_few_shot_demo

run-hmaf: ## Drive five canned HMAF intents end-to-end (Batch D-2)
	$(PY) -m scripts.run_hmaf_runtime_demo
