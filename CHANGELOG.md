# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Advanced data pipeline** at `i3/data/` — cleaning, quality
  filtering, deduplication, deterministic splitting, and provenance
  tracking for real-world dialogue corpora.
  - `i3/data/cleaning.py`: NFKC normalisation, zero-width + bidi
    override stripping, HTML entity decoding, newline
    canonicalisation, whitespace collapse, control-character
    stripping.
  - `i3/data/quality.py`: eight built-in quality rules
    (`min_length`, `max_length`, `latin_ratio`,
    `unique_token_ratio`, `no_url_dump`, `no_email_dump`,
    `no_control_density`, `profanity_budget`) plus a
    `QualityReport` with per-rule rejection counts and length
    histogram.
  - `i3/data/dedup.py`: exact content-hash deduplication plus
    min-hash + LSH near-duplicate detection (pure Python, 128
    permutations, 16 bands by default; no external dependency).
  - `i3/data/sources.py`: source adapters for JSONL, CSV (column-
    mapped), plain text, DailyDialog (Li et al. 2017), and
    EmpatheticDialogues (FAIR).
  - `i3/data/lineage.py`: `Lineage` provenance metadata that
    travels with every record; `RecordSchema` Pydantic v2 contract
    with `extra="forbid"` and frozen invariants.
  - `i3/data/pipeline.py`: the `DataPipeline` orchestrator that
    composes every stage, writes split-aware JSONL
    (`train.jsonl` / `val.jsonl` / `test.jsonl`), and emits a
    structured `report.json` with schema version, duration,
    per-source counts, per-label counts, per-language signal,
    and the full quality-rule breakdown.
  - `i3/data/stats.py`: post-hoc dataset diagnostics — vocabulary
    size, type-token ratio, Zipf slope on the top-N tokens, OOV
    rate of every non-train split against the train vocabulary,
    vocab overlap, label entropy + Gini, length histograms in
    tokens and characters, residual-duplicate fingerprint.
- **Bundled sample corpus** at `data/corpora/sample_dialogues.jsonl` —
  35 curated dialogue turns across 12 conversations for end-to-end
  smoke-testing the pipeline without external downloads.
- **`training/prepare_dialogue_v2.py`** — CLI driver for the new
  pipeline. Repeatable `--jsonl` / `--txt` / `--csv` /
  `--dailydialog` / `--empathetic` flags let one invocation consume
  multiple sources together. The original `prepare_dialogue.py` is
  preserved for backwards compatibility.
- **Makefile targets** — `prepare-dialogue` / `prepare-data` run the
  pipeline on the bundled corpus end-to-end.
- **Sentiment lexicon expanded** (`i3/interaction/data/sentiment_lexicon.json`):
  123 → 690 curated entries (347 positive + 343 negative) with richer
  HCI / developer-experience vocabulary and informal interjections.
- **TF-IDF corpus expanded** (`i3/diary/logger.py`): 60 → 342 terms
  across communication, time, software engineering, data + ML,
  productivity, thinking, daily life, health, affective, and
  conversational-pattern categories.
- **`.env.example`** now documents 20+ previously-undocumented I3_*
  and observability environment variables.

### Tests

- **`tests/test_data_pipeline.py`** — 58 tests covering cleaning,
  quality rules, dedup, every source adapter, end-to-end pipeline
  runs, deterministic splitting, conv_id split-leakage guard,
  lineage roundtrip, report-JSON schema, and custom-rule
  extensibility.
- **`tests/test_data_properties.py`** — 19 Hypothesis property-based
  tests over all inputs: cleaner idempotence, unicode normalisation
  idempotence, control-char absence in output, length bounds,
  content-hash determinism, Jaccard reflexivity / symmetry /
  unit-interval range, deduplicator stats conservation, schema
  frozen-ness.
- **`tests/test_middleware_integration.py`** — 9 tests of the full
  middleware stack (security headers, body-size 413, rate-limit 429,
  exempt paths, `/whatif/*` inclusion in throttling).
- **`tests/test_auth.py`** — 13 tests for `server/auth.py` covering
  both activation modes, malformed JSON, cross-user rejection,
  `secrets.compare_digest` usage, POST-variant.
- **`tests/test_sentiment_lexicon.py`** — 16 tests (shape invariants
  + calibration golden set).
- **`tests/test_verify_harness.py`** — 20 tests for the 46-check
  harness framework itself.
- **`tests/test_privacy_sanitizer_hardening.py`** — 7 tests for the
  2026-04-23 audit sanitiser hardening.
- **`tests/test_bandit_concurrency.py`** — 7 concurrency tests for
  `ContextualThompsonBandit`.
- **`tests/test_config_schema.py`** — 8 tests that pin the canonical
  `configs/default.yaml` to the strict schema.
- **`tests/test_data_stats.py`** — 14 tests for the stats module
  (type-token ratio, Zipf slope range, label entropy / Gini, OOV
  rate against train, malformed-input resilience).

### Changed

- **`tests/conftest.py`** — torch-stub fallback so every test module
  collects cleanly on environments where the binary torch install is
  broken (Windows without VC++ runtime).
- **`i3/data/cleaning.py::_collapse_whitespace`** — trims both
  leading and trailing whitespace on every line (previously
  trailing-only).

### Fixed

- `MinHashLSH` slots-dataclass now declares `_rows_per_band` as a
  field so `__post_init__` can assign it.
- `DailyDialogSource.iter_records` closes its emotion-label file
  handle (previously leaked on some platforms).



## [1.1.0] — 2026-04-23

This release shifts the repository from "research prototype" to
"production-shaped application" — it adds the containerisation,
observability, supply-chain, policy, and universal-provider layers
around the v1.0.0 core, plus research extensions for interpretability,
uncertainty, red-teaming, and multimodal adaptation. Two deep audits
(security + robustness) were carried out and every blocker / high /
medium finding fixed; the verification harness runs green under
`--strict`.

### Added

- **Deployment.** Production multi-stage `Dockerfile`, a hardened
  `docker-compose.prod.yml` with TLS sidecar, full Kubernetes manifests
  (Deployment, HPA, PDB, NetworkPolicy, ServiceMonitor) with
  dev/staging/prod Kustomize overlays, a Helm chart, a Terraform
  reference module for AWS EKS, Skaffold + ArgoCD wiring.
  Alternate Dockerfiles (`dev`, `wolfi`, `mcp`) live under `docker/`.
- **Observability.** OpenTelemetry (traces, batched OTLP gRPC), Prometheus
  metrics (HTTP, pipeline stage P95, router arm distribution, SLM
  prefill/decode, PII sanitiser hits), structlog JSON logging with a
  sensitive-key redaction processor, Sentry with PII-scrubbing
  `before_send`, Grafana/Tempo/Prometheus docker-compose stack and a
  ten-panel overview dashboard, Langfuse LLM tracer with token- and
  cost-attribution for Anthropic Sonnet 4.5, and a request-correlation
  middleware (`X-Request-ID` + contextvars). Health probes at
  `/api/health`, `/api/live`, `/api/ready`, and `/api/metrics`.
- **Supply chain.** GitHub Actions workflows for SBOM (CycloneDX + Syft),
  OSSF Scorecard (weekly), Semgrep, Trivy, release-please + SLSA L3,
  cosign-signed multi-arch images, MkDocs build + gh-pages deploy,
  pytest-benchmark with regression alerting, lockfile audit,
  conventional-commit PR titles, and markdown link checking.
  `docs/security/slsa.md` maps Build Level 3, `docs/security/supply-chain.md`
  covers SBOM/scanner matrix and vulnerability SLA.
- **Policy.** Kyverno ClusterPolicies (signed images, non-root, default-deny
  NetworkPolicy, no `:latest`), OPA Rego admission, Cedar 4.x
  application-level authorisation, Falco + Tracee runtime rules,
  Sigstore policy-controller configuration, OpenSSF Allstar.
  `docs/security/policy_as_code.md` maps findings to NIST 800-53 and
  CIS Kubernetes Benchmark.
- **ML components.** `i3/encoder/loss.py` (SimCLR NT-Xent,
  fp16-compatible), `i3/interaction/sentiment.py` with a JSON-backed
  valence lexicon, INT8/INT4 quantisation via both
  `torch.quantization` and `torchao`, ONNX export with parity
  verification, ExecuTorch hooks, 11 alternative edge-runtime exporters
  (MLX, llama.cpp, TVM, IREE, Core ML, TensorRT-LLM, OpenVINO,
  MediaPipe, …).
- **MLOps.** MLflow-backed experiment tracker, DVC pipeline,
  SHA-256 checkpoint sidecars with JSON metadata, OpenSSF Model Signing
  v1.0 (sigstore / PKI / bare-key backends), model registry with
  optional MLflow/W&B mirroring.
- **Universal LLM provider layer.** 11 first-class adapters
  (Anthropic, OpenAI, Google, Azure, Bedrock, Mistral, Cohere, Ollama,
  OpenRouter, LiteLLM, Huawei PanGu) behind a single
  `MultiProviderClient` with sequential / parallel / best-of-N
  strategies and a circuit breaker, plus a prompt translator and a
  cost tracker with 2026 pricing.
- **Cloud ecosystem integrations.** DSPy compile-time prompt
  optimisation, NeMo Guardrails with a `.co` rulebook, Pydantic AI and
  Instructor adapters, Outlines constrained generation, Logfire and
  OpenLLMetry.
- **Research and interpretability.** Preregistered ablation study,
  mechanistic-interpretability study (activation patching, probing
  classifiers, attention circuits), ImplicitAdaptBench benchmark with
  three baselines, closed-loop persona-simulation evaluation with eight
  personas, MC-Dropout uncertainty quantification + counterfactual
  explanations exposed at `/api/explain/adaptation`, sparse
  autoencoders for cross-attention interpretability, a
  provider-agnostic LLM-as-judge harness, and a 55-attack
  adversarial red-team corpus with four target surfaces and four
  runtime invariants.
- **Huawei alignment.** HMAF runtime adapter (`i3/huawei/`), Kirin
  device profiles (9000 / 9010 / A2 / Smart Hanhan) with Da Vinci op
  coverage, Huawei Watch integration via PPG/HRV, translation endpoint
  targeting the AI Glasses use case, PDDL-grounded privacy-safety
  planner, speculative decoding, and an adaptive fast/slow compute
  router. Ecosystem alignment notes live under `docs/huawei/`.
- **Multimodal, continual, and meta-learning.** Voice prosody via
  librosa, facial affect via MediaPipe Face Mesh, three fusion
  strategies (`i3/multimodal/`); Elastic Weight Consolidation + ADWIN
  drift detection (`i3/continual/`); MAML, Reptile, and a task
  generator for few-shot user adaptation (`i3/meta_learning/`).
- **Advanced surfaces.** Preference-learning endpoint
  (`i3/router/preference_learning.py`, Bradley-Terry + Mehta 2025
  active selection), adaptation-conditioned TTS
  (`i3/tts/` + `server/routes_tts.py`), counterfactual / what-if
  endpoint (`server/routes_whatif.py`), in-browser ONNX-Runtime-Web
  inference with a COOP/COEP path-traversal-safe server
  (`server/routes_inference.py`), and a seven-panel cinematic demo UI at
  `/advanced`.
- **Federated / privacy / fairness / cross-device future work.**
  Flower client + FedAvg server, Opacus DP-SGD wrapper for the router
  posterior, HarmonyOS Distributed Data Management sync, per-archetype
  fairness metrics with bootstrap CI, and a keystroke-biometric ID
  module.
- **Documentation.** MkDocs Material site with ten ADRs, a 7 126-word
  research-paper draft (`docs/paper/`), an attorney-ready patent
  disclosure (`docs/patent/`), a conference poster, model + data cards
  and an accessibility statement under `docs/responsible_ai/`, the
  architecture full-reference, an edge-profiling report, demo script,
  and 15 slides with speaker notes.
- **Testing.** 80+ test modules covering unit, property (Hypothesis),
  contract (schemathesis), snapshot (syrupy), fuzz, load (locust,
  30-minute soak), mutation (mutmut), chaos, and benchmark scenarios.
- **Scripts.** Reorganised into topical subdirectories
  (`benchmarks/`, `demos/`, `experiments/`, `export/`, `security/`,
  `training/`, `verification/`) with a top-level `scripts/README.md`.
  Notable entry points: `verify_all.py` (46-check harness),
  `security/run_redteam.py` (55-attack adversarial harness),
  `security/run_redteam_notorch.py` (Windows/torch-DLL workaround).
- **Security infrastructure.** `server/auth.py` — opt-in caller-
  identity dependencies with two activation modes (bearer-token map
  or `X-I3-User-Id` header), `secrets.compare_digest` throughout,
  off by default (`I3_REQUIRE_USER_AUTH=1` to activate).

### Changed

- **Repository layout.** Scripts split into topical subdirectories;
  alternate Dockerfiles moved into `docker/`; `SLSA.md` and
  `SUPPLY_CHAIN.md` moved into `docs/security/`; top-level audit
  reports consolidated under `reports/audits/` with date-prefixed
  names; verification artefacts split into `reports/verification/` +
  `reports/redteam/`; research quickstart duplicates renamed from
  `*_README.md` to `*_quickstart.md`; `docs/ARCHITECTURE.md` moved
  to `docs/architecture/full-reference.md` and `docs/DEMO_SCRIPT.md`
  to `docs/slides/demo-script.md`.
- **Configuration.** `Config` gains `extra="forbid"` so typoed YAML
  sections fail at load time; `CloudConfig.model` default aligned with
  `configs/default.yaml` (`claude-sonnet-4-5`); `RouterConfig` carries
  `prior_alpha` (Beta prior) and `prior_precision` (Gaussian weight
  precision) as distinct fields; every environment variable is
  documented in `.env.example`; `app.state.config` reuses a single
  `load_config` call across the lifespan and factory.
- **Verification harness.** `_env_missing_result` / `_is_os_env_issue`
  recognise torch DLL-load failures on Windows (`WinError 1114`,
  `c10.dll`, `cudart`, `DLL load failed`, `KeyError` from partial
  binary imports, `AttributeError` on a `torch` stub) and return
  SKIP rather than a false FAIL.
- **Build + dependency management.** `pyproject.toml` adds
  `observability`, `mlops`, `ml-advanced`, `analytics`, `distributed`,
  `llm-ecosystem`, `providers`, `edge-runtimes`, `multimodal`,
  `future-work`, `policy`, `mcp`, `tts` Poetry groups; `dev` expanded
  with Hypothesis, schemathesis, syrupy, mutmut, pytest-benchmark,
  jsonschema; `docs` expanded with the MkDocs Material ecosystem
  plugins; `detect-secrets` added to `security`.

### Fixed

- **Keystroke events were never reaching the TCN.**
  `server/websocket.py` called the `async def process_keystroke`
  coroutine without `await`, so every keystroke was dropped and every
  feature window was fed the zero-metrics fallback. One-line fix,
  biggest behavioural-correctness regression in the project.
- **Rate limiter bypassed for `/whatif/*`.** The middleware used an
  include-list (`/api/*` only); it now uses an exclude-list so every
  new route inherits throttling by default.
- **Cross-user PII harvesting via preference routes.** Free-text
  prompts and A/B responses now pass through `PrivacySanitizer`
  before persistence, and all per-user GETs are gated by
  `require_user_identity`.
- **Missing authentication gates.** Six POST routes that accept
  `user_id` in the body (`/whatif/respond`, `/whatif/compare`,
  `/api/tts`, `/api/translate`, `/api/preference/record`,
  `/api/explain/adaptation`) now depend on
  `require_user_identity_from_body`; the three user-scoped GETs
  in `server/routes.py` depend on `require_user_identity`.
- **`DiaryStore` defeated its own FK enforcement.** The previous
  per-operation `aiosqlite.connect` reopened the connection on every
  call, and `PRAGMA foreign_keys = ON` is per-connection — so FK
  enforcement was effectively off. Now holds one connection for the
  store's lifetime with WAL journal + FK pragma set once; 10 call
  sites migrated via a drop-in async context manager. Also adds
  idempotent `close()`.
- **SLM generation blocked the event loop.** `_generate_response` now
  offloads synchronous PyTorch generation to
  `loop.run_in_executor(...)`, mirroring the encoder pattern.
  `generate_session_summary` is wrapped in `asyncio.wait_for` with
  `timeout * 1.2` to bound session-end latency.
- **Unbounded per-user memory growth.** `Pipeline.user_models` is now
  an `OrderedDict` capped at `I3_MAX_TRACKED_USERS` (default 10 000)
  with O(1) LRU eviction and full per-user footprint cleanup
  (response-time, length, engagement, previous-route dicts all
  cleared).
- **Bandit concurrency races.** `ContextualThompsonBandit.select_arm` /
  `update` / `_refit_posterior` now serialise under a reentrant lock;
  history uses `deque(maxlen=N)` for O(1) overflow (previously O(n)
  slice churn). Stress-tested under 8-thread / 800-op concurrency.
- **`httpx.AsyncClient` lazy-init races.** The Anthropic, OpenRouter,
  Ollama, and Huawei PanGu clients now guard lazy construction with
  a lock (`asyncio.Lock` or a double-checked `threading.Lock`) so a
  concurrent first hit cannot orphan one of the clients.
- **Global RNG mutation per explain request.**
  `_surrogate_mapping_fn` previously called `torch.manual_seed` on
  every request, silently breaking Thompson-sampling exploration in
  every other in-flight coroutine. Now uses a scoped
  `torch.Generator` with a module-level cached layer.
- **Exception class names leaked to the wire.** The pipeline error
  path no longer sets `adaptation["error"] = type(exc).__name__`;
  it uses the constant `"pipeline_error"` instead.
- **`prior_alpha` passed as `prior_precision`.** `IntelligentRouter`
  now passes the right `RouterConfig` field to the bandit
  constructor.
- **Pydantic exception chaining.** `routes_translate.py` preserves
  the cause via `raise HTTPException(...) from exc`.
- **Cloud provider body echo.** OpenRouter, Huawei PanGu, and Ollama
  no longer include `response.text` in exception messages; the body
  moves to `logger.debug`. All four clients now pin `verify=True`,
  `follow_redirects=False`, and `httpx.Limits(...)` explicitly.
- **Sanitiser false positives.** The IP-address regex now requires
  each octet ≤ 255, so Windows build numbers (`10.0.22621`) and
  SemVer strings no longer trip the PII detector. The auditor's
  recursion is depth-capped at 32 with O(n) path joining; its
  findings buffer is a `deque(maxlen=1_000)`.
- **Admin export enumeration oracle.** `admin_export` now returns
  404 when profile + diary + bandit stats are all empty.
- **Limiter eviction cost.** `_SlidingWindowLimiter` now uses
  `OrderedDict` + `popitem(last=False)` for amortised O(1)
  eviction (was O(n) via `min(..., key=...)`).
- **`torch.load` pickle-RCE sinks.**
  `i3/interpretability/activation_cache.py` now uses
  `weights_only=True` on both single-file and sharded paths, with a
  1 MiB cap on the manifest, structural validation
  (`dict[str, list[str]]`), and per-shard `resolve()` +
  `relative_to()` checks that block `../` traversal.
  `i3/slm/train.py::load_checkpoint` verifies an optional
  `<path>.sha256` sidecar before loading with constant-time compare.
- **Cloud RNG and config drift.** `load_config` is now called once
  in `create_app`; `server/app.py` refuses to start with
  `I3_WORKERS > 1` unless `I3_ALLOW_LOCAL_LIMITER=1` acknowledges
  the per-process limiter semantics.
- **Minor.** ONNX export CLI `print()` → `sys.stderr.write()`;
  `configs/default.yaml` cloud model pinned; honesty-slide title
  fixed; Trivy and Semgrep GitHub Action tags pinned to versions.

### Security

- `scripts/verify_all.py --strict` — **28 pass / 0 fail / 16 skip**
  (skips all environment-gated: torch DLL, missing binaries like
  `ruff`, `mypy`, `helm`, `cedarpy`, `mkdocs`).
- Red-team harness invariants — **3 / 4 pass**
  (`privacy_invariant`, `sensitive_topic_invariant`,
  `pddl_soundness`). The fourth (`rate_limit_invariant`) fails
  only because the FastAPI surface is not exercised on a host
  with a broken torch install.
- Two deep audits recorded under
  [`reports/audits/`](reports/audits/): security review,
  robustness/performance/code-quality audit, and a per-finding
  fix log with file:line citations.

## [1.0.0] — 2026-04-12

### Added

#### Core ML Components
- **TCN Encoder** — Temporal Convolutional Network built from scratch in PyTorch.
  Four dilated causal convolution blocks (dilations `[1, 2, 4, 8]`) with residual
  connections and LayerNorm. Trained with NT-Xent contrastive loss on synthetic
  interaction data.
- **Custom SLM** — ~6.3M parameter transformer built entirely from first principles
  (no HuggingFace). Includes:
  - Word-level tokenizer with special tokens and vocabulary building
  - Token embeddings with sinusoidal positional encoding
  - Multi-head self-attention with KV caching for inference
  - Novel cross-attention conditioning to AdaptationVector + UserStateEmbedding
  - Pre-LN transformer blocks (4 layers, 256 d_model, 4 heads)
  - Weight-tied output projection
  - Top-k / top-p / repetition-penalty sampling
  - INT8 dynamic quantization for edge deployment
- **Contextual Thompson Sampling Bandit** — Two-arm contextual bandit with
  Bayesian logistic regression posteriors, Laplace approximation refitted via
  Newton-Raphson MAP estimation, and Beta-Bernoulli cold-start fallback.
- **Three-Timescale User Model** — Instant state, session profile (EMA α=0.3),
  and long-term profile (EMA α=0.1) with Welford's online algorithm for
  running feature statistics.

#### Behavioural Perception
- **Interaction Monitor** — Real-time keystroke event processing with per-user
  buffers, typing burst detection (500ms pause threshold), and composition metrics.
- **32-Dimensional Feature Vector** — Four groups of 8 features covering keystroke
  dynamics, message content, session dynamics, and deviation from baseline.
- **Linguistic Analyzer** — Flesch-Kincaid grade, type-token ratio, formality
  scoring (52 contractions + 54 slang markers), syllable counting, and
  ~365-word sentiment lexicon with negation handling — all implemented from
  scratch with no external NLP libraries.

#### Adaptation Layer
- **Four Adaptation Dimensions** — CognitiveLoad, StyleMirror (4-dim formality/
  verbosity/emotionality/directness), EmotionalTone, and Accessibility adapters.
- **AdaptationVector** — 8-dimensional vector serializable to/from tensors for
  model conditioning.

#### Cloud Integration
- **Async Anthropic Client** — Built with httpx, supports retry with exponential
  backoff, token usage tracking, and graceful fallback.
- **Dynamic Prompt Builder** — Translates AdaptationVector to natural-language
  system prompt instructions.
- **Response Post-Processor** — Enforces length limits and vocabulary
  simplification for accessibility.

#### Persistence
- **Async SQLite Stores** — User models and interaction diary use `aiosqlite`
  for non-blocking I/O.
- **Privacy-Safe Diary** — Logs only embeddings, scalar metrics, TF-IDF topic
  keywords, and adaptation parameters — never raw user text.
- **TF-IDF Topic Extraction** — 175 stopwords, 60 pre-computed IDF scores with
  rare-term fallback.

#### Web Application
- **FastAPI Backend** — Async application factory with lifespan management,
  WebSocket handler for real-time interaction, and REST API.
- **Dark-Theme Frontend** — Vanilla HTML/CSS/JS (no build step) with:
  - KeystrokeMonitor capturing inter-key intervals, bursts, and composition time
  - Canvas-based 2D embedding visualization with fading trail
  - Animated gauge bars for all adaptation dimensions
  - Collapsible diary panel
  - WebSocket client with exponential backoff reconnection

#### Privacy & Security
- **10 PII Regex Patterns** — Email, phone (US/UK/intl), SSN, credit card, IP
  address, physical address, DOB, URL.
- **Fernet Encryption** — Symmetric encryption for user model embeddings at rest
  with environment-based key management.
- **Privacy Auditor** — Async database scanner that detects raw-text leaks in
  SQLite tables.
- **Topic Sensitivity Detector** — 12 regex patterns across 8 categories
  (mental health, credentials, abuse, financial, medical, relationship, legal,
  employment) with severity scoring for privacy-override routing.

#### Edge Feasibility
- **Memory Profiler** — `tracemalloc`-based peak memory measurement, FP32 vs
  INT8 size comparison, parameter counting.
- **Latency Benchmark** — P50/P95/P99 percentiles with warmup iterations and
  FP32 vs INT8 speedup comparison.
- **Device Feasibility Matrix** — Assessments against Kirin 9000, Kirin A2, and
  Smart Hanhan with configurable memory budgets.
- **Markdown Report Generation** — For use in presentation materials.

#### Testing
- **80+ Unit Tests** — Across TCN, SLM, bandit, user model, and pipeline components.
- **Security Test Suite** — Dedicated tests for PII sanitization, encryption
  round-trips, topic sensitivity, input validation, and DoS resistance.
- **Property-Based Tests** — Shape invariants, bandit convergence, adaptation
  vector bounds.
- **Integration Tests** — End-to-end pipeline flow with privacy guarantee checks.
- **Shared Fixtures** — `conftest.py` with 7 reusable fixtures including async
  temporary diary store.

#### Documentation
- **README.md** — Portfolio-grade project overview with box-drawn architecture
  diagrams, layer-by-layer descriptions, and edge feasibility tables.
- **ARCHITECTURE.md** — ~750-line research-paper-style design document covering
  system overview, data flow, the 32-dim feature vector, TCN architecture math,
  three-timescale user model, adaptation dimensions, Thompson sampling
  Bayesian formulation, cross-attention conditioning novelty, privacy
  architecture, and design trade-offs.
- **DEMO_SCRIPT.md** — Operational 4-phase demo playbook with pre-flight
  checklist, exact dialogue, recovery procedures, and timing budget.
- **CONTRIBUTING.md** — Development workflow, coding standards, and
  contribution process.
- **SECURITY.md** — Security policy, threat model, audit report, and mitigations.

#### Tooling & Infrastructure
- **Poetry** dependency management with 4 dependency groups (main, dev, security, docs).
- **Ruff** linting and formatting with security lints (`S`, `B`, `PTH`).
- **Mypy** type checking with per-module overrides.
- **Pytest** with asyncio support, coverage, parallel execution (xdist), and markers.
- **Pre-commit hooks** for automated quality checks.
- **GitHub Actions CI** with matrix testing across Python 3.10/3.11/3.12 on
  Ubuntu and macOS.
- **Security workflows** running Bandit, pip-audit, Safety, and CodeQL.
- **Dependabot** for weekly dependency updates grouped by ecosystem.
- **Issue and PR templates** for GitHub.
- **Makefile** with 25+ self-documenting targets and colored output.
- **Setup scripts** (`scripts/setup.sh`, `scripts/run_demo.sh`,
  `scripts/security/generate_encryption_key.py`).

### Security
- Formal security audit conducted — see [SECURITY.md](SECURITY.md).
- **CORS** restricted to configurable origins; wildcard only permitted behind
  an explicit opt-in environment variable.
- **Rate limiting** on API (60 req/min per IP) and WebSocket (600 msg/min per
  user) endpoints.
- **Security headers middleware** — X-Frame-Options, X-Content-Type-Options,
  Content-Security-Policy, Referrer-Policy, Permissions-Policy.
- **Request size limiting** — 1 MB maximum on REST requests.
- **WebSocket limits** — 64 KB max message size, 1000 messages per session,
  1 hour maximum session duration.
- **Input validation** — User IDs restricted to `^[a-zA-Z0-9_-]{1,64}$`;
  pagination params bounded.
- **torch.load** with `weights_only=True` for safer deserialization.
- **yaml.safe_load** for all configuration loading.
- **Exception handlers** sanitise error responses — no stack traces or
  internal paths exposed.
- **API key redaction** in logs (never logs full key, only prefix/suffix).
- **Default bind** to loopback only (`127.0.0.1`) — public bind requires
  explicit `I3_HOST` override.

### Infrastructure Choices
- Python 3.10+ required (uses modern typing syntax).
- PyTorch 2.0+ for eager mode and native quantization.
- FastAPI 0.110+ with Starlette middleware.
- Pydantic 2.6+ for configuration validation.
- aiosqlite 0.20+ for async database access.
- cryptography 42+ for Fernet symmetric encryption.

[Unreleased]: https://github.com/abailey81/implicit-interaction-intelligence/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/abailey81/implicit-interaction-intelligence/releases/tag/v1.0.0
