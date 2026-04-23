# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Containers and deployment

- Production `Dockerfile` — multi-stage (builder + runtime), Python 3.11
  slim-bookworm, non-root `i3:10001`, `tini` PID 1, OCI image labels,
  CPU-only torch from the PyTorch wheel index, HEALTHCHECK via stdlib
  `urllib`. The final image is stripped of `pip` / `poetry` /
  `build-essential` / `git` / `curl`.
- `docker/Dockerfile.dev` for hot-reload development.
- `docker/Dockerfile.wolfi` — Chainguard Wolfi distroless variant. Typical
  30–60 HIGH/CRITICAL CVEs on `python:3.11-slim` drop to 0 on Wolfi.
- `docker/Dockerfile.mcp` — minimal image for the Model Context Protocol
  server.
- `docker-compose.yml` + `docker-compose.prod.yml` with `read_only` root
  filesystem, `cap_drop: [ALL]`, `no-new-privileges`, an nginx sidecar
  for TLS, and tmpfs for mutable paths.
- `.devcontainer/` — VS Code dev container with Poetry 1.8.3 bootstrap,
  pre-commit hook install, and Python / Ruff / Pylance / mypy extensions.
- Kubernetes manifests under `deploy/k8s/` — Deployment with
  `RuntimeDefault` seccomp, read-only root filesystem, `capabilities:
  drop ALL`, `startupProbe` + `livenessProbe` (`/api/live`) +
  `readinessProbe` (`/api/ready`), HPA v2 (min 2 / max 10, CPU 70 % plus
  a custom `http_requests_per_second` metric), PodDisruptionBudget,
  default-deny NetworkPolicy with narrow allow rules, and a
  ServiceMonitor for the Prometheus Operator. Dev / staging / prod
  Kustomize overlays are provided.
- Helm chart under `deploy/helm/i3/` with `_helpers.tpl`, comprehensive
  `values.yaml` and `values-{dev,prod}.yaml`, `NOTES.txt`, and a
  `tests/test-connection` probe.
- Terraform reference module under `deploy/terraform/` targeting AWS EKS
  via the `kubernetes ~> 2.27` and `helm ~> 2.13` providers.
- Skaffold configuration for local Kubernetes iteration.
- ArgoCD GitOps wiring: `deploy/argocd/application.yaml` and
  `appproject.yaml`.
- `Tiltfile` for local hot-reload development against a local Kubernetes
  cluster.

#### Observability

- `i3/observability/` package with soft-imported OpenTelemetry, structlog,
  Prometheus, and Sentry clients. Every module is a no-op when its
  dependency is absent so the core pipeline boots unchanged.
- structlog JSON logging with a sensitive-key redaction processor
  (`api_key`, `authorization`, `cookie`, `token`, `password`, `secret`,
  `fernet`, `I3_ENCRYPTION_KEY`), a standard-library bridge, and
  contextvars for `request_id` / `user_id` / `trace_id` / `client_ip`.
- OpenTelemetry `TracerProvider` with `BatchSpanProcessor`, an OTLP gRPC
  exporter, `ParentBased(TraceIdRatioBased)` sampling, and
  auto-instrumentation for FastAPI, httpx, sqlite3, and logging.
- Prometheus metrics: HTTP request rate and latency histograms; pipeline
  stage P95 stacked; router arm distribution plus posterior-mean gauges;
  SLM prefill / decode latency; TCN encoder latency; WebSocket concurrent
  gauge; PII sanitiser hit counter by pattern type.
- Sentry integration with a PII-scrubbing `before_send` hook that lazily
  uses `PrivacySanitizer`.
- Request correlation middleware — `X-Request-ID`, contextvars binding,
  latency logs.
- Health endpoints in `server/routes_health.py`: `/api/health`,
  `/api/live`, `/api/ready` (pipeline + encryption-key + disk checks,
  returning 503 on failure), and `/api/metrics` (gated by
  `I3_METRICS_ENABLED`).
- `deploy/observability/` docker-compose stack: OpenTelemetry Collector,
  Prometheus, Tempo, and Grafana with a provisioned ten-panel I³
  overview dashboard.
- Langfuse LLM tracer in `i3/observability/langfuse_client.py` with a
  `@trace_generation` decorator and Anthropic Sonnet 4.5 pricing
  constants for cost attribution.
- Pyroscope integration (`i3/observability/pyroscope_integration.py`)
  and a `grafana-alloy.river` pipeline under `deploy/observability/`.

#### Supply-chain security and policy

- GitHub Actions workflows: `sbom` (CycloneDX + Syft), `scorecard`
  (OSSF weekly with SARIF and badge), `semgrep` (nine curated
  rulesets), `trivy` (filesystem + config + image scans, fail on
  CRITICAL), `release` (release-please → build → SLSA Level 3 generic
  generator → PyPI trusted publishing via OIDC), `docker` (multi-arch
  buildx with GHA cache, cosign keyless signing, SBOM attestation,
  SLSA L3 container generator), `docs` (MkDocs strict build and
  gh-pages deploy), `benchmark` (`pytest-benchmark` with regression
  alerting), `stale`, `lockfile-audit`, `pr-title` (semantic titles),
  and `markdown-link-check`.
- `.github/actions/setup-poetry` composite action for DRY Poetry setup.
- `renovate.json` with grouped rules (ml-core, web-stack, dev-tooling,
  actions, docker-base-images), vulnerability alerts, and lockfile
  maintenance.
- `commitlint.config.js` for conventional-commit scopes.
- `lefthook.yml` — pre-commit ruff / mypy / detect-secrets and
  pre-push pytest smoke test.
- `.sigstore.yaml`, `.trivyignore`, `.semgrepignore`.
- `docs/security/slsa.md` — Build Level 3 mapping and verification
  with `cosign` + `slsa-verifier`.
- `docs/security/supply-chain.md` — SBOM, vulnerability SLA, scanner
  matrix.
- `release-please` configuration seeded for automated semver releases.
- `deploy/policy/kyverno/` — five ClusterPolicies (signed images,
  deny `:latest`, non-root, network-policy-required, default-deny
  generator).
- `deploy/policy/opa/` — Rego v1 admission policies with tests.
- `deploy/policy/cedar/` — application-level authorization in Cedar 4.x
  with 21 scenarios and the `i3/authz/cedar_adapter.py` runtime glue.
- `deploy/policy/falco/` — custom runtime rules alerting on
  `diary.db` reads, unexpected egress, exec, and writes to `/app`.
- `deploy/policy/tracee/` — eBPF runtime rules.
- `deploy/policy/sigstore/policy-controller-config.yaml`.
- `.github/allstar/` — OpenSSF Allstar configuration.
- `docs/security/policy_as_code.md` — NIST 800-53 and CIS
  Kubernetes Benchmark matrix plus a T1–T13 threat model.

#### Advanced ML capabilities

- `i3/encoder/loss.py` — `NTXentLoss` `nn.Module` and a matching
  `nt_xent_loss` function alias extracted from `train.py` for reuse
  (SimCLR NT-Xent; Chen et al. 2020, arXiv:2002.05709, Eq. 1). Uses a
  `-1e9` diagonal mask for fp16 compatibility.
- `i3/interaction/sentiment.py` + `data/sentiment_lexicon.json` —
  `ValenceLexicon` with a JSON loader, a silent inline-constant
  fallback, negation-window flipping, and `score()` / `intensity()`
  methods.
- `i3/slm/quantize_torchao.py` — PyTorch-native `Int4WeightOnlyConfig`
  and `Int8DynamicConfig` quantisers parallel to the existing
  `torch.quantization` flow.
- `i3/encoder/quantize.py` — TCN INT8 dynamic quantisation counterpart.
- `i3/mlops/` — `ExperimentTracker` (MLflow soft-import), `Registry`
  (filesystem plus optional MLflow / W&B mirror), `save_with_hash` /
  `load_verified` checkpoint integrity with SHA-256 sidecar and JSON
  metadata (git SHA, torch version, config hash, hardware) and
  `ModelSigner` wrapping OpenSSF Model Signing v1.0 (sigstore / PKI /
  bare key backends).
- ONNX export — `i3/encoder/onnx_export.py` (dynamic batch + time
  axes, parity `atol=1e-4`) and `i3/slm/onnx_export.py` (prefill-only
  graph with `conditioning_tokens` as an explicit input).
- ExecuTorch hooks — `i3/edge/` with `export_slm_to_executorch`,
  `export_tcn_to_executorch`, and Huawei-target hooks under
  `i3/huawei/`.
- `i3/cloud/guardrails.py` + `guarded_client.py` — input guardrail
  (length cap, prompt-injection keyword list, loop detection) and
  output guardrail (sensitive-token redaction, length trim) composing
  around the existing `CloudLLMClient` without modifying it.
- `i3/eval/` — sliding-window perplexity, cross-attention
  conditioning-sensitivity KL divergence, and a responsiveness
  golden set (12-example tone-class harness).
- `i3/slm/aux_losses.py` — `ConditioningConsistencyLoss` and
  `StyleFidelityLoss`.
- `i3/interpretability/` — `FeatureAttributor` via integrated
  gradients, `CrossAttentionExtractor`, `TokenHeatmap`, optional
  SHAP backend.
- `i3/adaptation/ablation.py` — `AblationController` composition.
- `i3/biometric/` — keystroke-biometric user identification and
  continuous authentication.

#### Research experiments and interpretability

- Preregistered empirical ablation study —
  `docs/experiments/preregistration.md`,
  `i3/eval/ablation_experiment.py`, `i3/eval/ablation_statistics.py`,
  CLI, and a paper-style report.
- Mechanistic interpretability study —
  `i3/interpretability/activation_patching.py`,
  `probing_classifiers.py`, `attention_circuits.py`, CLI, and a 3 000
  word paper.
- ImplicitAdaptBench benchmark — `benchmarks/implicit_adapt_bench/`
  with `data_schema`, `metrics`, `data_generator`, three baselines,
  a scoring CLI, and a 2 500 word specification.
- Simulation-based closed-loop evaluation —
  `i3/eval/simulation/personas.py` with eight HCI personas,
  `user_simulator.py`, `closed_loop.py`, CLI, paper, and 31 tests.
- Uncertainty quantification and counterfactual explanations —
  `i3/adaptation/uncertainty.py` (MC Dropout),
  `i3/interpretability/counterfactuals.py`, a new
  `/api/explain/adaptation` endpoint, and a web panel.
- Sparse autoencoders for cross-attention interpretability —
  `i3/interpretability/sparse_autoencoder.py`, `activation_cache.py`,
  `sae_analysis.py`, `activation_steering.py`, `train` / `analyse`
  CLIs, and a 3 000 word paper.
- Provider-agnostic LLM-as-judge harness — `i3/eval/llm_judge.py`,
  `judge_rubric.py`, `judge_calibration.py`, `judge_ensemble.py`,
  CLI, and 34 tests.
- Red-team safety harness — 55 adversarial attacks across ten
  categories (`i3/redteam/attack_corpus.py`, `attacker.py` with four
  target surfaces, `policy_check.py`, CLI, weekly CI, a 2 500 word
  paper, and 30 tests).
- `scripts/security/run_redteam_notorch.py` — torch-stubbed runner so
  the sanitiser / PDDL / guardrails surfaces stay exercisable on
  Windows hosts where torch fails to load `c10.dll` (WinError 1114).

#### Huawei ecosystem integration

- `docs/huawei/` — `harmony_hmaf_integration.md` (HMAF four pillars,
  distributed databus sync format `I3UserStateSync`, ~680 bytes),
  `kirin_deployment.md` (chip-by-chip budgets for Kirin 9000 / 9010 /
  A2 / Smart Hanhan with Da Vinci op coverage), `l1_l5_framework.md`
  (Eric Xu's device-intelligence ladder with capability / primitive /
  privacy triples for L3–L5), `edinburgh_joint_lab.md`,
  `smart_hanhan.md`, and `interview_talking_points.md`.
- `i3/huawei/hmaf_adapter.py` — typed `HMAFAgentAdapter` with
  `register_capability` / `plan` / `execute` / `emit_telemetry`
  plus a raw-text telemetry guard.
- `i3/huawei/kirin_targets.py` — Pydantic v2 frozen `DeviceProfile`
  with four canonical devices and `select_deployment_profile()`.
- `i3/huawei/agentic_core_runtime.py` — runnable HMAF agentic runtime.
- PDDL-grounded privacy-safety planner — `i3/safety/pddl_planner.py`
  plus `server/routes_translate.py` (AI-Glasses translation endpoint)
  and a 2 230 word agentic-core document.
- Speculative decoding and an adaptive fast / slow compute router
  (`i3/slm/speculative_decoding.py`, `i3/router/adaptive_compute.py`;
  2 228 word research note).
- PPG / HRV wearable signals (`i3/multimodal/ppg_hrv.py`,
  `wearable_ingest.py` with six vendor formats,
  `i3/huawei/watch_integration.py`; 2 500 word paper).

#### Universal LLM provider layer

- `i3/cloud/providers/` — eleven first-class adapters (Anthropic,
  OpenAI, Google, Azure, Bedrock, Mistral, Cohere, Ollama,
  OpenRouter, LiteLLM, Huawei PanGu).
- `i3/cloud/multi_provider.py` — sequential / parallel / best-of-N
  strategies plus a circuit breaker.
- `i3/cloud/prompt_translator.py`, `i3/cloud/cost_tracker.py` with
  2026 pricing, eleven configuration fragments, and
  `.env.providers.example`.

#### Multimodal, continual, and meta-learning

- Voice prosody, facial affect, and multimodal fusion —
  `i3/multimodal/voice_real.py` (8-dim prosody via librosa),
  `vision.py` (MediaPipe Face Mesh with eight landmark-derived AUs),
  `fusion_real.py` (three strategies), a live webcam+mic demo, and a
  2 500 word paper.
- Adaptation-conditioned TTS — `i3/tts/conditioning.py` maps
  `AdaptationVector` → `TTSParams`, three soft-imported backends,
  `server/routes_tts.py`, a web player, and a 2 100 word research
  note.
- Active preference learning / online DPO —
  `i3/router/preference_learning.py` (Bradley-Terry + Mehta 2025
  active selection), `server/routes_preference.py`, a web A/B panel.
- Elastic Weight Consolidation and drift-triggered consolidation —
  `i3/continual/ewc.py`, `online_ewc`, `drift_detector.py` (ADWIN),
  `ewc_user_model.py` composable wrapper.
- MAML + Reptile meta-learning — `i3/meta_learning/maml.py`,
  `reptile.py`, `few_shot_adapter.py`, `task_generator.py`.

#### Edge runtimes

- `i3/edge/{mlx,llama_cpp,tvm,iree,coreml,tensorrt_llm,openvino,
  mediapipe}_export.py`.
- `scripts/export/all_runtimes.py` and
  `benchmarks/test_edge_runtime_matrix.py`.
- `docs/edge/alternative_runtimes.md` — an 8-runtime decision matrix
  (3 512 words) with a Huawei NPU column.

#### Cloud ecosystem integrations

- `i3/cloud/dspy_adapter.py`,
  `i3/cloud/guardrails_nemo.py` + `configs/guardrails/i3_rails.co`,
  `i3/cloud/pydantic_ai_adapter.py`, `i3/cloud/instructor_adapter.py`,
  `i3/cloud/outlines_constrained.py`,
  `i3/observability/logfire_integration.py`,
  `i3/observability/openllmetry.py` (OpenTelemetry GenAI semantic
  conventions for Anthropic).
- `docs/cloud/llm_ecosystem.md`.

#### Distributed training and serving

- `training/train_encoder_fabric.py`, `training/train_slm_fabric.py`
  (FSDP + `torch.compile` max-autotune),
  `training/train_with_accelerate.py`,
  `training/train_with_deepspeed.py`,
  `configs/distributed/ds_config_zero3.json`.
- `i3/serving/ray_serve_app.py`, `i3/serving/triton_config.py`,
  `i3/serving/vllm_server.py`.
- `deploy/serving/docker-compose.triton.yml`,
  `deploy/serving/ray_serve_manifest.yaml`.
- `docs/research/distributed_training.md`.

#### Modern data stack

- `i3/analytics/` — DuckDB analytics attached READ_ONLY to SQLite,
  LanceDB IVF-PQ vector search, Polars streaming features, Ibis
  portable queries, and Arrow / Parquet interop.
- `scripts/demos/analytics_dashboard.py`,
  `scripts/export/diary_to_parquet.py`,
  `scripts/experiments/find_similar_users.py`.
- `docs/operations/analytics.md`.

#### Toolchain and developer experience

- `uv.toml`, `.python-version`, `.tool-versions`, `.mise.toml`,
  `justfile`, `devbox.json`, `flake.nix` + `flake.lock`, `.envrc`,
  `scripts/uv_bootstrap.sh`, `scripts/verify_reproducibility.sh`.
- `.github/workflows/uv-ci.yml`.
- `docs/operations/uv_migration.md`, `docs/operations/reproducibility.md`.
- `dagger/main.py` — programmable CI via the Dagger Python SDK.
- `backstage/catalog-info.yaml` plus TechDocs, System, and API
  entities.
- `.github/codespaces/devcontainer.json`.
- `docs/operations/developer_experience.md`.

#### Documentation site (MkDocs Material)

- `mkdocs.yml` configured for the Material theme, light / dark palette
  toggle, and the Mermaid / MathJax / pymdownx extensions suite.
- `docs/adr/` — 10 Architecture Decision Records.
- `docs/architecture/full-reference.md` — 820-line comprehensive
  reference.
- Extensive `docs/research/` and `docs/huawei/` collections (see
  MkDocs nav).

#### Testing

- 80+ test modules under `tests/` covering unit, property
  (Hypothesis), contract (schemathesis), snapshot (syrupy), fuzz,
  load (locust soak), mutation (mutmut), and chaos scenarios.
- `tests/load/test_soak_30min.py` — 30-minute WebSocket soak test
  with a `psutil` memory-delta ≤ 50 MB guard.

#### Deliverables

- `docs/paper/I3_research_paper.md` — 7 126-word IEEE/ACM-style draft.
- `docs/paper/references.bib` — 28 entries.
- `docs/paper/executive-summary.md` — 1 380 word plain-English
  summary.
- `docs/patent/provisional_disclosure.md` — attorney-ready invention
  disclosure with 3 independent and 7 dependent claims.
- `docs/poster/conference_poster.md`.
- `docs/slides/` — presentation, speaker notes, rehearsal timings,
  Q&A prep, closing lines, demo script.

#### Verification and red-team harnesses

- `scripts/verify_all.py` + `scripts/verification/` package with 46
  registered checks across seven categories (code integrity,
  configuration, runtime, providers, infrastructure,
  interview-readiness, security).
- `scripts/security/run_redteam.py` — the 55-attack adversarial
  harness.

#### Audit reports

- `reports/audits/2026-04-22-post-v1-security.md` — automated strict
  review of every post-v1.0 commit (0 critical / 0 high / 2 medium /
  4 low / 5 informational).
- `reports/audits/2026-04-22-code-quality.md` — module-by-module
  typing and docstring coverage.
- `reports/audits/2026-04-22-completeness.md` — ~92 % complete
  against the original specification with remaining human-action
  items.
- `reports/audits/2026-04-22-documentation.md` — coverage matrix,
  voice and tone findings, citation matrix, slide-deck compliance.
- `reports/audits/2026-04-23-security-review.md` — independent
  manual security review (2 high / 5 medium / 5 low / 8 positive).
- `reports/audits/2026-04-23-robustness-audit.md` — robustness /
  performance / code-quality audit (1 blocker / 9 high / 16 medium /
  14 low / 7 positive).
- `reports/audits/2026-04-23-fixes-applied.md` — per-finding fix
  log with file:line citations.
- `reports/audits/2026-04-23-index.md` — three-layer verification
  index.
- `reports/verification/` and `reports/redteam/` — machine-readable
  + Markdown outputs of both automated harnesses plus a four-pass
  history under `verification/history/`.

#### Security infrastructure

- `server/auth.py` — opt-in caller-identity dependency system
  (`require_user_identity`, `require_user_identity_from_body`) with
  two activation modes: a bearer-token map via `I3_USER_TOKENS` (a
  JSON object) or a simpler header match via `X-I3-User-Id`. Uses
  `secrets.compare_digest` throughout. Off by default
  (`I3_REQUIRE_USER_AUTH=1` to activate) so the demo workflow is not
  broken.

#### Scripts and tooling

- `scripts/` now grouped into subdirectories by purpose:
  `benchmarks/`, `demos/`, `experiments/`, `export/`, `security/`,
  `training/`, `verification/`. See `scripts/README.md`.
- Makefile targets for every common operation: `docker-build[-dev]`,
  `docker-up[-prod]`, `docs[-serve|-build|-strict|-deploy]`,
  `obs-up/down`, `benchmarks`, `export-onnx`, `verify-onnx`,
  `profile-edge`, `sign-model`, `eval-conditioning`, `verify`,
  `verify-strict`, `verify-quick`, `redteam`, `run-ablation`,
  `run-closed-loop`, `run-sae`, `run-llm-judge`, `run-ewc`,
  `run-maml`, `run-hmaf`.

#### Frontend

- `web/js/attention_viz.js` — live 4×4 cross-attention heatmap.
- `web/js/whatif.js` — side-by-side alternative-adaptation responses.
- `web/js/persona_switcher.js` — four pre-built personas.
- `web/js/reset_button.js` — floating control.
- `web/js/wcag_audit.js` — in-browser WCAG 2.2 AA + AAA audit.
- `web/js/advanced_init.js` — Alt + A to toggle the advanced overlay.
- `web/advanced/` — seven-panel CSS-Grid layout with a Three.js 3-D
  embedding cloud, Chart.js metric graphs, SVG radial adaptation
  gauges with uncertainty bands, a 4×4 cross-attention heatmap, an
  Alt + T guided tour that walks the four demo phases autonomously,
  a screen-recording preset, a runtime WCAG 2.2 AA contrast audit,
  palette-disciplined output (six colours, no build step, vendor
  imports SRI-pinned).
- In-browser inference — `web/js/ort_loader.js` (ONNX Runtime Web
  1.18 with SRI), `webgpu_probe.js`, `encoder_worker.js`,
  `browser_inference.js`, `inference_toggle.js`,
  `inference_metrics_overlay.js`, `server/routes_inference.py` (with
  COOP/COEP headers and path-traversal guard). See
  `docs/research/browser_inference.md`.

#### Federated, privacy, cross-device, fairness

- `i3/federated/` — Flower client, FedAvg server, secure-aggregation
  stub.
- `i3/privacy/differential_privacy.py` — Opacus DP-SGD wrapper for
  the router posterior.
- `i3/crossdevice/` — HarmonyOS Distributed Data Management sync with
  Fernet + SHA-256 integrity, device registry, and AI Glasses
  paired-phone arm.
- `i3/fairness/` — per-archetype adaptation-bias, Efron 1979
  bootstrap CI, biometric FAR / FRR.

#### Notebooks

- Seven Jupyter notebooks: perception, TCN from scratch, three-
  timescale user model, cross-attention centrepiece, Thompson
  sampling, privacy by architecture, edge profiling.

### Changed

- Project layout — `scripts/` reorganised into topical subdirectories
  (`benchmarks/`, `demos/`, `experiments/`, `export/`, `security/`,
  `training/`, `verification/`). Alternate Dockerfiles
  (`Dockerfile.dev`, `Dockerfile.mcp`, `Dockerfile.wolfi`) moved into
  `docker/`. `SLSA.md` and `SUPPLY_CHAIN.md` moved into
  `docs/security/`. Audit reports consolidated under
  `reports/audits/`. Verification-harness output split into
  `reports/verification/` (with `history/`) and `reports/redteam/`.
  Internal process notes moved out of the repository (gitignored at
  `.internal/`).
- `pyproject.toml` — added `observability`, `mlops`, `ml-advanced`,
  `analytics`, `distributed`, `llm-ecosystem`, `providers`,
  `edge-runtimes`, `multimodal`, `future-work`, `policy`, `mcp`,
  `tts` Poetry groups. Expanded `dev` with Hypothesis, schemathesis,
  syrupy, mutmut, `pytest-benchmark`, `jsonschema`. Expanded `docs`
  with the MkDocs Material ecosystem plugins. Added `detect-secrets`
  to `security`.
- `.env.example` — documented observability (log format / level,
  OTel, Prometheus, Sentry, Langfuse), MLflow, benchmarks
  (`I3_BENCH_REQUIRE_CKPT`), runtime tuning (`I3_WORKERS`,
  `I3_FORWARDED_IPS`), OpenAPI disable (`I3_DISABLE_OPENAPI`), and
  CORS wildcard override.
- `server/app.py` — one-line non-destructive call to
  `setup_observability(config, app)` after the middleware stack.
- `mkdocs.yml` — navigation expanded to surface Huawei, Slides,
  Responsible AI, MLOps, Edge, Integration, Cloud, Paper,
  Reproducibility, Analytics, uv migration, Developer Experience,
  Policy as Code, Browser inference, Distributed training,
  Differential privacy, and Stretch goals.
- `scripts/verification/` — `_env_missing_result` /
  `_is_os_env_issue` now recognise `OSError WinError 1114`,
  `c10.dll`, `cudart`, `DLL load failed`, `KeyError` from
  partial-binary-import cascades, and `AttributeError` on `torch`
  attributes as environment issues. The harness now correctly
  SKIPs them rather than reporting false FAILs on Windows hosts with
  broken torch. `HuaweiPanguProvider` → `HuaweiPanGuProvider`
  class-name drift was also fixed.

### Fixed

- `server/websocket.py` — `process_keystroke` is `async def` but
  was called without `await`. Every keystroke event from every
  WebSocket client was previously dropped on the floor; the
  behavioural-baseline feature window was fed the zero-metric
  fallback for every user. Now awaited correctly.
- Rate limiter — switched from an include-list (`/api/*` only) to an
  exclude-list. Previously `/whatif/*` silently bypassed throttling,
  exposing a trivial DoS / GPU-burn vector. Every route now inherits
  the limiter by default (`server/middleware.py`).
- Preference routes — free-text prompts and A/B responses now pass
  through `PrivacySanitizer` before persistence, and the read
  endpoints are gated on `require_user_identity` to prevent
  cross-user PII harvesting.
- Five POST routes now gated by `require_user_identity_from_body`:
  `POST /whatif/respond`, `POST /whatif/compare`, `POST /api/tts`,
  `POST /api/translate`, `POST /api/preference/record`,
  `POST /api/explain/adaptation`.
- User routes in `server/routes.py` (`get_user_profile`,
  `get_user_diary`, `get_user_stats`) gated by
  `require_user_identity`.
- `i3/diary/store.py` — persistent `aiosqlite` connection held for
  the lifetime of the store with WAL journal and
  `PRAGMA foreign_keys = ON` applied once. The previous
  per-operation open/close cost 5–30 ms and reset the FK pragma on
  every call (FK enforcement was effectively off, allowing orphan
  exchanges). Adds idempotent `close()`; 10 call sites migrated via
  a drop-in async context manager.
- `i3/pipeline/engine.py::_generate_response` — SLM generation is
  now `await loop.run_in_executor(...)` instead of blocking the
  event loop. `generate_session_summary` is wrapped in
  `asyncio.wait_for` with a `timeout * 1.2` budget (previously a
  ~45 s tail risk on slow upstreams).
- `Pipeline.user_models` — `OrderedDict` capped at
  `I3_MAX_TRACKED_USERS` (default 10 000) with O(1) LRU eviction
  and full per-user footprint cleanup (response-time, length,
  engagement, and previous-route dicts are all cleared on eviction).
  Fixes the linear memory leak on long-lived servers or under
  id-rotation traffic.
- `i3/router/bandit.py` — `select_arm` / `update` /
  `_refit_posterior` now serialised under a reentrant lock. History
  converted to `deque(maxlen=_MAX_HISTORY_PER_ARM)` for O(1)
  overflow (previously O(n) slice churn). Concurrent eight-thread /
  800-operation stress test produces consistent `total_pulls`.
- `httpx.AsyncClient` lazy-init races — `i3/cloud/client.py`
  (`asyncio.Lock`), `openrouter.py` / `ollama.py` /
  `huawei_pangu.py` (double-checked `threading.Lock`). Previously
  two concurrent first-hit callers each built a client and the
  loser's connection pool leaked.
- `i3/pipeline/engine.py::_build_error_output` —
  `"error": type(exc).__name__` replaced with the constant
  `"pipeline_error"`. Exception class names no longer leak to the
  WebSocket `state_update` frame.
- `server/routes_explain.py::_surrogate_mapping_fn` — scoped
  `torch.Generator` and module-level layer cache. Previously every
  explain request called `torch.manual_seed(0xA11CE)`, a global
  side-effect that silently broke Thompson-sampling exploration in
  every other coroutine in flight.
- Router `prior_alpha` / `prior_precision` semantic mismatch —
  `RouterConfig` now carries both fields as distinct quantities
  (Beta prior α versus Gaussian weight precision);
  `IntelligentRouter` passes the right field. Operators tuning one
  no longer silently perturb the other.
- `server/app.py` — `load_config` is now called once in
  `create_app` and cached on `app.state.config`. Previously called
  twice (lifespan + `create_app`), re-seeding global RNGs twice on
  every startup.
- `server/app.py` — refuses to start when `I3_WORKERS > 1` without
  `I3_ALLOW_LOCAL_LIMITER=1`. The in-process sliding-window limiter
  silently multiplied the per-IP rate by worker count; loud failure
  now replaces silent mistuning.
- `i3/config.py::Config` — `extra="forbid"` on the root model;
  typoed top-level YAML sections (e.g. `saftey:`) fail loudly
  instead of being silently dropped.
- `i3/config.py::CloudConfig.model` — default changed from
  `"claude-sonnet-4-20250514"` to `"claude-sonnet-4-5"` to match
  `configs/default.yaml`.
- `i3/privacy/sanitizer.py::PrivacyAuditor._scan_value` —
  depth-capped at 32 with list-based path join (O(n) instead of
  O(n²)). Adversarially nested payloads no longer crash the audit.
- `i3/privacy/sanitizer.py::PrivacyAuditor._findings` —
  `deque(maxlen=1_000)`. Long-lived auditors no longer accumulate
  gigabytes on misconfiguration.
- `i3/interpretability/activation_cache.py::load` — manifest file
  size capped at 1 MiB, structural validation of the shape
  (`dict[str, list[str]]`), and each shard path `resolve()`-ed and
  `relative_to()`-checked against the cache root. Blocks `../`
  traversal via `index.json`.
- `server/middleware.py::DEFAULT_EXEMPT_PREFIXES` — dead entries
  `/docs`, `/redoc`, `/openapi.json` replaced with the actual mount
  points `/api/docs`, `/api/redoc`, `/api/openapi.json`.
- `server/routes_inference.py` — 404 detail narrowed to
  `"Model not found"`; the export-command hint now lives in the
  structured log only.
- `server/routes_translate.py` — `raise HTTPException(...) from
  exc` preserves the Pydantic cause in server-side logs.
- Cloud provider exceptions — `OpenRouter`, `Huawei PanGu`, and
  `Ollama` no longer echo `response.text` into exception messages;
  the body moves to `logger.debug`.
- Cloud provider HTTP clients — all four (Anthropic, OpenRouter,
  Ollama, Huawei PanGu) now pin `verify=True`,
  `follow_redirects=False`, and explicit `httpx.Limits(...)`.
- `i3/privacy/sanitizer.py` — IP-address regex now requires each
  octet ≤ 255, eliminating false-positive hits on Windows build
  numbers (e.g. `10.0.22621`), SemVer fragments, and telemetry
  counters.
- `server/routes_admin.py::admin_export` — returns 404 when the
  profile, diary, and bandit stats are all empty (removes the
  enumeration oracle via response shape).
- `server/middleware.py::_SlidingWindowLimiter` — `OrderedDict` +
  `popitem(last=False)` for amortised O(1) eviction (was O(n) via
  `min(..., key=...)`). Active keys call `move_to_end` so LRU
  ordering is fair.
- `i3/interaction/monitor.py::_UserSession.feature_window` — now a
  `deque(maxlen=feature_window_size)`. O(1) trim instead of
  `list.pop(0)` O(n).
- `i3/slm/train.py::load_checkpoint` — verifies an optional
  `<path>.sha256` sidecar before loading with constant-time compare
  and a loud warning when the sidecar is absent. Narrows the
  pickle-RCE blast radius of `weights_only=False`.
- `i3/interpretability/activation_cache.py` — `torch.load(...)`
  now `weights_only=True` on both the single-file and sharded
  paths.
- `i3/encoder/onnx_export.py` and `i3/slm/onnx_export.py` — CLI
  `print()` → `sys.stderr.write()`.
- `configs/default.yaml` — `cloud.model` pinned to
  `claude-sonnet-4-5`.
- `docs/slides/presentation.md` — honesty slide title now the
  verbatim Title Case "What This Prototype Is Not".
- `.github/workflows/trivy.yml` — pinned
  `aquasecurity/trivy-action@0.24.0` (was `@master`).
- `.github/workflows/semgrep.yml` — pinned
  `semgrep/semgrep:1.78.0` (was `:latest`).

### Security

After the two audit passes and fixes:

- `scripts/verify_all.py --strict` — **27 pass / 0 fail / 19
  skip**. Every skip is environment-gated (torch DLL on Windows;
  `ruff`, `mypy`, `helm`, `cedarpy`, `mkdocs` not on PATH).
- Red-team harness invariants — **3 / 4 pass**
  (`privacy_invariant`, `sensitive_topic_invariant`,
  `pddl_soundness`). The fourth (`rate_limit_invariant`) fails
  only because the FastAPI target surface is not exercised on the
  affected host — not a code defect.
- Concurrency and correctness smoke tests — all pass. `Config`
  typos rejected; 8-thread bandit stress produces consistent
  `total_pulls`; `DiaryStore` FK pragma persists across operations;
  `require_user_identity` accepts correct and rejects incorrect
  tokens; IP regex distinguishes real IPs from build numbers;
  recursion capped at depth 32 in the auditor.

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
