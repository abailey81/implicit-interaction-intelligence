# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-04-23)


### Features

* **analytics:** DuckDB + LanceDB + Polars + Ibis + Arrow — modern data stack ([bc73464](https://github.com/abailey81/implicit-interaction-intelligence/commit/bc73464a45b76b42ac4be1a85ab21b6bd32defdc))
* **browser:** in-browser TCN inference via ONNX Runtime Web + WebGPU ([2682775](https://github.com/abailey81/implicit-interaction-intelligence/commit/26827756023fc09e11251225b3e2d8a3ac2e4930))
* **cloud:** Batch G7 — universal LLM provider layer (11 providers, fallback chain, cost tracker) ([da20671](https://github.com/abailey81/implicit-interaction-intelligence/commit/da206718732034c7f9012bc4b20d7bbaf27a6ccd))
* **compute:** speculative decoding (Celia parallel) + adaptive fast/slow router (PanGu 5.5 parallel) ([0a09eba](https://github.com/abailey81/implicit-interaction-intelligence/commit/0a09ebaf83f3c236bdb94126f74960b7e891bc69))
* **continual:** Batch F-5 — Elastic Weight Consolidation + drift-triggered auto-consolidation ([869b752](https://github.com/abailey81/implicit-interaction-intelligence/commit/869b7521fe0207616b8ae69d1f441b0075a0c8e3))
* **core:** behavioural perception, TCN encoder, user model, adaptation, router ([14859d2](https://github.com/abailey81/implicit-interaction-intelligence/commit/14859d27b2b895ec2bfb312c9de27c4d4d9d0487))
* **core:** NOTES.md disclosure, NT-Xent loss module, valence lexicon ([04cb9cd](https://github.com/abailey81/implicit-interaction-intelligence/commit/04cb9cd5469d5189b8d60a43f9cab16d77168112))
* **demo:** admin endpoints, pre-seed, 4-phase runner, soak test, preflight, GDPR export ([f350f7a](https://github.com/abailey81/implicit-interaction-intelligence/commit/f350f7aa211563956f79d926024577e8b1d463c4))
* **deploy:** Kubernetes manifests, Helm chart, Terraform, Skaffold, ArgoCD ([2390581](https://github.com/abailey81/implicit-interaction-intelligence/commit/23905815d95309bcd71f5eb28a860d57e9748e07))
* **devex:** Dagger + Tilt + Pyroscope + Backstage + Grafana Alloy + Codespaces ([b7f2584](https://github.com/abailey81/implicit-interaction-intelligence/commit/b7f25842207856a5ff3b0b39b396610f448ef3e2))
* **distributed:** Lightning Fabric + Ray Serve + Triton + DeepSpeed + vLLM scaffolds ([c20aa45](https://github.com/abailey81/implicit-interaction-intelligence/commit/c20aa45b37aeb9fe0d2c956ecac2c5d47e2749c9))
* **docker:** production-grade multi-stage container + devcontainer ([fb49c0a](https://github.com/abailey81/implicit-interaction-intelligence/commit/fb49c0a05f0a3fb298407cb88aea332a7bbb362e))
* **edge:** MLX + llama.cpp GGUF + TVM + IREE + CoreML + TensorRT-LLM + OpenVINO + MediaPipe ([a079da4](https://github.com/abailey81/implicit-interaction-intelligence/commit/a079da491024f4fd78807e3f6ccd87bdf567ae7c))
* **eval:** Batch G4 — provider-agnostic LLM-as-judge evaluation harness ([897360f](https://github.com/abailey81/implicit-interaction-intelligence/commit/897360f0b67ca1b8aab0f173b12e05e3334cb590))
* **experiments:** Batch A — preregistered empirical ablation study ([20a61d3](https://github.com/abailey81/implicit-interaction-intelligence/commit/20a61d3038de29d67319ccdcdfa3bcebd7829d0b))
* **experiments:** Batch B — mechanistic interpretability study ([7ac024c](https://github.com/abailey81/implicit-interaction-intelligence/commit/7ac024c8cc431eeea4009fae5e7d4748d23a34a9))
* **experiments:** Batch C — ImplicitAdaptBench benchmark + baselines ([3a239cb](https://github.com/abailey81/implicit-interaction-intelligence/commit/3a239cbbe36ae2ed82ab0bc546d81ed87342d23a))
* **experiments:** Batch G1 (completion) — closed-loop evaluator + CLI + docs + tests ([5a9c91f](https://github.com/abailey81/implicit-interaction-intelligence/commit/5a9c91f083472323dfde6ba2c461fd1748950f9b))
* **experiments:** Batch G1 (partial) — HCI persona library + user simulator ([4e44245](https://github.com/abailey81/implicit-interaction-intelligence/commit/4e44245c9b590fcd879cab8f35912d13b3f164e3))
* **frontend+notebooks:** attention heatmap, what-if, persona, WCAG audit + 7 teaching notebooks ([9459ac2](https://github.com/abailey81/implicit-interaction-intelligence/commit/9459ac22d8fff4cb66e4710d510ea1623a837e6d))
* **future-work:** multi-modal, federated, cross-device, fairness, DP sketches ([5ebe7a0](https://github.com/abailey81/implicit-interaction-intelligence/commit/5ebe7a04e4191728f087446b9e8c640b99bcb1d6))
* **huawei:** HMAF integration, Kirin deployment, L1-L5, Edinburgh, talking points ([27f0a4d](https://github.com/abailey81/implicit-interaction-intelligence/commit/27f0a4d71c9afa291f809b45a7688a3d89d8c3e4))
* **interp:** Batch G2 — uncertainty quantification + counterfactual explanations ([e62f9dd](https://github.com/abailey81/implicit-interaction-intelligence/commit/e62f9dda0a8502e144107df0816c8376c6052265))
* **interp:** Batch G3 — sparse autoencoders for cross-attention interpretability ([37a9620](https://github.com/abailey81/implicit-interaction-intelligence/commit/37a9620366f8a0fcfa12969e998ca5c770176807))
* **llm-ecosystem:** DSPy + NeMo Guardrails + Pydantic AI + Instructor + Outlines + Logfire + OpenLLMetry ([bd1ec1f](https://github.com/abailey81/implicit-interaction-intelligence/commit/bd1ec1f3e47370e5f2ddbe70145188f18b55fe74))
* **mcp:** Anthropic Model Context Protocol server + Claude Desktop integration ([bcfbf26](https://github.com/abailey81/implicit-interaction-intelligence/commit/bcfbf26e19178f461ef816abfe86f80da7cbd815))
* **meta:** Batch G5 — MAML + Reptile for few-shot user adaptation ([92e5915](https://github.com/abailey81/implicit-interaction-intelligence/commit/92e591580cdea5da9897257211ef177841dd00b8))
* **ml-advanced:** Langfuse + torchao INT4 + OpenSSF signing + ExecuTorch + guardrails + eval ([2848eac](https://github.com/abailey81/implicit-interaction-intelligence/commit/2848eac6a522dda1df507c4004956dd3c770f5b9))
* **mlops:** MLflow tracking, ONNX export, benchmarks, Locust/k6, DVC pipeline ([e42f6e3](https://github.com/abailey81/implicit-interaction-intelligence/commit/e42f6e310098c97e065b444e9b8e15215f402b7d))
* **multimodal:** Batch F-1 — voice prosody + facial affect + runnable fusion ([bcd71b5](https://github.com/abailey81/implicit-interaction-intelligence/commit/bcd71b59fc33c71a07a7ebe656c414e4015180ed))
* **multimodal:** Batch F-2 — PPG/HRV wearable signals (Huawei Watch 5 parallel) ([64c1e86](https://github.com/abailey81/implicit-interaction-intelligence/commit/64c1e868ef1c2b82aa97d9c8d5ac08f517effed8))
* **observability:** structlog + OpenTelemetry + Prometheus + optional Sentry ([026659a](https://github.com/abailey81/implicit-interaction-intelligence/commit/026659a839a4f623cfb0926b237d50dd3bbe5ea4))
* **pipeline:** diary, privacy, edge profiling, and orchestration engine ([d63f817](https://github.com/abailey81/implicit-interaction-intelligence/commit/d63f817d1caacfee12c49cca053a740b6c5adbc9))
* **policy:** Kyverno + OPA + Cedar + Falco + Tracee + Sigstore PC + Allstar ([a0ba6be](https://github.com/abailey81/implicit-interaction-intelligence/commit/a0ba6be0bacb15214edbe7f06fa1aca798e41f03))
* **router:** Batch F-4 — active preference learning / online DPO (closes reward loop) ([03d28b8](https://github.com/abailey81/implicit-interaction-intelligence/commit/03d28b85b5458be7cf0fdec125f1689e68a967f5))
* **safety+agentic+translate:** PDDL planner + runnable HMAF runtime + AI-Glasses translate endpoint ([2a3e251](https://github.com/abailey81/implicit-interaction-intelligence/commit/2a3e251f70441ffb70a3bd9f27a7c876d44dbe3f))
* **security:** Batch G6 — red-team safety harness (55 adversarial attacks, 4 target surfaces) ([2df0c29](https://github.com/abailey81/implicit-interaction-intelligence/commit/2df0c29172262185b0e27ac9a8e2cb23b752e785))
* **slides:** 15-slide interview deck, 52-item Q&A, speaker notes, rehearsal cues ([f8ed03c](https://github.com/abailey81/implicit-interaction-intelligence/commit/f8ed03cae5f2385f3c90adce87f45beef2af0b40))
* **slm:** custom transformer with cross-attention conditioning + cloud integration ([eee8195](https://github.com/abailey81/implicit-interaction-intelligence/commit/eee81953aaab0ef8ab1bd637ea9cfe0fbcc8bdfd))
* **stretch:** aux conditioning loss, interpretability, what-if, ablation, biometric ([660ac6b](https://github.com/abailey81/implicit-interaction-intelligence/commit/660ac6b34fbae9d17b8b34b62be75622a4809651))
* **supply-chain:** SBOM, cosign, SLSA L3, Scorecard, Trivy, Semgrep, release-please ([9cd6e0f](https://github.com/abailey81/implicit-interaction-intelligence/commit/9cd6e0f003a821301d48cf1c3173586399ff4644))
* **tests+data:** expand sentiment lexicon 5.6x and add 55 targeted tests ([7dc2592](https://github.com/abailey81/implicit-interaction-intelligence/commit/7dc25926854f4ae4f1fffcf77208548e29169b82))
* **tests:** property, contract, fuzz, load, mutation, chaos, snapshot suites ([7d5b3ba](https://github.com/abailey81/implicit-interaction-intelligence/commit/7d5b3ba9cefb1b8c6924c4296927af55fd2ce602))
* **training:** training scripts, synthetic data generation, and demo utilities ([d64bbc0](https://github.com/abailey81/implicit-interaction-intelligence/commit/d64bbc01a2d86493a7656fd8b1ab363db9a944bd))
* **tts:** Batch F-TTS — adaptation-conditioned speech synthesis (AI Glasses / Celia / Hanhan) ([83c1bb0](https://github.com/abailey81/implicit-interaction-intelligence/commit/83c1bb06d95db6dcb1f7befc1e0b54d8a4bd26f9))
* **ui:** Batch G9 — advanced cinematic command-center demo UI at /advanced ([d4ab940](https://github.com/abailey81/implicit-interaction-intelligence/commit/d4ab9400eaf9fb81bd0bdd783a08b1e8451894dd))
* **verify:** Batch G8 — comprehensive verification harness (46 PASS/FAIL/SKIP checks) ([56a447f](https://github.com/abailey81/implicit-interaction-intelligence/commit/56a447f488bfccb3627c63812348dffbed757d24))
* **web:** FastAPI backend, WebSocket handler, and dark-theme frontend ([9d000f5](https://github.com/abailey81/implicit-interaction-intelligence/commit/9d000f56342c841450a213da8662903ec111312c))


### Bug Fixes

* **audit:** address P0 findings from the four final-audit agents ([c0e2e2a](https://github.com/abailey81/implicit-interaction-intelligence/commit/c0e2e2a136b664b2265c041169569be08c04d9d4))
* **security+robustness:** wave 7 — deep audit fixes (1 blocker, 11 high, 10 medium, 7 low) ([534b780](https://github.com/abailey81/implicit-interaction-intelligence/commit/534b7806cfc8c8f39cf4ea2726efd5ed33a7a76a))
* **verify:** Batch G10 — 4 iterative verification passes, 14 FAIL → 0 FAIL, exit 0 strict ([108c256](https://github.com/abailey81/implicit-interaction-intelligence/commit/108c256948a14add53fbd22d4e1c7bc631214dc3))


### Documentation

* **academic:** research paper, patent disclosure, executive summary, conference poster ([df844ac](https://github.com/abailey81/implicit-interaction-intelligence/commit/df844ac421206e4703e28c69f130c88c831da073))
* **alignment:** Huawei-anchored advancement plan + JD-grounded interview pivots ([461e4b0](https://github.com/abailey81/implicit-interaction-intelligence/commit/461e4b013b56ddc6ed7c5dc6e32ecea85b3cc9d4))
* **plan:** ADVANCEMENT_PLAN v3 appendix — empirical-rigour + ecosystem-fit batches ([7bd9d8d](https://github.com/abailey81/implicit-interaction-intelligence/commit/7bd9d8d62ba7b27673f2a6250050e4733110bf26))
* **reports:** edge profiling report, model cards, data card, accessibility statement ([9b3146c](https://github.com/abailey81/implicit-interaction-intelligence/commit/9b3146c9d870609395b240a3892da559c713aecd))
* **site:** add trailing MkDocs pages (changelog include, contributing, security) ([e2c119e](https://github.com/abailey81/implicit-interaction-intelligence/commit/e2c119ef0c0fcf7ac9ee6fb29b77011f1b756caf))
* **site:** MkDocs Material documentation site + 10 ADRs ([0662dee](https://github.com/abailey81/implicit-interaction-intelligence/commit/0662dee4b4f4d0901ba5ed4643f72ee08c106fd5))

## [Unreleased]

_No unreleased changes; see [1.1.0] below._

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
