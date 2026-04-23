# Project Structure

A map of every top-level directory and what lives in it. Open this file first
when you are finding your way around the repository for the first time.

```
implicit-interaction-intelligence/
├── .devcontainer/         # VS Code dev container definition
├── .github/               # GitHub Actions workflows, issue templates, Allstar
├── backstage/             # Spotify Backstage catalog (service catalog entries)
├── benchmarks/            # pytest-benchmark tests + ImplicitAdaptBench harness
├── checkpoints/           # Trained model artefacts (gitignored; regenerated)
├── configs/               # YAML configuration: default.yaml, training overrides
├── dagger/                # Dagger Python SDK — programmable CI/CD
├── data/                  # Bundled datasets and lexicons (small, checked in)
├── demo/                  # Pre-seeded demo state and fixtures
├── deploy/                # Kubernetes / Helm / Terraform / policy / serving
│   ├── argocd/
│   ├── helm/i3/
│   ├── k8s/               # Deployment + HPA + PDB + NetworkPolicy + Kustomize
│   ├── observability/     # OTel Collector + Prometheus + Tempo + Grafana
│   ├── policy/            # Kyverno / OPA / Cedar / Falco / Tracee / Sigstore
│   ├── serving/           # Triton, Ray Serve, vLLM manifests
│   └── terraform/         # AWS EKS reference module
├── docker/                # Dockerfile variants + entrypoint + healthcheck
│   ├── Dockerfile.dev     # Hot-reload development image
│   ├── Dockerfile.mcp     # MCP server image
│   └── Dockerfile.wolfi   # Chainguard Wolfi distroless image
├── docs/                  # MkDocs Material source tree
│   ├── adr/               # Architecture Decision Records (10 files)
│   ├── api/               # REST + WebSocket + Python SDK references
│   ├── architecture/      # Overview, layers, cross-attention, router, privacy
│   ├── cloud/             # LLM ecosystem + provider universality
│   ├── contributing/      # Contributor guide
│   ├── edge/              # Edge runtime matrix + profiling report
│   ├── experiments/       # Preregistration, ablation report
│   ├── getting-started/   # Installation, quick start, configuration, training
│   ├── huawei/            # HMAF, Kirin deployment, L1-L5, Edinburgh joint lab
│   ├── integration/       # MCP integration guide
│   ├── mlops/             # MLOps runbook
│   ├── operations/        # Deployment, observability, runbook, uv migration
│   ├── paper/             # Research paper + executive summary + references.bib
│   ├── patent/            # Provisional patent disclosure
│   ├── poster/            # Conference poster source
│   ├── research/          # Research notes (20+ pages)
│   ├── responsible_ai/    # Model cards, data card, accessibility statement
│   ├── security/          # SLSA, supply chain, policy as code, security index
│   └── slides/            # Presentation, speaker notes, Q&A prep, demo script
├── i3/                    # Main Python package (the product)
│   ├── adaptation/        # Controller + AdaptationVector + ablation
│   ├── analytics/         # DuckDB + LanceDB + Polars + Ibis
│   ├── authz/             # Cedar adapter
│   ├── biometric/         # Keystroke-biometric user ID + continuous auth
│   ├── cloud/             # CloudLLMClient, 11 provider adapters, guardrails
│   ├── config.py          # Pydantic v2 config schema
│   ├── continual/         # EWC + drift detector + online EWC
│   ├── crossdevice/       # HarmonyOS distributed-databus sync
│   ├── diary/             # Async SQLite interaction diary
│   ├── edge/              # ONNX + ExecuTorch + 8 alternative runtimes
│   ├── encoder/           # TCN encoder + loss + quantisation + ONNX export
│   ├── eval/              # Perplexity + conditioning KL + closed-loop + ablation
│   ├── fairness/          # Per-archetype bias + FAR/FRR + bootstrap CI
│   ├── federated/         # Flower client + FedAvg server + secure-agg stub
│   ├── huawei/            # HMAF adapter, Kirin targets, Watch integration
│   ├── interaction/       # Feature extraction, baseline, monitor
│   ├── interpretability/  # Counterfactuals, SAE, activation patching, probes
│   ├── mcp/               # Model Context Protocol server
│   ├── meta_learning/     # MAML + Reptile + task generator
│   ├── mlops/             # Experiment tracker + registry + signing
│   ├── multimodal/        # Voice, vision, PPG/HRV, touch, fusion
│   ├── observability/     # OTel + structlog + Prometheus + Sentry + Langfuse
│   ├── pipeline/          # The main async Pipeline (ingest → encode → route → …)
│   ├── privacy/           # Sanitiser, encryption, DP-SGD
│   ├── profiling/         # Edge-feasibility profiler
│   ├── redteam/           # Adversarial corpus + attacker + policy invariants
│   ├── router/            # Thompson sampling bandit + preference learning
│   ├── safety/            # PDDL planner + safety certificates
│   ├── serving/           # Ray Serve + Triton + vLLM
│   ├── slm/               # Custom ~6.3 M-param SLM + speculative decoding
│   ├── tts/               # Adaptation-conditioned TTS
│   └── user_model/        # Three-timescale user model + async SQLite store
├── notebooks/             # 7 teaching notebooks
├── reports/               # Audit + verification + red-team artefacts
│   ├── audits/            # Narrative review reports (dated, per-finding)
│   ├── redteam/           # 55-attack harness output (latest.{json,md})
│   └── verification/      # 46-check harness output (+ history/)
├── scripts/               # Operator entry points — see scripts/README.md
│   ├── benchmarks/        # Latency / edge profiling micro-benchmarks
│   ├── demos/             # Standalone feature demos
│   ├── experiments/       # Research runs (ablation, DPO, LLM-judge, …)
│   ├── export/            # Model + data export (ONNX, ExecuTorch, GDPR)
│   ├── security/          # Red-team, model signing, key generation
│   ├── training/          # Training entry points
│   └── verification/      # The 46 registered checks
├── server/                # FastAPI app (routes, middleware, WebSocket, auth)
├── tests/                 # 80+ test modules (unit, property, contract, …)
│   ├── benchmarks/        # pytest-benchmark
│   ├── chaos/             # Chaos-engineering scenarios
│   ├── contract/          # schemathesis OpenAPI contract tests
│   ├── fuzz/              # Hypothesis property-based fuzzing
│   ├── load/              # locust soak tests
│   ├── mutation/          # mutmut
│   ├── property/          # Hypothesis property tests
│   └── snapshot/          # syrupy snapshots
├── training/              # Training scripts (Fabric, Accelerate, DeepSpeed)
└── web/                   # Demo UI — plain CSS/JS, no build step
    ├── advanced/          # Cinematic command-centre UI at /advanced
    ├── css/
    └── js/
```

## Top-level files

| File | What it is |
|---|---|
| `README.md` | Project overview, demo, architecture summary, how to run. |
| `CHANGELOG.md` | Keep-a-Changelog history (semantic-versioned). |
| `CONTRIBUTING.md` | How to contribute — code style, branching, commit format. |
| `CODE_OF_CONDUCT.md` | Contributor Covenant. |
| `SECURITY.md` | Threat model, vulnerability reporting process. |
| `LICENSE` | MIT. |
| `PROJECT_STRUCTURE.md` | This file. |
| `pyproject.toml` | Poetry + tool configuration (ruff, mypy, pytest, coverage). |
| `Makefile` | Task runner (primary). |
| `justfile` | Alternative task runner. |
| `Dockerfile` | Production image (multi-stage, non-root, distroless-adjacent). |
| `docker-compose.yml` | Local development stack. |
| `docker-compose.prod.yml` | Production stack with TLS sidecar. |
| `mkdocs.yml` | Documentation site configuration. |
| `.env.example` | Template for environment variables (every one documented). |
| `.env.providers.example` | Per-provider API key template. |
| `flake.nix`, `.mise.toml`, `devbox.json`, `uv.toml` | Alternative tool-version managers. |
| `.pre-commit-config.yaml`, `lefthook.yml` | Pre-commit hooks. |
| `renovate.json` | Dependency update policy. |
| `commitlint.config.js` | Conventional-commit linting. |
| `.sigstore.yaml`, `.trivyignore`, `.semgrepignore` | Security-tool configuration. |

## Conventions

- **`i3/` is the product.** Everything importable lives there.
- **`server/` is the API wrapper.** FastAPI routes only — no ML logic.
- **`scripts/` is for humans, `i3/` is for programs.** A script is anything
  run directly; `i3/` modules must be importable.
- **`tests/` mirrors `i3/` where it matters.** Complex subsystems have
  dedicated test modules at the top level; integration + property + fuzz
  tests live in subdirectories.
- **`configs/` holds YAML, not Python.** Every setting a non-developer
  might want to change is surfaced in `configs/default.yaml`.
- **`reports/` is generated output.** Everything under it is reproducible
  from the harnesses in `scripts/`.
- **`docs/` is the MkDocs source tree.** The live site is at
  `https://abailey81.github.io/implicit-interaction-intelligence/`.
