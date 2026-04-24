# Implicit Interaction Intelligence (I³)

An adaptive AI companion that learns from *how* a user interacts — keystroke
dynamics, linguistic complexity, temporal patterns — and conditions every
downstream decision (routing, generation, TTS) on that evolving signal.

The system is a full async pipeline, built from first principles in PyTorch:
a custom TCN encoder, a ~6.3 M-parameter transformer with cross-attention
conditioning, a contextual Thompson-sampling bandit that decides when to
route to a foundation model, and a privacy-preserving diary that persists
only embeddings and aggregate metrics — never raw text.

- **Docs site:** <https://abailey81.github.io/implicit-interaction-intelligence/>
- **License:** MIT — see [`LICENSE`](LICENSE)
- **Supported:** Python 3.10 / 3.11 / 3.12, PyTorch 2.0 +, Linux + macOS + Windows

---

## Table of contents

1. [Quickstart — one command](#quickstart--one-command)
2. [Why this project exists](#why-this-project-exists)
3. [Architecture](#architecture)
4. [Novel contribution — cross-attention conditioning](#novel-contribution--cross-attention-conditioning)
5. [Repository layout](#repository-layout)
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Training the models](#training-the-models)
10. [Running the server](#running-the-server)
11. [Using the API](#using-the-api)
12. [The demo UI](#the-demo-ui)
13. [Docker and Docker Compose](#docker-and-docker-compose)
14. [Kubernetes and Helm](#kubernetes-and-helm)
15. [Testing and verification](#testing-and-verification)
16. [Observability](#observability)
17. [Edge deployment](#edge-deployment)
18. [Privacy and security](#privacy-and-security)
19. [Command reference (`make`)](#command-reference-make)
20. [Contributing](#contributing)
21. [Further reading](#further-reading)

---

## Quickstart — one command

Two canonical run profiles, both driven by a single orchestrator that
shows a live ``rich`` dashboard with spinners, progress bars, ETAs,
per-stage log tails, and a calibrated-to-your-machine total projection.

```bash
git clone https://github.com/abailey81/implicit-interaction-intelligence.git
cd implicit-interaction-intelligence

# ── Option A — Fast (≈ 5 minutes on a laptop) ──
# prereq → install → .env + encryption key → demo seed → verification → serve
make all-fast
#   ▶ opens http://127.0.0.1:8000/  (UI) and /api/docs (Swagger)

# ── Option B — Full end-to-end (hours on CPU, ≈ 15 min on GPU) ──
# Everything in fast mode PLUS:
#   synthetic data → dialogue corpus → TCN encoder training →
#   SLM training → evaluation → conditioning sensitivity →
#   lint → typecheck → tests → bandit + pip-audit → red-team →
#   verification harness → benchmarks → ONNX export → edge profiling →
#   MkDocs build → serve
make all-full
```

Both targets call [`scripts/run_everything.py`](scripts/run_everything.py),
which you can invoke directly for finer control:

```bash
# See every stage the orchestrator knows about (fast + full tables)
make all-list

# Skip stages whose outputs already exist on disk (useful after a crash)
make all-resume

# Only run specific stages
python scripts/run_everything.py --mode full --only install,data,train-encoder

# Skip stages
python scripts/run_everything.py --mode full --skip benchmarks,docs

# Run everything except launching the server at the end
python scripts/run_everything.py --mode full --no-serve

# Include the Docker image-build stage (off by default)
python scripts/run_everything.py --mode full --with-docker
```

### What the orchestrator shows

```
┌────────────────────── Pipeline status ──────────────────────┐
│ #   Stage           Category  Description         Status   Elapsed   ETA │
│ ✓   prereq          setup     Check Python + …    done      0.2s     — │
│ ✓   install         setup     Install deps …      done    184.3s     — │
│ ✓   env             setup     Create .env + …     done      0.3s     — │
│ ✓   data            data      Generate synth …    done     27.1s     — │
│ ●   train-encoder   train     Train TCN (NT-…)    running  12.4m  28.7m │
│ ·   train-slm       train     Train SLM …         pending     —   50.0m │
│ ·   evaluate        eval      Perplexity + …      pending     —    2.0m │
│ …                                                                      │
└────────────────────────────────────────────────────────────────────────┘
 ⠋ train-encoder — Train TCN encoder     ████████░░░░  42/100   12:25  ETA 17:35
┌─────────── live log tail — train-encoder ───────────┐
│ epoch  4/10  step  120/480  loss 2.431  lr 2.8e-04  │
│ epoch  4/10  step  160/480  loss 2.388  lr 2.9e-04  │
│ epoch  4/10  step  200/480  loss 2.341  lr 3.0e-04  │
│ epoch  4/10  step  240/480  loss 2.302  lr 3.0e-04  │
└─────────────────────────────────────────────────────┘
```

- **Per-stage spinner + bar** — updates 6 Hz from a heartbeat thread so
  quiet stages still move.
- **ETA** — seeded from historical timings in
  `reports/orchestration.json`; after one full run the bars calibrate
  to your exact machine via EMA (α = 0.7).
- **Live log tail** — last 6 lines of stdout/stderr for the current
  stage; full log at `reports/orchestration/<stage>.log`.
- **Failure panel** — on any non-zero exit, the orchestrator stops and
  prints the last 12 lines of the offending log.
- **Category colour-coding** — setup, data, train, eval, quality,
  security, verify, perf, docs, deploy, serve.

### Resuming after a crash / re-running incrementally

```bash
# Skips every stage whose output file already exists.
make all-resume
# Equivalent: python scripts/run_everything.py --mode full --resume
```

### Flags cheat-sheet

| Flag | Effect |
|---|---|
| `--mode fast` / `--mode full` | Choose profile (default: `fast`) |
| `--only NAMES` | Run only the named stages (comma-separated) |
| `--skip NAMES` | Skip the named stages |
| `--resume` | Skip stages whose outputs already exist |
| `--no-serve` | Do not launch the server at the end |
| `--list` | Print the stage graph for the chosen mode |
| `--skip-install` | Assume the env is ready, skip `poetry install` |
| `--skip-onnx` | Skip ONNX export (full mode) |
| `--skip-docs` | Skip MkDocs build (full mode) |
| `--skip-benchmarks` | Skip benchmarks (full mode) |
| `--with-docker` | Include Docker build stage (full mode) |
| `--sessions-per-archetype N` | Synthetic data size (default 400) |
| `--encoder-epochs N` | TCN encoder epochs (default 10) |
| `--slm-epochs N` | SLM epochs (default 5) |

### Minimum environment for the fast path

Just two env vars (the orchestrator creates both for you):

| Var | Source |
|---|---|
| `I3_ENCRYPTION_KEY` | Auto-generated into `.env` by the `env` stage |
| `ANTHROPIC_API_KEY` | *Optional.*  Paste into `.env` to exercise the cloud arm; falls back to the local SLM if absent.  [console.anthropic.com](https://console.anthropic.com/). |

Every other env var ([`.env.example`](.env.example) · [`.env.providers.example`](.env.providers.example))
has a sensible default.

---

## Why this project exists

Conversational AI systems today respond to *what* a user types. They parse
tokens, run attention, and return a response. But humans — especially
close friends and collaborators — notice something richer: they notice
*how* we say things. A friend knows when you are tired from the cadence
of your replies, when you are stressed from the rhythm of your typing,
when you are struggling from the effort you are putting into a single
sentence.

I³ is an attempt to build that capability into a companion. It observes
keystroke dynamics, typing patterns, and linguistic complexity to build
an evolving model of each user's cognitive state, communication style,
and accessibility needs. The model then conditions response generation
at every level — from the routing decision (local SLM vs. cloud LLM)
down to the token-by-token cross-attention inside a custom transformer.

Three tiers of AI capability are exercised end to end:

1. **A custom ML encoder** — a TCN with dilated causal convolutions,
   trained with NT-Xent contrastive loss from scratch.
2. **A custom small language model** — a ~6.3 M-parameter transformer
   with no HuggingFace dependency, featuring cross-attention
   conditioning tokens derived from the user-state embedding.
3. **Intelligent routing to foundational models** — a Bayesian
   logistic-regression bandit with Laplace-approximated posterior
   and Thompson sampling, with a privacy override that forces
   local-SLM routing on sensitive topics.

Every non-trivial component is implemented directly: the encoder, the
transformer, the cross-attention conditioning, the Thompson sampling
bandit, the sentiment lexicon, the cosine warmup scheduler, the Fernet
key-rotation wrapper, the PDDL safety planner, and so on.

---

## Architecture

```
USER KEYSTROKE
      │
      ▼
┌─────────────────────┐
│  Perception         │  32-dim InteractionFeatureVector
│  i3/interaction/    │  (keystroke dynamics, linguistic, session)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  TCN Encoder        │  64-dim user-state embedding
│  i3/encoder/        │  (dilations [1,2,4,8], NT-Xent contrastive)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  User Model         │  Instant / Session / Long-term EMAs
│  i3/user_model/     │  (Welford's online algorithm)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Adaptation         │  8-dim AdaptationVector
│  i3/adaptation/     │  (cognitive load, style, tone, accessibility)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Router             │  Contextual Thompson sampling
│  i3/router/         │  (Bayesian logistic regression + Laplace)
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌────────┐
│  SLM   │   │ Cloud  │   Local ~6.3 M-param SLM, or any of
│ i3/slm/│   │i3/cloud│   11 first-class provider adapters
└────┬───┘   └────┬───┘   (Anthropic, OpenAI, Google, Azure, Bedrock,
     └─────┬─────┘         Mistral, Cohere, Ollama, OpenRouter,
           ▼               LiteLLM, Huawei PanGu)
┌─────────────────────┐
│  Diary              │  Embeddings + topics only — raw text never stored
│  i3/diary/          │
└─────────────────────┘

Cross-cutting: i3/privacy/  ·  i3/safety/  ·  i3/profiling/  ·  i3/observability/
```

Orchestration lives in [`i3/pipeline/engine.py`](i3/pipeline/engine.py).
The full design rationale, math, and data flow is in
[`docs/architecture/full-reference.md`](docs/architecture/full-reference.md).

### Layer-by-layer

- **Perception (`i3/interaction/`).** Extracts a 32-dim
  `InteractionFeatureVector` per message: keystroke dynamics
  (inter-key intervals, burst / pause ratios, correction rate),
  message content (length, word length, vocabulary diversity),
  linguistic complexity (Flesch–Kincaid grade, formality, sentiment
  valence), and session dynamics (deviation from the user's baseline).
- **Encoding (`i3/encoder/`).** A TCN of stacked dilated causal
  convolutions (dilations `[1, 2, 4, 8]`) compresses the feature
  sequence into a 64-dim user-state embedding. Trained with NT-Xent
  contrastive loss; exportable to ONNX with parity `atol = 1e-4`.
- **User modelling (`i3/user_model/`).** Three-timescale EMAs
  (instant / session / long-term) track the user's style across
  sessions. Baseline statistics are maintained with Welford's
  online algorithm. Persistence is async SQLite via `aiosqlite`;
  embeddings are Fernet-encrypted at rest.
- **Adaptation (`i3/adaptation/`).** Four adapters compute
  cognitive-load, style-mirror, emotional-tone, and accessibility
  dimensions, yielding an 8-dim `AdaptationVector`. The controller
  handles ablation (disabling individual dimensions at runtime).
- **Routing (`i3/router/`).** A contextual Thompson-sampling bandit
  decides between the local SLM and the cloud LLM. Each arm's
  posterior is a Bayesian logistic regression with Laplace
  approximation (Newton-Raphson MAP + Hessian for the covariance).
  A privacy override forces local routing for sensitive topics.
- **Local SLM (`i3/slm/`).** A ~6.3 M-parameter Pre-LN transformer
  with token + sinusoidal positional embeddings, multi-head
  self-attention with KV cache for incremental decode, and a
  dedicated cross-attention layer per block that attends to four
  conditioning tokens derived from the user-state embedding.
- **Cloud (`i3/cloud/`).** 11 first-class provider adapters behind a
  unified `MultiProviderClient` with sequential / parallel /
  best-of-N strategies, prompt translation across provider message
  formats, and a cost tracker with 2026 pricing.
- **Diary (`i3/diary/`).** Privacy-safe interaction history: only
  embeddings, scalar metrics, and TF-IDF topic keywords. Raw text is
  never stored.

---

## Novel contribution — cross-attention conditioning

Most systems personalise LLM responses via prompt engineering —
stuffing the system prompt with a user description and hoping the
model pays attention. This is brittle, ignored at long context
lengths, and wastes tokens.

I³ conditions generation architecturally, at every layer, at every
token position. A `ConditioningProjector` maps

```
concat(AdaptationVector[8], UserStateEmbedding[64])
```

into four conditioning tokens of dim 256. Every transformer block
contains a dedicated cross-attention layer that attends to them:

```python
class AdaptiveTransformerBlock(nn.Module):
    def forward(self, x, conditioning_tokens, causal_mask=None):
        # 1. Self-attention over the token sequence
        x = x + self.dropout(self.self_attn(self.ln1(x), mask=causal_mask))

        # 2. Cross-attention to the user-state conditioning tokens.
        #    This is the novel part — user state is woven into
        #    generation at every block, not stitched onto the prompt.
        x = x + self.dropout(self.cross_attn(
            query=self.ln2(x),
            key=conditioning_tokens,
            value=conditioning_tokens,
        ))

        # 3. Feed-forward
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x
```

This mechanism is why the project ships an SLM built from scratch:
cross-attention conditioning cannot be retrofitted onto a pre-trained
HuggingFace transformer without re-initialising weights.

---

## Repository layout

```
i3/                 Core Python package (36 subpackages)
  interaction/      Feature extraction + baseline tracking
  encoder/          TCN encoder + NT-Xent loss + ONNX export
  user_model/       Three-timescale EMAs + async SQLite store
  adaptation/       AdaptationVector + controller + ablation
  router/           Thompson-sampling bandit + preference learning
  slm/              Custom transformer (no HuggingFace)
  cloud/            11 provider adapters + guardrails + MultiProvider
  diary/            Privacy-preserving async SQLite diary
  privacy/          PII sanitiser + Fernet encryption + DP-SGD
  safety/           PDDL planner + safety certificates
  interpretability/ Counterfactuals, SAEs, activation patching, probes
  redteam/          55-attack adversarial corpus + runner
  mlops/            Experiment tracker + model signing
  observability/    OpenTelemetry + structlog + Prometheus + Sentry
  biometric/        Keystroke-biometric user ID + continuous auth
  continual/        Elastic Weight Consolidation + drift detection
  meta_learning/    MAML + Reptile + task generator
  multimodal/       Voice, vision, PPG/HRV, touch, fusion
  federated/        Flower client + FedAvg server
  crossdevice/      HarmonyOS Distributed Data Management sync
  fairness/         Per-archetype bias + bootstrap CI + FAR/FRR
  edge/             ONNX + ExecuTorch + 8 alternative runtimes
  eval/             Perplexity + conditioning KL + closed-loop + ablation
  serving/          Ray Serve + Triton + vLLM
  tts/              Adaptation-conditioned TTS
  huawei/           HMAF runtime + Kirin profiles + Watch integration
  authz/            Cedar policy adapter
  analytics/        DuckDB + LanceDB + Polars + Ibis
  mcp/              Anthropic Model Context Protocol server
  profiling/        Edge-feasibility profiler
  pipeline/         The async Pipeline — orchestrates everything
server/             FastAPI app — routes, middleware, WebSocket, auth
web/                Demo UI (vanilla CSS/JS, no build step)
training/           Training entry points (Fabric, Accelerate, DeepSpeed)
demo/               Pre-seeded state + scenarios + profiles
tests/              80+ test modules (unit, property, contract, fuzz, …)
scripts/            Operator tooling — see scripts/README.md
  benchmarks/       Latency and edge-profiling micro-benchmarks
  demos/            Standalone feature demos
  experiments/      Research runs (ablation, DPO, LLM-judge, …)
  export/           Model + data export (ONNX, ExecuTorch, GDPR)
  security/         Red-team, model signing, key generation
  training/         Training entry points
  verification/     46 registered checks invoked by verify_all.py
configs/            YAML configuration
docs/               MkDocs Material source tree
deploy/             Kubernetes, Helm, Terraform, policy-as-code
docker/             Dockerfile variants + entrypoint + healthcheck
reports/            Audit + verification + red-team output
  audits/           Dated narrative reports
  redteam/          55-attack harness output
  verification/     46-check harness output (+ history/)
benchmarks/         pytest-benchmark + ImplicitAdaptBench harness
notebooks/          Teaching notebooks
```

Top-level files of note:
[`Makefile`](Makefile) · [`Dockerfile`](Dockerfile) ·
[`docker-compose.yml`](docker-compose.yml) ·
[`pyproject.toml`](pyproject.toml) · [`mkdocs.yml`](mkdocs.yml) ·
[`.env.example`](.env.example) ·
[`.env.providers.example`](.env.providers.example) ·
[`CHANGELOG.md`](CHANGELOG.md) · [`SECURITY.md`](SECURITY.md) ·
[`CONTRIBUTING.md`](CONTRIBUTING.md) ·
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) ·
[`CITATION.cff`](CITATION.cff).

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| PyTorch | 2.0 | 2.3 (CPU is fine) |
| Poetry | 1.8 | latest |
| Disk space | ~2 GB for checkpoints + deps | 5 GB |
| RAM | 4 GB for inference, 8 GB for training | 16 GB |
| OS | Linux, macOS, Windows (WSL2 recommended) | |

Optional (features degrade gracefully if absent):

- **Docker 24+** — for the containerised stack.
- **`make`** — the primary task runner.
- **Node.js 20+** — only for the MCP server and some CI workflows.
- **`helm`, `kubectl`** — to deploy the Kubernetes manifests.
- **`cosign`, `syft`, `trivy`** — for supply-chain verification.

---

## Installation

> For the impatient: **`make all-fast`** orchestrates install + env + verify
> + serve with live progress bars.  See
> [Quickstart — one command](#quickstart--one-command).

### With Poetry (recommended)

```bash
git clone https://github.com/abailey81/implicit-interaction-intelligence.git
cd implicit-interaction-intelligence

# Install core + dev dependencies (the orchestrator default)
poetry install --with dev,security

# Or full fat with every optional group
poetry install --with dev,docs,security,observability,mlops,providers

# Or core only (inference without dev tools)
poetry install
```

Optional dependency groups (see [`pyproject.toml`](pyproject.toml)):
`dev`, `docs`, `security`, `observability`, `mlops`, `ml-advanced`,
`analytics`, `distributed`, `llm-ecosystem`, `providers`,
`edge-runtimes`, `multimodal`, `future-work`, `policy`, `mcp`, `tts`.

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Verifying the install

```bash
poetry run python -c "import i3; print(i3.__version__)"
poetry run make verify           # 46-check verification harness (strict)
poetry run make test             # pytest suite
```

---

## Configuration

### YAML configuration

Everything configurable lives in [`configs/default.yaml`](configs/default.yaml).
The schema is declared with Pydantic v2 in [`i3/config.py`](i3/config.py) —
every section is frozen (`frozen=True`) and the root model enforces
`extra="forbid"` so a typoed section fails loudly at load time.

```yaml
# configs/default.yaml — excerpt
project:
  name: I3
  version: 1.1.0
  seed: 42

cloud:
  model: claude-sonnet-4-5       # pinned, not a placeholder
  max_tokens: 1024
  timeout: 10.0
  fallback_on_error: true

router:
  arms: [local_slm, cloud_llm]
  context_dim: 12
  prior_alpha: 1.0               # Beta prior (cold-start)
  prior_precision: 1.0           # Gaussian prior on logistic weights
  exploration_bonus: 0.1
  privacy_override: true

privacy:
  strip_pii: true
  encrypt_embeddings: true
  encryption_key_env: I3_ENCRYPTION_KEY
```

Overlay configs (e.g. `configs/demo.yaml`) can be merged on top for
fast experimentation.

### Environment variables

Copy [`.env.example`](.env.example) to `.env` and populate the values
you need. Every variable is documented inline. Highlights:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Cloud route for the default Anthropic adapter. |
| `I3_ENCRYPTION_KEY` | Fernet key for user-state-embedding encryption at rest. Generate with `scripts/security/generate_encryption_key.py`. |
| `I3_ADMIN_TOKEN` | Bearer token for the `/admin/*` endpoints. |
| `I3_REQUIRE_USER_AUTH` | Set to `1` to gate per-user routes. |
| `I3_USER_TOKENS` | JSON map `{"user_id": "token"}` for caller-identity auth. |
| `I3_DEMO_MODE` | Enable the destructive demo endpoints. |
| `I3_DISABLE_ADMIN` | Hard-disable the admin router. |
| `I3_DISABLE_OPENAPI` | Hard-disable `/api/docs` + `/api/redoc`. |
| `I3_CORS_ORIGINS` | Comma-separated CORS allow-list. |
| `I3_MAX_TRACKED_USERS` | LRU cap on per-user state (default 10 000). |
| `I3_WORKERS` | Uvicorn worker count. Setting `> 1` requires `I3_ALLOW_LOCAL_LIMITER=1`. |

Per-provider API keys (Google, Azure, Bedrock, Mistral, Cohere,
OpenRouter, LiteLLM, Huawei PanGu, etc.) live in
[`.env.providers.example`](.env.providers.example).

---

## Training the models

Two models need training before the pipeline can produce non-trivial
output: the TCN encoder and the SLM. Both can be trained on a laptop
CPU in under an hour on the default configuration.

> **Single-command path.**  `make all-full` runs every step below in
> the right order — with live progress bars, ETAs, log tails, and
> crash-resumable ``--resume`` support.  See
> [Quickstart — one command](#quickstart--one-command).  The manual
> path below gives you the option to run individual steps.

### 1. Generate synthetic interaction data

```bash
poetry run python training/generate_synthetic.py \
    --archetypes 8 \
    --sessions-per-archetype 400 \
    --output data/interaction_dataset.jsonl
```

Eight user archetypes follow the Epp 2011 / Vizer 2009 / Zimmermann
2014 literature (fresh user, fatigued developer, motor-impaired user,
second-language speaker, high-load user, dyslexic user, energetic
user, low-vision user). Transitions between typing states use a
lightweight Markov model.

### 2. Prepare the dialogue corpus for SLM training

```bash
poetry run python training/prepare_dialogue.py \
    --daily-dialog data/daily_dialog/ \
    --empathetic   data/empathetic_dialogues/ \
    --output       data/slm_corpus.jsonl
```

The default corpora are DailyDialog and EmpatheticDialogues (public,
consented research corpora). A privacy-safe data card is emitted at
[`docs/responsible_ai/data_card.md`](docs/responsible_ai/data_card.md).

### 3. Train the TCN encoder

```bash
poetry run python training/train_encoder.py \
    --config configs/default.yaml \
    --epochs 10 \
    --output checkpoints/encoder/tcn_v1.pt
```

NT-Xent contrastive loss (Chen et al. 2020). Produces a 64-dim
user-state embedding.

### 4. Train the adaptive SLM

```bash
poetry run python training/train_slm.py \
    --config configs/default.yaml \
    --epochs 5 \
    --conditioning-encoder checkpoints/encoder/tcn_v1.pt \
    --output checkpoints/slm/slm_v1.pt
```

Cross-entropy with the cross-attention conditioning tokens derived
from the trained encoder. Cosine warmup (linear warmup + cosine
decay, implemented from scratch).

### 5. Evaluate

```bash
poetry run python training/evaluate.py \
    --config configs/default.yaml \
    --slm checkpoints/slm/slm_v1.pt \
    --encoder checkpoints/encoder/tcn_v1.pt \
    --out reports/evaluation.json
```

Metrics: sliding-window perplexity, cross-attention
conditioning-sensitivity KL divergence, and responsiveness on a
12-example tone-class golden set.

### Distributed training (optional)

```bash
# Lightning Fabric — DDP + FSDP + torch.compile max-autotune
poetry run python training/train_slm_fabric.py --config configs/default.yaml

# Accelerate
poetry run accelerate launch training/train_with_accelerate.py

# DeepSpeed ZeRO-3
poetry run deepspeed training/train_with_deepspeed.py \
    --deepspeed_config configs/distributed/ds_config_zero3.json
```

---

## Running the server

> **One command.**  `make all-fast` handles install + env + verify
> *and* launches the server.  `make all-full` does the same after
> training, evaluation, tests, security, and benchmarks.  Use the raw
> commands below when you want to launch the server without running
> the other stages.

### Development

```bash
poetry run uvicorn server.app:create_app --factory --reload
# Serves at http://127.0.0.1:8000
```

Or the Makefile shortcut:

```bash
make serve-dev
```

### Production

```bash
poetry run uvicorn server.app:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1 \
    --log-config configs/logging.yaml
```

> **Note on workers.** The in-memory rate limiter is per-process.
> Running `--workers > 1` multiplies the effective per-IP rate unless
> a shared store is configured. The app refuses to start with
> `I3_WORKERS > 1` unless `I3_ALLOW_LOCAL_LIMITER=1` acknowledges
> this trade-off.

### Endpoints

- **REST** — mounted under `/api/*`, documented at `/api/docs`.
- **WebSocket** — `/ws/{user_id}` streams keystroke events and
  receives adaptation updates.
- **Admin** — `/admin/*` (requires `I3_ADMIN_TOKEN`; disable in
  production with `I3_DISABLE_ADMIN=1`).
- **Health** — `/api/health`, `/api/live`, `/api/ready`.
- **Metrics** — `/api/metrics` (gated by `I3_METRICS_ENABLED`).
- **Demo UI** — `/` (basic) and `/advanced` (cinematic command centre).

---

## Using the API

### REST — message processing

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
    -H 'Content-Type: application/json' \
    -H 'X-I3-User-Id: alice' \
    -d '{
        "user_id": "alice",
        "text": "Explain how TCNs work.",
        "target_lang": "en",
        "keystroke_intervals_ms": [120, 145, 98, 210, 88]
    }'
```

### REST — adaptation explanation

```bash
curl -X POST http://127.0.0.1:8000/api/explain/adaptation \
    -H 'Content-Type: application/json' \
    -H 'X-I3-User-Id: alice' \
    -d '{
        "user_id": "alice",
        "message": "I am tired and need a break",
        "mc_dropout_samples": 30
    }'
```

Returns per-dimension confidence, counterfactuals, and a natural-language
explanation of the adaptation choice.

### WebSocket

```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/ws/alice");

ws.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    // frame.type: "state_update" | "response" | "error"
    console.log(frame);
};

// Stream keystroke events at ~20 Hz
ws.send(JSON.stringify({
    type: "keystroke",
    timestamp: Date.now() / 1000,
    key_type: "char",
    inter_key_interval_ms: 145,
}));

// Submit a message
ws.send(JSON.stringify({
    type: "message",
    text: "Can you explain how TCNs work?",
}));
```

### Python SDK

```python
import asyncio
from i3.config import load_config
from i3.pipeline.engine import Pipeline
from i3.pipeline.types import PipelineInput

async def main() -> None:
    config = load_config("configs/default.yaml")
    pipeline = Pipeline(config)
    await pipeline.initialize()
    try:
        output = await pipeline.process_message(PipelineInput(
            user_id="alice",
            message="Can you explain how TCNs work?",
            keystroke_intervals_ms=[120, 145, 98, 210, 88],
            timestamp=1712534400.0,
        ))
        print(output.response_text)
        print(output.route_chosen)        # "local_slm" or "cloud_llm"
        print(output.adaptation)          # 8-dim AdaptationVector
        print(output.latency_ms)
    finally:
        await pipeline.shutdown()

asyncio.run(main())
```

The full API reference is at
[`docs/api/`](docs/api/) (REST, WebSocket, Python SDK).

---

## The demo UI

Two user interfaces ship with the project.

- **Basic UI** at `/` — a single-page app in vanilla HTML/CSS/JS with
  a chat area, route/latency badges, and animated adaptation gauges.
- **Advanced UI** at `/advanced` — a cinematic command-centre layout:
  a seven-panel CSS-Grid with a Three.js 3-D embedding cloud, Chart.js
  metric graphs, SVG radial adaptation gauges with uncertainty bands,
  a 4 × 4 cross-attention heatmap, an `Alt + T` guided tour that
  walks the four demo phases autonomously, and a runtime WCAG 2.2 AA
  contrast audit.

To run:

```bash
make run
# Open http://127.0.0.1:8000/
# Or:   http://127.0.0.1:8000/advanced
```

Keystroke capture is browser-local — nothing is transmitted to the
server beyond the inter-key-interval list and the submitted message
text.

---

## Docker and Docker Compose

### Building

```bash
# Production image (multi-stage, non-root, tini PID 1)
make docker-build
# or
docker build -t i3:latest -f Dockerfile .

# Development image (hot reload)
docker build -t i3:dev -f docker/Dockerfile.dev .

# Chainguard Wolfi distroless variant (zero H/C CVEs on base)
docker build -t i3:wolfi -f docker/Dockerfile.wolfi .

# Minimal MCP server image
docker build -t i3:mcp -f docker/Dockerfile.mcp .
```

### Running with Compose

```bash
# Development stack
docker compose up

# Production stack — read-only root FS, cap_drop ALL, no-new-privileges,
# nginx TLS sidecar, tmpfs for mutable paths
docker compose -f docker-compose.prod.yml up -d
```

### Pulling signed images

Release images are signed with cosign (keyless OIDC) and carry an
SLSA L3 provenance attestation. To verify:

```bash
cosign verify ghcr.io/abailey81/i3:v1.1.0 \
    --certificate-identity-regexp='https://github\.com/abailey81/.*' \
    --certificate-oidc-issuer=https://token.actions.githubusercontent.com

slsa-verifier verify-artifact \
    --provenance-path i3.intoto.jsonl \
    --source-uri github.com/abailey81/implicit-interaction-intelligence \
    --source-tag v1.1.0 \
    i3.tar
```

See [`docs/security/slsa.md`](docs/security/slsa.md).

---

## Kubernetes and Helm

```bash
# Raw manifests via Kustomize overlays
kubectl apply -k deploy/k8s/overlays/dev
kubectl apply -k deploy/k8s/overlays/prod

# Helm
helm install i3 deploy/helm/i3 -f deploy/helm/i3/values-prod.yaml

# Observability stack (OTel + Prometheus + Tempo + Grafana)
docker compose -f deploy/observability/docker-compose.yml up -d
```

Manifests enforce RuntimeDefault seccomp, read-only root FS,
`capabilities: drop: [ALL]`, `automountServiceAccountToken: false`,
a `NetworkPolicy` with default-deny + narrow allow, and a
`ServiceMonitor` for Prometheus Operator. HPA v2 targets 70 % CPU and
a custom `http_requests_per_second` metric. Runbook at
[`docs/operations/runbook.md`](docs/operations/runbook.md).

An AWS EKS Terraform reference module is available under
[`deploy/terraform/`](deploy/terraform/).

---

## Testing and verification

The project has three independent verification layers.

### Layer 1 — pytest

```bash
make test              # Full pytest suite (80+ modules)
make test-cov          # With coverage report
make test-fast         # Skip slow integration tests
pytest tests/property/ # Hypothesis property-based tests
pytest tests/contract/ # schemathesis OpenAPI contract tests
pytest tests/fuzz/     # Atheris fuzz targets
pytest tests/load/     # Locust soak scenarios
pytest tests/mutation/ # mutmut
pytest tests/chaos/    # Chaos-engineering scenarios
```

### Layer 2 — the 46-check verification harness

```bash
make verify            # Strict mode: fail on blocker or high severity
make verify-quick      # Skip slow runtime checks
```

Seven categories: code integrity, configuration, runtime, providers,
infrastructure, interview-readiness, security. Output under
[`reports/verification/latest.{json,md}`](reports/verification/).

### Layer 3 — the 55-attack red-team harness

```bash
make redteam
# or the torch-DLL-safe Windows wrapper
python scripts/security/run_redteam_notorch.py --targets sanitizer,pddl,guardrails
```

Four runtime invariants are evaluated: privacy, rate-limit,
sensitive-topic, PDDL soundness. Results at
[`reports/redteam/latest.{json,md}`](reports/redteam/).

### Benchmarks

```bash
pytest benchmarks/ --benchmark-only
# Or use the ImplicitAdaptBench harness:
python scripts/experiments/implicit_adapt_bench.py
```

---

## Observability

The observability stack is soft-imported — every module is a no-op
when its dependency is absent, so the core pipeline boots unchanged
in stripped environments.

| Signal | Tool | Entry point |
|---|---|---|
| Traces | OpenTelemetry (OTLP gRPC, batched) | `i3/observability/tracing.py` |
| Metrics | Prometheus (`/api/metrics`) | `i3/observability/metrics.py` |
| Logs | structlog JSON with PII redaction | `i3/observability/logging.py` |
| Errors | Sentry (PII-scrubbing `before_send`) | `i3/observability/sentry_integration.py` |
| LLM tracing | Langfuse with token + cost attribution | `i3/observability/langfuse_client.py` |
| Profiling | Grafana Pyroscope (opt-in) | `i3/observability/pyroscope_integration.py` |

To bring up the local stack:

```bash
docker compose -f deploy/observability/docker-compose.yml up -d
# Grafana at http://localhost:3000 (ten-panel I³ overview dashboard
# is provisioned automatically)
```

---

## Edge deployment

After INT8 dynamic quantisation:

| Model          | Parameters | FP32    | INT8   | P50 latency |
|:---------------|-----------:|--------:|-------:|------------:|
| TCN encoder    |     ~50 K  | ~200 KB | ~60 KB |      ~3 ms  |
| Adaptive SLM   |    ~6.3 M  |  ~25 MB |  ~7 MB |    ~150 ms  |

Device feasibility at the 50-%-of-available-memory threshold:

| Device                 | Memory | TOPS | Fits | Notes                    |
|:-----------------------|-------:|-----:|:----:|:-------------------------|
| Kirin 9000 (phone)     | 512 MB |  2.0 |  ✓   | Comfortable headroom     |
| Kirin A2 (wearable)    | 128 MB |  0.5 |  ✓   | Tight but feasible       |
| Smart Hanhan (IoT)     |  64 MB |  0.1 |  ~   | Encoder-only recommended |

**Export paths**

```bash
# ONNX with parity verification (atol=1e-4)
python scripts/export/onnx.py --output web/models/

# ExecuTorch
python scripts/export/executorch.py --output checkpoints/slm_v1.pte

# Every alternative runtime in one go
python scripts/export/all_runtimes.py
```

Alternative runtimes supported: Apple MLX, llama.cpp GGUF, Apache
TVM, IREE, Core ML, TensorRT-LLM, OpenVINO, MediaPipe. See
[`docs/edge/alternative_runtimes.md`](docs/edge/alternative_runtimes.md)
for the 8-runtime decision matrix.

**In-browser inference**

The demo UI can run the encoder entirely in the browser via ONNX
Runtime Web + WebGPU — keystroke packets never leave the device.
Toggle in the advanced UI settings panel.

---

## Privacy and security

Privacy-preserving by construction, not by policy.

- **Raw text is never persisted.** The interaction diary stores only
  64-dim embeddings, scalar metrics, and TF-IDF topic keywords.
- **Fernet-encrypted user-state embeddings at rest**
  ([`i3/privacy/encryption.py`](i3/privacy/encryption.py)); supports
  `MultiFernet` key rotation.
- **Ten PII regex patterns** sanitise every outbound cloud payload
  (email, phone, SSN, credit card, IBAN, address, IP, URL, DOB,
  passport) — [`i3/privacy/sanitizer.py`](i3/privacy/sanitizer.py).
- **Privacy override** — sensitive topics (health, finance,
  credentials, security) force local-SLM routing regardless of the
  Thompson sample.
- **PDDL-grounded safety planner** emits a machine-checkable
  `SafetyCertificate` for every cloud-routed request —
  [`i3/safety/pddl_planner.py`](i3/safety/pddl_planner.py).
- **Differential privacy** — Opacus DP-SGD wrapper for the router
  posterior is available under
  [`i3/privacy/differential_privacy.py`](i3/privacy/differential_privacy.py).
- **Caller-identity auth** — `server/auth.py` provides opt-in
  per-user authentication via `I3_REQUIRE_USER_AUTH=1` + either an
  `X-I3-User-Id` header match or a JSON token map in
  `I3_USER_TOKENS`; uses `secrets.compare_digest` throughout.

Threat model, disclosure process, and hardening guide:
[`SECURITY.md`](SECURITY.md). SLSA Level 3 build provenance:
[`docs/security/slsa.md`](docs/security/slsa.md). Supply-chain
posture: [`docs/security/supply-chain.md`](docs/security/supply-chain.md).
Policy-as-code (Kyverno, OPA, Cedar, Falco, Tracee):
[`docs/security/policy_as_code.md`](docs/security/policy_as_code.md).

---

## Command reference (`make`)

```
End-to-end orchestrator  (one command, live progress bars + ETAs)
  make all                Alias for all-fast
  make all-fast           Fast path: install → env → verify → serve (~5 min)
  make all-full           Full path: data → train → eval → tests → security
                          → verify → bench → onnx → docs → serve (hours/~15 min GPU)
  make all-list           Print the stage graph for both modes
  make all-resume         Re-run, skipping stages whose outputs already exist

Installation + setup
  make install            Install core + dev dependencies
  make install-prod       Install without dev dependencies
  make install-all        Install with dev + security + docs groups
  make setup              Alias for install
  make clean              Remove build artefacts + caches
  make clean-data         Remove generated synthetic data
  make clean-checkpoints  Remove model checkpoints (confirms first)
  make clean-all          clean + clean-data

Code quality
  make lint               Ruff lint
  make format             Ruff format + autofix
  make typecheck          Mypy type check
  make check              lint + typecheck + test  (pre-commit gate)
  make security-check     Bandit + pip-audit
  make audit              Alias for security-check

Testing + verification
  make test               Full pytest suite
  make test-cov           With coverage
  make test-fast          Skip slow tests
  make test-security      Security-marked tests only
  make test-parallel      pytest-xdist parallel
  make verify             44-check verification harness
  make verify-strict      Strict mode (any FAIL fails the build)
  make verify-quick       Only code + config + interview categories
  make redteam            55-attack red-team harness
  make benchmarks         pytest-benchmark micro-benchmarks
  make bench-ci           Emit CI-friendly JSON

Training + data
  make generate-data      Synthetic interaction corpus
  make prepare-dialogue   Clean + dedup + split sample dialogue corpus
  make prepare-data       Alias for prepare-dialogue
  make train-encoder      Train the TCN encoder
  make train-slm          Train the adaptive SLM
  make train-all          Generate data + train encoder + train SLM
  make evaluate           Perplexity + latency report
  make eval-conditioning  Cross-attention KL sensitivity

Server + demo
  make serve              Production server (0.0.0.0:8000)
  make serve-dev          Dev server with hot reload
  make seed-demo          Pre-seed the demo database
  make demo               seed-demo + serve-dev (opens browser)

Export + edge
  make export-onnx        Export TCN + SLM to ONNX
  make verify-onnx        ONNX ↔ PyTorch parity check
  make profile-edge       Edge-feasibility profiling report
  make sign-model         Sign a checkpoint with Sigstore

Docker
  make docker-build       Production image (multi-stage, non-root, tini PID 1)
  make docker-build-dev   Dev image (Dockerfile.dev, hot reload)
  make docker-up          docker compose up (base profile)
  make docker-up-prod     Hardened profile (read-only rootfs, cap_drop, nginx)
  make docker-down        Tear down

Documentation
  make docs               Alias for docs-serve
  make docs-serve         MkDocs at http://127.0.0.1:8001 with hot reload
  make docs-build         Build static site into ./site
  make docs-strict        Build with --strict (fails on warnings)
  make docs-deploy        Deploy to gh-pages

Observability
  make obs-up             Grafana :3000, Prometheus :9090, Tempo, Loki
  make obs-down           Tear down

Build + release
  make build              Build wheel + sdist
  make publish            Publish to PyPI
  make release-check      All quality + security gates
```

Full target listing: `make help`.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branching model,
commit-message convention, code-review expectations, and the local
test loop. Security-sensitive reports should follow the process in
[`SECURITY.md`](SECURITY.md). The code of conduct is the Contributor
Covenant ([`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)).

---

## Further reading

- **Architecture full reference** —
  [`docs/architecture/full-reference.md`](docs/architecture/full-reference.md)
- **Cross-attention conditioning** —
  [`docs/architecture/cross-attention-conditioning.md`](docs/architecture/cross-attention-conditioning.md)
- **Router and bandit theory** —
  [`docs/research/bandit_theory.md`](docs/research/bandit_theory.md)
- **Privacy architecture** —
  [`docs/architecture/privacy.md`](docs/architecture/privacy.md)
- **Edge profiling** —
  [`docs/edge/profiling-report.md`](docs/edge/profiling-report.md)
- **Research paper** —
  [`docs/paper/I3_research_paper.md`](docs/paper/I3_research_paper.md)
- **ADRs** — ten Architecture Decision Records under
  [`docs/adr/`](docs/adr/).
- **Runbook** — [`docs/operations/runbook.md`](docs/operations/runbook.md)

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Draws on Eric Xu's L1–L5 device-intelligence framework, Edinburgh
Joint Lab research on personalisation from sparse signals, and the
HarmonyOS Multi-Agent Framework (HMAF).
