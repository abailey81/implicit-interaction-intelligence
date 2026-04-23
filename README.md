<div align="center">

# Implicit Interaction Intelligence (I³)

### Adaptive AI companion systems that learn from *how* you interact

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-80+-success.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*An AI companion that builds a rich, evolving model of each user from implicit
behavioural signals — keystroke dynamics, linguistic complexity, temporal
patterns — and continuously adapts its responses across cognitive load,
communication style, emotional tone, and accessibility needs.*

</div>

---

## The Idea

Current conversational AI systems respond to **what** you say. They parse your
tokens, compute their attention, and return a response. But humans — especially
the people who know us well — notice something richer: they notice **how** we
say things. A close friend knows when you're tired from the pace of your reply,
when you're stressed from the rhythm of your typing, when you're struggling
from the effort you're putting into a single sentence. They don't need to ask.
They adapt.

**Implicit Interaction Intelligence (I³)** is an attempt to build that
capability into an AI companion. It observes keystroke dynamics, typing
patterns, and linguistic complexity to build a continuous, evolving model of
each user's cognitive state, communication style, and accessibility needs.
That model then conditions response generation at every level — from the
routing decision (local SLM vs. cloud LLM) down to the token-by-token
cross-attention inside a custom transformer.

The project demonstrates three tiers of AI capability working together: a
**custom ML encoder** (a TCN built from scratch), a **custom small language
model** (a ~6.3 M-parameter transformer with no HuggingFace dependency), and
**intelligent routing to foundational models** (a Bayesian bandit that decides
when cloud capacity is worth the privacy and latency trade-off). Every
non-trivial component — the encoder, the transformer, the cross-attention
conditioning, the Thompson sampling bandit, the sentiment lexicon, even the
cosine warmup scheduler — is implemented from first principles.

## Architecture

```
╭──────────────────────────────────────────────────────────────────────────╮
│                                                                          │
│    USER KEYSTROKE                                                        │
│         │                                                                │
│         ▼                                                                │
│    ┌─────────────────────┐                                               │
│    │  Layer 1            │   32-dim InteractionFeatureVector             │
│    │  Perception         │   (keystroke dynamics, linguistic, session)   │
│    │  i3/interaction/    │                                               │
│    └──────────┬──────────┘                                               │
│               │                                                          │
│               ▼                                                          │
│    ┌─────────────────────┐                                               │
│    │  Layer 2            │   64-dim user state embedding                 │
│    │  TCN Encoder        │   (dilations [1,2,4,8], NT-Xent contrastive)  │
│    │  i3/encoder/        │                                               │
│    └──────────┬──────────┘                                               │
│               │                                                          │
│               ▼                                                          │
│    ┌─────────────────────┐                                               │
│    │  Layer 3            │   Instant / Session / Long-term EMAs          │
│    │  User Model         │   (Welford's online algorithm)                │
│    │  i3/user_model/     │                                               │
│    └──────────┬──────────┘                                               │
│               │                                                          │
│               ▼                                                          │
│    ┌─────────────────────┐                                               │
│    │  Layer 4            │   8-dim AdaptationVector                      │
│    │  Adaptation         │   (cognitive load, style, tone, a11y)         │
│    │  i3/adaptation/     │                                               │
│    └──────────┬──────────┘                                               │
│               │                                                          │
│               ▼                                                          │
│    ┌─────────────────────┐                                               │
│    │  Layer 5            │   Contextual Thompson sampling                │
│    │  Router             │   (Bayesian logistic regression + Laplace)    │
│    │  i3/router/         │                                               │
│    └──────────┬──────────┘                                               │
│               │                                                          │
│               ├────────────────────────┐                                 │
│               ▼                        ▼                                 │
│    ┌─────────────────────┐   ┌─────────────────────┐                     │
│    │  Layer 6a           │   │  Layer 6b           │                     │
│    │  Adaptive SLM       │   │  Cloud LLM          │                     │
│    │  ~6.3M params       │   │  Claude API         │                     │
│    │  i3/slm/            │   │  i3/cloud/          │                     │
│    └──────────┬──────────┘   └──────────┬──────────┘                     │
│               │                         │                                │
│               └────────────┬────────────┘                                │
│                            │                                             │
│                            ▼                                             │
│                 ┌─────────────────────┐                                  │
│                 │  Layer 7            │   Embeddings + topics only       │
│                 │  Interaction Diary  │   (raw text never stored)        │
│                 │  i3/diary/          │                                  │
│                 └──────────┬──────────┘                                  │
│                            │                                             │
│                            ▼                                             │
│                     ADAPTED RESPONSE                                     │
│                                                                          │
│    ╷                                                  ╷                  │
│    ├─ Cross-cutting: i3/privacy/  (PII, Fernet, audit) ┤                 │
│    └─ Cross-cutting: i3/profiling/ (latency, memory)   ┘                 │
│                                                                          │
╰──────────────────────────────────────────────────────────────────────────╯
```

The system is structured as **seven sequential layers** plus **two
cross-cutting concerns** (privacy, profiling), all orchestrated by a central
async pipeline in `i3/pipeline/engine.py`.

### Layer 1 — Perception (`i3/interaction/`)

Captures **how** the user interacts, not just what they say. Extracts a
32-dimensional `InteractionFeatureVector` per message covering four feature
groups of eight features each: keystroke dynamics (inter-key intervals,
burst/pause ratios, correction rate), message content (length, average word
length, vocabulary diversity), linguistic complexity (Flesch-Kincaid grade,
formality, sentiment valence), and session dynamics (within-session deviation
from the user's personal baseline).

The linguistic module contains a **~365-word valence lexicon** with negation
handling, a **syllable counter** for Flesch-Kincaid grade-level estimation,
and a formality classifier — all implemented from scratch with zero external
NLP dependencies.

### Layer 2 — Encoding (`i3/encoder/`)

A **Temporal Convolutional Network** built from scratch in PyTorch. Four
dilated causal convolution blocks with dilations `[1, 2, 4, 8]` and kernel
size 3 produce a receptive field of ~61 timesteps — enough to see a full
conversational window. Each block is a residual `CausalConv1d → LayerNorm →
GELU → Dropout → CausalConv1d → LayerNorm` pair with a 1×1 skip projection.

The network is trained with **NT-Xent contrastive loss** (the same objective
used in SimCLR) on synthetic interaction data generated from eight user
states with Markov transitions. The output is a compact 64-dimensional
embedding used both as routing context and as conditioning for the SLM.

### Layer 3 — User Modelling (`i3/user_model/`)

A persistent three-timescale representation:

```
  t=0 ──────────── t=∞
   │                │
   ├── instant ─────┤   current encoder output (stateless)
   │                │
   ├── session ─────┤   EMA within session,   α = 0.3
   │                │
   ├── long-term ───┤   EMA across sessions,  α = 0.1
```

Uses **Welford's online algorithm** for numerically stable incremental
feature-statistic updates (mean and variance without re-reading history),
with deviation metrics computed as z-scores against the long-term profile.
Persistent profiles are stored asynchronously in SQLite via `aiosqlite`,
Fernet-encrypted at rest.

### Layer 4 — Adaptation (`i3/adaptation/`)

Maps user state to an 8-dimensional `AdaptationVector` via four independent
adapters orchestrated by an `AdaptationController`:

| Dimension         | Adapter                    | Output                               |
|-------------------|----------------------------|--------------------------------------|
| `cognitive_load`  | `CognitiveLoadAdapter`     | Scalar ∈ [0, 1] — target complexity  |
| `style_mirror`    | `StyleMirrorAdapter`       | 4-dim `StyleVector`                  |
| `emotional_tone`  | `EmotionalToneAdapter`     | Scalar ∈ [0, 1] — warmth level       |
| `accessibility`   | `AccessibilityAdapter`     | Scalar ∈ [0, 1] — simplification     |

The `StyleVector` has four dimensions: `formality`, `verbosity`,
`emotionality`, `directness` — each mirrored from the user's long-term
baseline. The `AccessibilityAdapter` detects motor or cognitive difficulty
from typing metrics (elevated correction rate, slow inter-key intervals,
short bursts) and elevates simplification level.

### Layer 5 — Routing (`i3/router/`)

A **Contextual Thompson Sampling** multi-armed bandit built from scratch:

- Two arms — local SLM and cloud LLM
- 12-dimensional context vector: query complexity, topic sensitivity flags,
  user-state summary, session progress, device pressure
- **Bayesian logistic regression** posteriors per arm (weights `w ~ N(μ, Σ)`)
- **Laplace approximation** refitted every 10 updates via Newton-Raphson MAP
- Beta-Bernoulli cold-start fallback for the first few interactions
- **Privacy override** — sensitive topics (health, mental health, financial
  credentials, security credentials) are force-routed to the local SLM
  regardless of posterior sample

Thompson sampling is used rather than UCB because it naturally handles the
non-stationary reward landscape (user preferences drift over time) and
integrates cleanly with the Bayesian posterior.

### Layer 6a — Local SLM (`i3/slm/`)

A **~6.3M-parameter transformer** built from scratch in PyTorch. **No
HuggingFace. No pre-trained weights. No `transformers` import anywhere.**
Every component implemented from first principles:

- Word-level tokenizer with 8 192-entry vocabulary and special tokens
- Token embeddings + **sinusoidal positional encoding**
- **Multi-head self-attention** with KV caching for fast autoregressive decode
- **Novel cross-attention conditioning** (the architectural centrepiece —
  see below)
- Pre-LN `AdaptiveTransformerBlock` × 4 (`d_model=256`, `n_heads=4`)
- **Weight-tied** output projection for parameter efficiency
- Top-k / top-p / repetition-penalty sampling
- **INT8 dynamic quantization** for edge deployment
- Cosine-warmup learning-rate scheduler written from scratch

### Layer 6b — Cloud LLM (`i3/cloud/`)

Async Anthropic Claude client built directly on `httpx`. The `prompt_builder`
constructs dynamic system prompts from the `AdaptationVector`, translating the
adaptation parameters into explicit style instructions (formality level, target
verbosity, warmth, simplification depth). All requests are PII-sanitized before
transmission, retries use exponential backoff, and token usage is tracked
per-user for cost accounting.

### Layer 7 — Diary (`i3/diary/`)

Privacy-safe interaction diary. **Raw user text is never stored.** Per
exchange, the diary persists only:

- The 64-dim user-state embedding at the time of the exchange
- Scalar metrics (cognitive load, latency, route taken, reward)
- **TF-IDF topic keywords** (extracted on the fly with a ~175-word stopword
  list — no NLTK, no sklearn)
- The `AdaptationVector` applied to the response

Session summaries are generated from **aggregated metadata** using either a
template or a cloud call — but the cloud summariser only ever receives
metadata, never raw text.

### Cross-Cutting: Privacy (`i3/privacy/`)

- **10 compiled PII regex patterns** — email, phone, SSN, credit card,
  IBAN, street address, IP address, URL, DOB, passport
- **Fernet symmetric encryption** for user models at rest
- **Privacy auditor** that scans the database for potential raw-text leaks
  (hash-based detection of message fragments)

### Cross-Cutting: Profiling (`i3/profiling/`)

Edge deployment feasibility measurement:

- Parameter counting, FP32 vs INT8 on-disk size comparison
- Latency benchmarks (P50 / P95 / P99 percentiles) with warmup iterations
- Memory footprint via Python's `tracemalloc`
- Device feasibility matrix against Kirin 9000 (phone), Kirin A2 (wearable),
  and Smart Hanhan (IoT)

---

## Project Structure

The full tree with one-line descriptions is in
[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md). The high-level map:

```
implicit-interaction-intelligence/
├── i3/                   # Main Python package (36 subpackages — see PROJECT_STRUCTURE.md)
│   ├── interaction/      # Layer 1 — behavioural signal extraction (32-dim FV)
│   ├── encoder/          # Layer 2 — custom TCN (dilations [1,2,4,8] + NT-Xent)
│   ├── user_model/       # Layer 3 — three-timescale EMAs + SQLite store
│   ├── adaptation/       # Layer 4 — 8-dim AdaptationVector + controller
│   ├── router/           # Layer 5 — Thompson sampling bandit + preference learning
│   ├── slm/              # Layer 6a — ~6.3 M-param custom transformer (from scratch)
│   ├── cloud/            # Layer 6b — 11 provider adapters + guardrails + MultiProvider
│   ├── diary/            # Layer 7 — privacy-safe async SQLite diary
│   ├── privacy/          # Cross-cutting — sanitiser + Fernet + DP-SGD
│   ├── safety/           # Cross-cutting — PDDL planner + certificates
│   ├── interpretability/ # Counterfactuals + SAE + activation patching
│   ├── redteam/          # 55-attack adversarial corpus + runner
│   ├── mlops/            # Experiment tracker + registry + model signing
│   ├── observability/    # OTel + structlog + Prometheus + Sentry + Langfuse
│   └── pipeline/         # Orchestration — the async Pipeline
├── server/               # FastAPI app — routes, middleware, WebSocket, auth
├── web/                  # Demo UI (plain CSS/JS, zero build step)
├── training/             # Training entry points (Fabric, Accelerate, DeepSpeed)
├── demo/                 # Pre-seeded state + scenarios + profiles
├── tests/                # 80+ test modules (unit, property, contract, fuzz, …)
├── scripts/              # Operator tooling — see scripts/README.md
├── configs/              # YAML configuration
├── docs/                 # MkDocs Material source tree
├── deploy/               # Kubernetes, Helm, Terraform, policy-as-code
├── docker/               # Dockerfile variants + entrypoint + healthcheck
├── notebooks/            # Teaching notebooks
├── benchmarks/           # pytest-benchmark + ImplicitAdaptBench
├── reports/              # Audit + verification + red-team output
├── dagger/               # Dagger Python SDK CI pipeline
├── backstage/            # Spotify Backstage service catalog
├── Makefile              # Primary task runner
├── Dockerfile            # Production image
├── pyproject.toml        # Poetry + tool configuration
├── mkdocs.yml            # Documentation site configuration
├── CHANGELOG.md          # Keep-a-Changelog history
├── CONTRIBUTING.md       # Contributor guide
├── SECURITY.md           # Threat model + disclosure process
└── LICENSE               # MIT
```

---

## Quick Start

### Prerequisites

- **Python 3.10** or higher
- **PyTorch 2.0+** (CPU is fine for the demo)
- **`make`** (optional but recommended)
- **An Anthropic API key** (only required if you want to exercise the cloud
  routing path — the local SLM works without one)

### Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd implicit-interaction-intelligence

# Option A — one-shot setup script
./scripts/setup.sh

# Option B — manual setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                      # then edit with your API key
python scripts/security/generate_encryption_key.py  # generates a Fernet key
```

### Training

```bash
# Generate 10,000 synthetic interaction sessions
make generate-data

# Train the TCN encoder          (~30 min on laptop CPU)
make train-encoder

# Train the SLM                  (~2 hours on laptop CPU)
make train-slm

# Evaluate perplexity and conditioning sensitivity
make evaluate

# Or run the full training pipeline in one shot
make train-all
```

### Running the Demo

```bash
# Seed profiles, start the FastAPI server, open your browser
make demo

# Then open http://localhost:8000
```

---

## Usage Example

```python
import asyncio
from i3.config import load_config
from i3.pipeline.engine import PipelineEngine
from i3.pipeline.types import PipelineInput

async def main():
    config = load_config("configs/default.yaml")
    engine = await PipelineEngine.create(config)

    # Submit an interaction event (raw text + keystroke timing)
    output = await engine.process(PipelineInput(
        user_id="alice",
        message="Can you explain how TCNs work?",
        keystroke_intervals_ms=[120, 145, 98, 210, 88, ...],
        timestamp=1712534400.0,
    ))

    print(output.response)          # the adapted response
    print(output.route)              # "local" or "cloud"
    print(output.adaptation)         # 8-dim AdaptationVector
    print(output.latency_ms)         # wall-clock latency

asyncio.run(main())
```

---

## The Novel Contribution: Cross-Attention Conditioning

Most systems that personalise LLM responses do so via **prompt engineering** —
stuff a description of the user into the system prompt and hope the model
pays attention to it. This is brittle, ignored at long context lengths, and
wastes prompt tokens.

I³ takes a different approach: **the user state conditions generation
architecturally, at every layer, at every token position.**

A `ConditioningProjector` maps the concatenation of the `AdaptationVector`
(8-dim) and the `UserStateEmbedding` (64-dim) into **4 conditioning tokens**
of dimension 256. These conditioning tokens are passed to every transformer
block as the key-value pairs for a **dedicated cross-attention layer** that
sits between self-attention and feed-forward:

```python
class AdaptiveTransformerBlock(nn.Module):
    def forward(self, x, conditioning_tokens, causal_mask=None):
        # Pre-LN Self-Attention
        x = x + self.dropout(self.self_attn(self.ln1(x), mask=causal_mask))

        # Pre-LN Cross-Attention to user conditioning  ← the novel part
        x = x + self.dropout(self.cross_attn(
            query=self.ln2(x),
            key=conditioning_tokens,
            value=conditioning_tokens,
        ))

        # Pre-LN Feed-Forward
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x
```

The result: at every token position, during every forward pass, every
transformer block gets to **attend to the user's current state**. Adaptation
is not a prompt prefix — it is structurally woven into generation.

This mechanism is the architectural centrepiece of the project and is what
justifies building an SLM from scratch: you cannot retrofit cross-attention
conditioning into a pre-trained HuggingFace model without re-training from a
random init.

---

## Edge Feasibility

The entire system is designed for on-device deployment. After INT8
quantization:

| Model          | Parameters | FP32 Size | INT8 Size | P50 Latency |
|:---------------|-----------:|----------:|----------:|------------:|
| TCN Encoder    |      ~50K  |   ~200 KB |    ~60 KB |       ~3 ms |
| Adaptive SLM   |     ~6.3M  |    ~25 MB |     ~7 MB |     ~150 ms |
| **Total**      |  **~6.4M** | **~25 MB**| **~7 MB** | **~153 ms** |

Device feasibility (50%-of-available-memory rule):

| Target Device        | Memory | TOPS | Fits? | Notes                      |
|:---------------------|-------:|-----:|:-----:|:---------------------------|
| Kirin 9000 (phone)   | 512 MB | 2.0  |   ✓   | Comfortable headroom       |
| Kirin A2 (wearable)  | 128 MB | 0.5  |   ✓   | Tight but feasible         |
| Smart Hanhan (IoT)   |  64 MB | 0.1  |   ~   | Encoder-only recommended   |

The profiling package (`i3/profiling/`) produces these numbers automatically
and writes a Markdown device feasibility report.

---

## Privacy by Architecture

I³ is **privacy-preserving by construction**, not by policy.

- **Raw text is never persisted to disk.** The interaction diary stores only
  64-dim embeddings, scalar metrics, and TF-IDF topic keywords.
- **Fernet-encrypted user profiles at rest**, with key management via
  environment variables and support for key rotation.
- **10 PII regex patterns** sanitize all text before any cloud transmission
  (email, phone, SSN, credit card, IBAN, address, IP, URL, DOB, passport).
- **Router privacy override** — sensitive topics (health, mental health,
  finance, credentials, security) are force-routed to the local SLM
  regardless of what the Thompson sample recommends.
- **Session summaries generated from metadata only** — no raw text is ever
  sent to the cloud LLM for summarisation.
- **Database auditor** scans the diary SQLite file for potential raw-text
  leaks using hash-based detection of message fragments.

---

## Testing

```bash
make test         # Full test suite (80+ tests, ~30 s)
make test-cov     # With coverage report
make test-fast    # Skip slow integration tests
```

Test coverage includes:

| Test module          | Tests | Coverage                                      |
|:---------------------|------:|:----------------------------------------------|
| `test_tcn.py`        |    12 | Dilation math, receptive field, gradient flow |
| `test_slm.py`        |    15 | Forward pass, KV cache, generation, quant     |
| `test_bandit.py`     |    18 | Posterior updates, Laplace, privacy override  |
| `test_user_model.py` |    17 | EMA decay, Welford, encryption round-trips    |
| `test_pipeline.py`   |    18 | End-to-end async, error paths, latency        |

---

## Configuration

All hyperparameters, paths, and behavioural settings live in
`configs/default.yaml`. The config system uses **Pydantic v2 models** with
immutable `frozen=True` settings and field validators. Overlay configs
(e.g. `configs/demo.yaml`) can be merged on top of the default for fast
experimentation without mutating the production config.

---

## What I Built From Scratch

*No HuggingFace. No LangChain. No pre-trained weights. No off-the-shelf
bandits. No sklearn. No NLTK.*

| Component                      | Implementation                                                               |
|:-------------------------------|:-----------------------------------------------------------------------------|
| TCN Encoder                    | Causal Conv1d, dilated residual blocks, NT-Xent contrastive loss             |
| SLM                            | Token + sinusoidal embeddings, multi-head self-/cross-attention, Pre-LN      |
| Cross-attention conditioning   | `ConditioningProjector` + per-layer cross-attention (novel)                  |
| KV cache                       | Per-layer tensor cache for O(1) incremental decode                           |
| Thompson sampling bandit       | Bayesian logistic regression + Laplace approximation + Newton-Raphson MAP    |
| Tokenizer                      | Word-level with punctuation separation and special tokens                    |
| Sentiment analysis             | ~365-word valence lexicon with negation handling                             |
| Linguistic features            | Flesch-Kincaid grade, syllable counter, formality classifier                 |
| TF-IDF topic extraction        | ~175-word stopword list, document-frequency weighting                        |
| Cosine warmup scheduler        | Linear warmup + cosine decay (no `torch.optim.lr_scheduler`)                 |
| Welford online statistics      | Incremental mean/variance for streaming feature updates                      |
| Privacy auditor                | Hash-based raw-text leak detection across SQLite                             |
| Edge profiler                  | Parameter counting, INT8 size, P50/P95/P99 latency                           |

---

## Technical Stack

| Layer               | Technology                                         |
|:--------------------|:---------------------------------------------------|
| Deep learning       | PyTorch 2.0                                        |
| Web server          | FastAPI + WebSockets                               |
| Frontend            | Vanilla HTML/CSS/JS (no build step)                |
| Persistence         | SQLite (`aiosqlite` for async)                     |
| Cloud LLM           | Anthropic Claude API (direct `httpx` client)       |
| Encryption          | `cryptography` (Fernet)                            |
| Configuration       | Pydantic v2 + YAML                                 |
| Testing             | pytest + pytest-asyncio                            |
| Lint / format       | ruff + black + mypy                                |

---

## Documentation

The project ships a full MkDocs Material site. Serve it locally:

```bash
poetry install --with docs
poetry run mkdocs serve         # http://127.0.0.1:8000
```

Key documents:

- [**ARCHITECTURE.md**](docs/architecture/full-reference.md) — deep-dive system design,
  math, data flows, and design rationale.
- [**DEMO_SCRIPT.md**](docs/slides/demo-script.md) — the 4-phase interview demo
  script with exact dialogue and timing.
- [**edge_profiling_report.md**](docs/edge/profiling-report.md) — Kirin
  9000 / 9010 / A2 / Smart Hanhan feasibility matrix + MindSpore Lite
  conversion path + power-budget analysis.
- [**docs/huawei/**](docs/huawei/) — HarmonyOS 6 / HMAF integration,
  Kirin deployment, L1–L5 framework, Edinburgh Joint Lab positioning,
  Smart Hanhan deployment, interview talking points.
- [**docs/adr/**](docs/adr/) — 10 Architecture Decision Records (MADR 4.0).
- [**docs/responsible_ai/**](docs/responsible_ai/) — model cards (SLM
  and TCN), data card, accessibility statement.
- [**docs/slides/**](docs/slides/) — 15-slide Marp deck + speaker notes
  + 52 Q&A prep + rehearsal cue sheet.
- [**docs/security/slsa.md**](docs/security/slsa.md), [**docs/security/supply-chain.md**](docs/security/supply-chain.md) —
  Build Level 3 posture, SBOM distribution, image signing, vulnerability
  SLAs.
- [pyproject.toml](pyproject.toml) — dependencies + tooling config.
- [configs/default.yaml](configs/default.yaml) — all hyperparameters
  with inline comments.

## Production Features (beyond the demo)

This repository is configured as a production-grade Python ML service,
not just a notebook. Everything below is opt-in; the core pipeline boots
with zero extra dependencies.

| Area               | Capability                                                                                          |
|:-------------------|:----------------------------------------------------------------------------------------------------|
| Containers         | Multi-stage Dockerfile (non-root, tini PID 1), Compose + hardened prod profile, VSCode devcontainer |
| Kubernetes         | Full `deploy/k8s/` manifests + Helm chart + Kustomize dev/staging/prod overlays + Terraform module  |
| Observability      | OpenTelemetry traces, Prometheus metrics, structlog JSON logs, Sentry (opt-in), Grafana dashboard    |
| LLM observability  | Langfuse tracer with Anthropic Sonnet 4.5 token + cost attribution                                  |
| MLOps              | MLflow tracking, DVC pipeline, SHA-256 checkpoint sidecars, OpenSSF Model Signing v1.0 (sigstore)   |
| Edge               | ONNX export + parity verification, torchao INT4/INT8, ExecuTorch `.pte` hooks                       |
| Supply chain       | CycloneDX SBOM, Syft + Trivy image scans, cosign keyless sign, SLSA L3 provenance, OSSF Scorecard   |
| CI/CD              | 13 workflows (CI, security, sbom, scorecard, semgrep, trivy, docker, release, docs, benchmark, …)   |
| Testing            | Property (Hypothesis), contract (schemathesis), fuzz (atheris), load, mutation (mutmut), chaos, snapshot (syrupy) |
| Benchmarks         | pytest-benchmark suites + Locust WS/REST scenarios + k6 script + SLO targets                        |
| Documentation      | MkDocs Material site with 10 ADRs, model cards, data card, glossary, runbook, accessibility statement |
| Huawei integration | HMAF adapter, Kirin target profiles, ExecuTorch hooks, MindSpore Lite conversion guide             |
| Supply-chain auto  | Renovate (grouped), release-please, commitlint, lefthook git hooks                                  |
| Deliverables       | 15-slide Marp deck, speaker notes, 52 Q&A pairs, research paper, patent disclosure, A0 poster       |

## Next-gen 2026 Technology Stack

Additional opt-in capability families, each soft-imported so the core
service boots without them:

| Family              | Capability                                                                                                         |
|:--------------------|:-------------------------------------------------------------------------------------------------------------------|
| Python toolchain    | uv + uv.lock alongside Poetry; `ty` / mypy / Ruff; Nix flake; Devbox; mise + asdf; justfile                        |
| Hardened containers | Chainguard Wolfi distroless variant (`docker/Dockerfile.wolfi`) — zero H/C CVEs on base image                             |
| MCP server          | Anthropic Model Context Protocol — 7 tools + 5 resources + 2 prompts for Claude Desktop / Code                      |
| Browser inference   | ONNX Runtime Web + WebGPU — TCN runs on the user's GPU; keystroke packets never leave the device                   |
| LLM ecosystem       | DSPy compile-time prompt optimisation, NeMo Guardrails, Pydantic AI, Instructor, Outlines, Logfire, OpenLLMetry    |
| Modern data stack   | DuckDB analytics over SQLite, LanceDB IVF-PQ vector search, Polars streaming features, Ibis portable queries       |
| Distributed         | Lightning Fabric (DDP + FSDP + `torch.compile`), Accelerate, DeepSpeed ZeRO-3, Ray Serve, NVIDIA Triton, vLLM      |
| Edge runtimes       | Apple MLX, llama.cpp GGUF, Apache TVM, IREE, Core ML, TensorRT-LLM, OpenVINO, MediaPipe (plus existing ExecuTorch) |
| Dev experience      | Dagger programmable CI, Tilt local k8s hot-reload, Grafana Pyroscope continuous profiling, Backstage, Alloy        |
| Policy as code      | Kyverno + OPA + Sigstore Policy Controller (cluster), Cedar 4.x (app authz), Falco + Tracee (runtime), Allstar     |
| Future-work code    | Multi-modal port (voice + touch + gaze + accelerometer), Flower federated learning, HarmonyOS DDM sync mock        |
| Brief stretch       | Attention-conditioning aux losses, integrated-gradients interpretability, what-if API, ablation mode, biometric ID |
| Interview package   | 7 Jupyter notebooks, research paper (7 126 words), attorney-ready patent disclosure, exec summary, A0 poster       |

Audit trail: every wave of these additions is accompanied by dated
reports under [`reports/audits/`](reports/audits/), covering security,
code quality, completeness, and documentation. The index is
[`reports/audits/2026-04-23-index.md`](reports/audits/2026-04-23-index.md).

See [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]` for the full list of
additions over the v1.0.0 baseline.

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

## Acknowledgements

Draws inspiration from:

- Eric Xu's **L1–L5 device intelligence framework**.
- **Edinburgh Joint Lab** research on personalisation from sparse signals.
- **HarmonyOS Multi-Agent Framework (HMAF)** and its philosophy of
  on-device-first AI.

---

<div align="center">

*Built by Tamer Atesyakar — UCL MSc Digital Finance & Banking*

</div>
