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

I³ was built to demonstrate the three tiers of AI capability required for the
Huawei HMI Lab role: a **custom ML encoder** (a TCN built from scratch), a
**custom small language model** (a ~6.3M-parameter transformer with no
HuggingFace dependency), and **intelligent routing to foundational models**
(a Bayesian bandit decides when cloud capacity is worth the privacy and
latency trade-off). Every non-trivial component — the encoder, the transformer,
the cross-attention conditioning, the Thompson sampling bandit, the sentiment
lexicon, even the cosine warmup scheduler — is implemented from first
principles.

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

```
implicit-interaction-intelligence/
│
├── i3/                                    # Main Python package
│   ├── config.py                         # Pydantic v2 config (17 nested models)
│   │
│   ├── interaction/                      # Layer 1 — Behavioural signal extraction
│   │   ├── types.py                      #   InteractionFeatureVector (32 features)
│   │   ├── monitor.py                    #   Real-time keystroke monitor
│   │   ├── features.py                   #   Feature extraction + baseline tracking
│   │   └── linguistic.py                 #   Flesch-Kincaid, formality, ~365-word lexicon
│   │
│   ├── encoder/                          # Layer 2 — User state encoding (custom TCN)
│   │   ├── blocks.py                     #   CausalConv1d, ResidualBlock
│   │   ├── tcn.py                        #   TemporalConvNet (dilations [1,2,4,8])
│   │   ├── train.py                      #   NT-Xent contrastive loss
│   │   └── inference.py                  #   Real-time inference wrapper
│   │
│   ├── user_model/                       # Layer 3 — Three-timescale user model
│   │   ├── types.py                      #   UserProfile, SessionState, DeviationMetrics
│   │   ├── model.py                      #   Instant / Session / Long-term EMA
│   │   ├── deviation.py                  #   Welford's online algorithm
│   │   └── store.py                      #   Async SQLite (aiosqlite)
│   │
│   ├── adaptation/                       # Layer 4 — Multi-dimensional adaptation
│   │   ├── types.py                      #   AdaptationVector, StyleVector
│   │   ├── dimensions.py                 #   CognitiveLoad, StyleMirror, Tone, A11y
│   │   └── controller.py                 #   Orchestrates the four adapters
│   │
│   ├── router/                           # Layer 5 — Contextual Thompson sampling
│   │   ├── types.py                      #   RoutingContext (12-dim), RoutingDecision
│   │   ├── bandit.py                     #   Bayesian logistic regression + Laplace
│   │   ├── complexity.py                 #   Query complexity estimator
│   │   ├── sensitivity.py                #   Topic sensitivity detector (regex)
│   │   └── router.py                     #   IntelligentRouter (privacy override)
│   │
│   ├── slm/                              # Layer 6a — Custom SLM (no HuggingFace)
│   │   ├── tokenizer.py                  #   Word-level with special tokens
│   │   ├── embeddings.py                 #   Token + sinusoidal positional
│   │   ├── attention.py                  #   MultiHeadSelfAttention + KV cache
│   │   ├── cross_attention.py            #   Cross-attention + ConditioningProjector
│   │   ├── transformer.py                #   Pre-LN AdaptiveTransformerBlock
│   │   ├── model.py                      #   AdaptiveSLM (~6.3M params)
│   │   ├── generate.py                   #   Top-k / top-p / repetition penalty
│   │   ├── quantize.py                   #   INT8 dynamic quantization
│   │   └── train.py                      #   Cosine warmup + cross-entropy
│   │
│   ├── cloud/                            # Layer 6b — Cloud LLM integration
│   │   ├── client.py                     #   Async Anthropic client (httpx)
│   │   ├── prompt_builder.py             #   System prompts from AdaptationVector
│   │   └── postprocess.py                #   Response enforcement
│   │
│   ├── diary/                            # Layer 7 — Privacy-safe interaction diary
│   │   ├── store.py                      #   Async SQLite for diary entries
│   │   ├── logger.py                     #   TF-IDF topic extraction (~175 stopwords)
│   │   └── summarizer.py                 #   Cloud + template session summaries
│   │
│   ├── privacy/                          # Cross-cutting — Privacy by architecture
│   │   ├── sanitizer.py                  #   10 PII regex patterns + auditor
│   │   └── encryption.py                 #   Fernet symmetric encryption
│   │
│   ├── profiling/                        # Cross-cutting — Edge feasibility
│   │   ├── memory.py                     #   tracemalloc + INT8 size measurement
│   │   ├── latency.py                    #   P50/P95/P99 percentile benchmarks
│   │   └── report.py                     #   Markdown reports + device feasibility
│   │
│   └── pipeline/                         # Orchestration
│       ├── types.py                      #   PipelineInput, PipelineOutput
│       └── engine.py                     #   Full 9-step async pipeline
│
├── configs/
│   ├── default.yaml                      # Full production config
│   └── demo.yaml                         # Demo overrides
│
├── server/                               # FastAPI + WebSocket backend
│   ├── app.py                            #   Application factory, lifespan
│   ├── websocket.py                      #   /ws/{user_id} real-time handler
│   └── routes.py                         #   REST endpoints
│
├── web/                                  # Single-page application frontend
│   ├── index.html                        #   Dark-theme SPA
│   ├── css/style.css                     #   Dark charcoal + warm amber accents
│   └── js/
│       ├── app.js                        #   I3App + KeystrokeMonitor
│       ├── websocket.js                  #   Exponential backoff reconnect
│       ├── chat.js                       #   Chat with route/latency badges
│       ├── dashboard.js                  #   Animated gauge bars
│       └── embedding_viz.js              #   Canvas 2D state visualization
│
├── training/                             # CLI training scripts
│   ├── generate_synthetic.py             #   8 user states, Markov transitions
│   ├── prepare_dialogue.py               #   DailyDialog + EmpatheticDialogues
│   ├── train_encoder.py                  #   TCN training CLI
│   ├── train_slm.py                      #   SLM training CLI
│   └── evaluate.py                       #   Perplexity, conditioning sensitivity
│
├── demo/                                 # Demo utilities
│   ├── seed_data.py                      #   Pre-seed profiles and diary
│   ├── scenarios.py                      #   5 scripted interaction arcs
│   └── profiles.py                       #   Pre-built user profiles
│
├── tests/                                # pytest suite (80+ tests)
│   ├── conftest.py                       #   Shared fixtures
│   ├── test_tcn.py                       #   12 tests
│   ├── test_slm.py                       #   15 tests
│   ├── test_bandit.py                    #   18 tests
│   ├── test_user_model.py                #   17 tests
│   └── test_pipeline.py                  #   18 tests
│
├── scripts/                              # Automation
│   ├── setup.sh                          #   One-shot setup
│   ├── run_demo.sh                       #   Launch demo server
│   └── generate_encryption_key.py        #   Fernet key generator
│
├── docs/                                 # Technical documentation
│   ├── ARCHITECTURE.md                   #   System design deep-dive
│   └── DEMO_SCRIPT.md                    #   4-phase interview demo script
│
├── data/                                 # Datasets (gitignored)
├── checkpoints/                          # Model weights (gitignored)
│
├── .env.example
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
└── pyproject.toml
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
python scripts/generate_encryption_key.py  # generates a Fernet key
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

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) — Deep-dive system design, with
  math, data flows, and design rationale
- [**DEMO_SCRIPT.md**](docs/DEMO_SCRIPT.md) — The 4-phase interview demo
  script with exact dialogue and timing
- [pyproject.toml](pyproject.toml) — Dependencies and tooling configuration
- [configs/default.yaml](configs/default.yaml) — All hyperparameters with
  inline comments

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Built for the **Huawei HMI Lab internship** technical presentation.

Draws inspiration from:

- Eric Xu's **L1–L5 device intelligence framework**
- **Edinburgh Joint Lab** research on personalisation from sparse signals
- **HarmonyOS Multi-Agent Framework (HMAF)** and its philosophy of
  on-device-first AI

---

<div align="center">

*Built with care by Tamer Atesyakar — UCL MSc Digital Finance & Banking*

</div>
