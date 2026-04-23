# Layers

One page, one section per layer. Each section states the responsibility,
lists the files involved, shows the public type contract, and links to
deeper treatment where it exists.

!!! tip "Canonical reference"
    The authoritative, maths-heavy deep dive is
    [`docs/architecture/full-reference.md`](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/docs/architecture/full-reference.md).
    This page is a navigation aid.

## Layer 1 — Perception { #l1 }

**Package**: `i3/interaction/`
**Responsibility**: extract a 32-dim feature vector from keystroke events
and raw text without persisting the text.

| File | Role |
|:-----|:-----|
| `types.py`      | `InteractionFeatureVector`, `KeystrokeEvent` |
| `monitor.py`    | Real-time per-user keystroke buffer |
| `features.py`   | Feature extraction + personal baseline tracking |
| `linguistic.py` | Flesch-Kincaid, formality, ~365-word valence lexicon |

The 32 features decompose into four groups of eight:

| Group | Examples |
|:------|:---------|
| Keystroke dynamics   | mean / variance of inter-key interval, burst ratio, pause ratio, correction rate |
| Message content      | length, mean word length, vocabulary diversity, punctuation density |
| Linguistic complexity| Flesch-Kincaid grade, formality score, valence, subjectivity |
| Session dynamics     | deviation-from-baseline z-scores, time since last message, session length |

!!! note "No NLP dependencies"
    Everything — the lexicon, the syllable counter, the formality
    classifier — is implemented from scratch. See
    [`i3/interaction/linguistic.py`](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/i3/interaction/linguistic.py).

## Layer 2 — Encoding { #l2 }

**Package**: `i3/encoder/`
**Responsibility**: map a sequence of feature vectors to a 64-dim
user-state embedding.

| File | Role |
|:-----|:-----|
| `blocks.py`    | `CausalConv1d`, `ResidualBlock` |
| `tcn.py`       | `TemporalConvNet` (dilations `[1,2,4,8]`) |
| `train.py`     | NT-Xent contrastive training loop |
| `inference.py` | Real-time wrapper with KV-cache-equivalent streaming |

Receptive field: four blocks × kernel size 3 × dilations `[1,2,4,8]` gives
a context window of ~61 timesteps — enough to see a full conversational
turn.

See [Training § Encoder](../getting-started/training.md#step-2) and
[ADR 0002](../adr/0002-tcn-over-lstm-transformer.md).

## Layer 3 — User model { #l3 }

**Package**: `i3/user_model/`
**Responsibility**: persist a three-timescale representation of the user.

| File | Role |
|:-----|:-----|
| `types.py`    | `UserProfile`, `SessionState`, `DeviationMetrics` |
| `model.py`    | Instant / Session / Long-term EMAs with `α=0.3` and `α=0.1` |
| `deviation.py`| Welford's online mean/variance for streaming z-scores |
| `store.py`    | Async SQLite (`aiosqlite`) + Fernet at rest |

Welford's algorithm avoids re-reading history for every update:

\[
\begin{aligned}
n &\gets n + 1 \\
\delta &\gets x - \bar{x} \\
\bar{x} &\gets \bar{x} + \delta / n \\
M_2 &\gets M_2 + \delta \cdot (x - \bar{x})
\end{aligned}
\]

See [Privacy](privacy.md) for the Fernet integration and
[ADR 0009](../adr/0009-sqlite-over-redis.md) for the persistence choice.

## Layer 4 — Adaptation { #l4 }

**Package**: `i3/adaptation/`
**Responsibility**: map user state to an 8-dim `AdaptationVector`.

| Dimension | Adapter | Output |
|:----------|:--------|:-------|
| `cognitive_load` | `CognitiveLoadAdapter` | Scalar ∈ [0, 1] — target complexity |
| `style_mirror`   | `StyleMirrorAdapter`   | 4-dim `StyleVector` (formality · verbosity · emotionality · directness) |
| `emotional_tone` | `EmotionalToneAdapter` | Scalar ∈ [0, 1] — warmth |
| `accessibility`  | `AccessibilityAdapter` | Scalar ∈ [0, 1] — simplification |

Orchestrated by `AdaptationController`. Each adapter is deterministic and
decoupled, so any subset may be replaced without re-training the SLM
(conditioning tolerates a slow change in the distribution of `a`).

## Layer 5 — Router { #l5 }

**Package**: `i3/router/`
**Responsibility**: per-message routing between local SLM and cloud LLM
under a Bayesian reward model.

| File | Role |
|:-----|:-----|
| `types.py`        | `RoutingContext` (12-dim), `RoutingDecision` |
| `bandit.py`       | Bayesian logistic regression + Laplace + Newton-Raphson MAP |
| `complexity.py`   | Query-complexity estimator (tokens, rare words, question-word density) |
| `sensitivity.py`  | Regex-based topic sensitivity detector |
| `router.py`       | `IntelligentRouter` with privacy override |

Detailed treatment: [Router](router.md),
[Research: Bandit theory](../research/bandit_theory.md), and
[ADR 0003](../adr/0003-thompson-sampling-over-ucb.md).

## Layer 6a — Local SLM { #l6a }

**Package**: `i3/slm/`
**Responsibility**: generate responses conditioned on user state, on device.

| File | Role |
|:-----|:-----|
| `tokenizer.py`       | Word-level with 8192-entry vocabulary, special tokens |
| `embeddings.py`      | Token + sinusoidal positional |
| `attention.py`       | Multi-head self-attention + KV cache |
| `cross_attention.py` | `ConditioningProjector` + cross-attention (**novel**) |
| `transformer.py`     | Pre-LN `AdaptiveTransformerBlock` × 4 |
| `model.py`           | `AdaptiveSLM` (`d_model=256`, `n_heads=4`) |
| `generate.py`        | Top-k / top-p / repetition-penalty sampling |
| `quantize.py`        | INT8 dynamic quantization |
| `train.py`           | Cosine warmup + cross-entropy loss |

See [Cross-attention conditioning](cross-attention-conditioning.md) and
[ADR 0001](../adr/0001-custom-slm-over-huggingface.md).

## Layer 6b — Cloud LLM { #l6b }

**Package**: `i3/cloud/`
**Responsibility**: call Anthropic Claude with an adaptation-derived
system prompt and PII-stripped user input.

| File | Role |
|:-----|:-----|
| `client.py`         | Async `httpx` client with retries + backoff |
| `prompt_builder.py` | Maps `AdaptationVector` to explicit style instructions |
| `postprocess.py`    | Response-level enforcement (length caps, format constraints) |

!!! warning "Raw text leaves the device"
    Only after PII sanitisation. See [Privacy](privacy.md) for the exact
    regex catalogue and the pre-send scan.

## Layer 7 — Diary { #l7 }

**Package**: `i3/diary/`
**Responsibility**: privacy-safe interaction log — embeddings, topic
keywords, metrics. Never raw text.

| File | Role |
|:-----|:-----|
| `store.py`      | Async SQLite schema (no text columns) |
| `logger.py`     | TF-IDF topic extraction (~175-word stopword list) |
| `summarizer.py` | Template-based session summaries; optional cloud path |

## Cross-cutting — Privacy { #x-privacy }

**Package**: `i3/privacy/` — see the dedicated [Privacy](privacy.md) page.

## Cross-cutting — Profiling { #x-profiling }

**Package**: `i3/profiling/`
**Responsibility**: measure latency, memory, and INT8 size; emit a
device-feasibility report.

| File | Role |
|:-----|:-----|
| `memory.py`  | `tracemalloc` + INT8 size comparison |
| `latency.py` | P50 / P95 / P99 percentile benchmarks with warmup |
| `report.py`  | Markdown device feasibility report |

Target devices: Kirin 9000 (phone, 512 MB, 2.0 TOPS), Kirin A2 (wearable,
128 MB, 0.5 TOPS), Smart Hanhan (IoT, 64 MB, 0.1 TOPS).

See [Operations · Observability](../operations/observability.md) for
runtime telemetry.
