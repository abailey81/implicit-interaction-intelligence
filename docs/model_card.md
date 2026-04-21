# Model Card — I³ Adaptive SLM

A model card in the style of Mitchell *et al.*, "Model Cards for Model
Reporting" (FAT\* 2019). Covers both the TCN encoder and the adaptive SLM
shipped with I³ 1.0.0.

!!! note "Scope"
    This card describes the **demo-quality** checkpoints shipped with the
    repository. For the full training procedure, see
    [Training](getting-started/training.md).

## Model Details { #model-details }

| Field | Value |
|:------|:------|
| **Name**                     | I³ Adaptive SLM (with TCN encoder) |
| **Version**                  | 1.0.0 |
| **Date**                     | 2026-04 |
| **Type**                     | Decoder-only transformer (SLM) + TCN encoder |
| **Parameters**               | Encoder: ~50K · SLM: ~6.3M · Total: ~6.4M |
| **Precision**                | FP32 train / INT8 dynamic quant at inference |
| **Input**                    | User text (tokenised) + `AdaptationVector` (8-dim) + `UserStateEmbedding` (64-dim) |
| **Output**                   | Next-token distribution over 8192-word vocabulary |
| **Framework**                | PyTorch 2.6+, from-scratch (no HuggingFace) |
| **Licence**                  | MIT |
| **Contact**                  | tamer.atesyakar@bk.ru |
| **Repository**               | <https://github.com/abailey81/implicit-interaction-intelligence> |
| **Novel architecture**       | Per-block cross-attention conditioning on user state (see [research note](research/cross_attention.md)) |

### Architectural summary

- Vocab: word-level, 8192 entries + special tokens.
- Positional encoding: sinusoidal.
- Blocks: 4 × Pre-LN `AdaptiveTransformerBlock`.
- Attention: self (causal), cross (user conditioning), feed-forward.
- Embedding dimensionality: 256; heads: 4; head dim: 64.
- Output projection: weight-tied with token embedding.

## Intended Use { #intended-use }

### Primary intended uses

- Research demonstrations of *implicit-signal* personalisation.
- On-device companion agents where privacy is non-negotiable.
- An educational reference for end-to-end transformer engineering without
  relying on pre-trained backbones.

### Primary intended users

- HMI and human-computer interaction researchers.
- Applied ML engineers evaluating edge-deployable language models.
- Engineering interview panels, as a demo artefact.

### Out-of-scope uses

- **Not a general-purpose assistant.** The model has ~6.3M parameters —
  factual recall and reasoning are intentionally limited.
- **Not a medical, legal, or financial adviser.** The router force-routes
  such topics to the local SLM; the local SLM should not be relied on for
  accurate answers.
- **Not safe for unfiltered multi-tenant deployment.** The shipped build
  has no caller authentication (see [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)).

## Factors { #factors }

### Relevant factors

- **Cognitive load** — the model's simplification/complexity axis.
- **Keystroke tempo** — detected as motor accessibility signal.
- **Style baseline** — formality, verbosity, emotionality, directness.
- **Session length** — long sessions change the EMA weight of the
  long-term profile.
- **Device pressure** — memory and battery bias the router towards local.

### Evaluation factors

We evaluate on synthetic traces spanning eight user states × three
conditioning modes × two device classes.

## Metrics { #metrics }

### Primary

- **Perplexity** on a held-out split of the combined DailyDialog +
  EmpatheticDialogues corpus.
- **Conditioning sensitivity**: ΔPerplexity between `none`, `prefix`,
  `full` conditioning modes.

### Secondary

- **Generation latency**: P50, P95, P99 over 1,000 generations of
  64 tokens, CPU single-threaded.
- **INT8 size on disk**.
- **Adaptation vector diversity**: average pairwise cosine distance of
  conditioning tokens across users. Proxy for whether the model uses the
  signal.

## Evaluation Data { #evaluation-data }

- **Source**: held-out 10 % split of `data/dialogue/` after seeded
  shuffling.
- **Pre-processing**: same PII sanitisation pipeline as production, then
  tokenisation via the shipped 8192-entry vocabulary.
- **Motivation**: public, multi-topic, neutral-domain dialogue.

## Training Data { #training-data }

See [Data Card](data_card.md) for the full datasheet. Summary:

- **Synthetic interaction traces** — 10,000 sessions, 8 Markov-modelled
  user states, seeded.
- **DailyDialog** — licensed CC BY-NC-SA 4.0. Chitchat + emotion labels.
- **EmpatheticDialogues** — licensed CC BY 4.0. Emotion-ground dialogue.

## Quantitative Analyses { #quant }

### Held-out perplexity

| Variant | Val. ppl | Δ vs full |
|:--------|---------:|----------:|
| no conditioning | 21.95 | +19.2 % |
| prefix prefixing| 19.87 | +7.9 %  |
| **full (ours)** | **18.42** | — |

### Latency (INT8, CPU)

| Model | P50 | P95 | P99 |
|:------|----:|----:|----:|
| TCN encoder | 3 ms | 4 ms | 5 ms |
| SLM (64-token gen) | 143 ms | 181 ms | 205 ms |
| **Full pipeline** | **149 ms** | **188 ms** | **213 ms** |

### Device feasibility (50 %-of-memory rule)

| Device | RAM | INT8 + activations | Fits? |
|:-------|----:|-------------------:|:-----:|
| Kirin 9000 (phone)   | 512 MB | ~15 MB | yes |
| Kirin A2 (wearable)  | 128 MB | ~15 MB | yes |
| Smart Hanhan (IoT)   |  64 MB | ~15 MB | tight — encoder-only recommended |

## Ethical Considerations { #ethics }

### Data

- Synthetic data is generator-bound — models trained on it cannot exhibit
  states the generator cannot express.
- DailyDialog and EmpatheticDialogues are English-only, Western-centric,
  and reflect annotator demographics.
- None of the data contains real users' keystroke dynamics (we synthesise
  those).

### Risk assessment

| Risk | Likelihood | Mitigation |
|:-----|:----------:|:-----------|
| **Stereotype reinforcement** (style mirroring amplifies user's phrasing) | Medium | `StyleMirrorAdapter` mirrors, never amplifies; clamped to \([-1, 1]\) |
| **Over-confident generation** (6.3M params, limited factual recall) | High | System prompt constrains tone; router biases factual queries to cloud |
| **Privacy leakage via embeddings** (64-dim vector is not text) | Low | Membership-inference test planned; encryption at rest; no cross-user diary access |
| **Prompt injection via conditioning** | Low | Conditioning tokens pass through a trained projector — no natural-language channel |
| **Adaptation manipulation** (adversarial keystrokes) | Medium | Bounded numeric coercion + rate limiting; adapters clamped |

### Bias and fairness

The adaptation layer mirrors *the user's own baseline* — verbosity,
formality, emotionality, directness. This is not a demographic inference
step; there is no "age" or "gender" estimator anywhere in the system.

### Transparency

Users see their own adaptation vector in the demo UI in real time. Any
shipped product should do the same.

## Caveats and Recommendations { #caveats }

- **Do not deploy without authentication.** The demo build has none; all
  REST and WS endpoints assume a trusted local caller.
- **Do not treat the SLM as authoritative.** At 6.3M parameters it is a
  research artefact. Use the cloud arm for factual queries.
- **Plan for key management.** Fernet at rest is good; key loss means
  total profile loss (by design).
- **Monitor adaptation drift.** If `accessibility` rises sustainedly for
  a user, flag for human review.

## Changelog { #changelog }

- **1.0.0 (2026-04)**: Initial public release. See
  [CHANGELOG](changelog.md).
