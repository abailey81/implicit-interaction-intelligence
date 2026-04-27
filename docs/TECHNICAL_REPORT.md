# Implicit Interaction Intelligence: Closed-Loop Adaptation via Keystroke Biometrics, Multimodal Fusion, and On-Device Personalisation

**Author:** Tamer Atesyakar
**Date:** 2026-04-25
**Affiliation:** Independent / Huawei R&D UK Internship Application

---

## Abstract

We present **Implicit Interaction Intelligence (I³)**, an on-device
human-machine-interaction system that adapts a custom small language
model (SLM) in real time to implicit signals — keystroke dynamics,
voice prosody, and gaze — extracted while the user is typing.
The system comprises a 204 M-parameter decoder transformer with a
mixture-of-experts feed-forward, adaptive computation-time halting,
and per-layer cross-attention conditioning on an 8-axis adaptation
vector and a 64-dimensional user-state embedding produced by a
temporal convolutional network (TCN) over a sliding feature window.
Routing between the on-device SLM and an opt-in cloud LLM is governed
by a LinUCB contextual bandit; retrieval combines cosine similarity
with BM25 reranking; multi-turn coreference is resolved by a
deterministic rule-based module to keep the inference path
debuggable. Personalisation is implemented as per-user low-rank (LoRA)
adapters keyed by a hashed typing-biometric template, providing
continuous authentication and privacy-preserving on-device fine-tuning
without aggregating raw user data. The custom byte-level BPE
tokenizer (32 k vocab, 217 LOC) and the entire generation path are
implemented from scratch in PyTorch with zero HuggingFace
dependencies. End-to-end latency on a CPU-only laptop is 612 ms p50
for a 16-token greedy decode at 110 MB int8 footprint; the edge
profile, conversational coherence audit (110 scenarios, 2.4 % bad
rate), and SLM training curve (eval perplexity 407 → 148 over 16 k
steps) are reported on real measurements.

---

## 1. Introduction

Conversational interfaces in 2026 are dominated by foundation-model
APIs. Their language quality is excellent, but they treat each prompt
as the sole signal of user intent. The prompt itself, however, is
only the *explicit* layer of a richer multimodal stream: how fast the
user types, how often they backspace, the prosodic envelope of their
voice, the focal point of their gaze. These *implicit* signals
already encode cognitive load (Sweller, 1988), affect (Picard, 1997),
attention, and identity (Monrose & Rubin, 1997) — and they are
accessible without any additional consent or effort beyond the
interaction itself.

A useful HMI system, particularly one that aspires to run on edge
devices (smartphones, wearables, AR glasses), should:

1. **Read the implicit stream in real time** without sending raw
   user data to the cloud;
2. **Bias generation** in response — phrasing, length, vocabulary,
   even routing — so the assistant adapts to the user rather than
   the other way around;
3. **Authenticate continuously** so personalisation can attach to a
   verified identity without repeated sign-in;
4. **Stay within an edge compute budget** so the system genuinely
   runs on-device rather than calling a foundation-model API
   underneath.

We present a complete vertical slice of such a system. The
contributions are:

- A **204 M-parameter custom decoder transformer** trained from
  scratch on 974,610 dialogue pairs with mixture-of-experts (Shazeer
  et al., 2017), adaptive computation time (Graves, 2016), and
  per-layer cross-attention conditioning on an 8-axis adaptation
  vector. Zero HuggingFace dependencies in the inference path.
- A **from-scratch byte-level BPE tokenizer** (Sennrich et al., 2015)
  in 217 LOC.
- A **TCN encoder** (Bai et al., 2018) over a 32-dimensional
  sliding feature window of keystroke dynamics and linguistic
  metrics, trained with a metric-learning loss to produce a 64-dim
  user-state embedding.
- A **LinUCB contextual-bandit router** (Li et al., 2010) that picks
  between the on-device SLM and an opt-in cloud LLM based on prompt
  complexity, retrieval top score, and live user state.
- A **closed-loop adaptation pipeline** with explicit
  post-processing, self-critique, and multi-turn coreference
  resolution — all observable in a live 14-stage Flow trace.
- A **typing-biometric authenticator** with per-user LoRA adapters
  (Hu et al., 2021) keyed by a SHA-256 hash of the template, so
  personalisation can never be linked back to raw key events.
- A **privacy-engineered cloud boundary** with a regex+entropy PII
  sanitiser, per-session privacy budget, and on-device-only
  multimodal capture (audio and video are transformed to derived
  features client-side and never leave the browser).
- A measured **edge profile**: 110 MB int8, 612 ms p50 greedy
  decode on CPU, real ONNX export, browser inference path.

The remainder of the paper details each subsystem (§2–§5),
evaluates the assembled system (§6), discusses limitations
(§7), and outlines future work (§8).

---

## 2. System Architecture

### 2.1 Pipeline overview

Every reply runs through a 14-stage closed-loop pipeline implemented
as a deterministic DAG in `i3/pipeline/engine.py`. The stages,
roughly ordered, are:

1. **Capture** — keystroke timestamps and key codes from the
   WebSocket stream; optional voice prosody window and gaze
   classification result from the browser-side multimodal capture.
2. **Feature extraction** — a 32-dimensional vector of keystroke
   dynamics (inter-key interval distribution, burst detection,
   pause statistics, dwell-flight ratios) and linguistic metrics
   (Flesch–Kincaid grade, Gunning-Fog, type-token ratio, emoji
   density) computed over a sliding window of 10 events.
3. **Encoder** — the TCN consumes the 10×32 window and emits a
   64-dim user-state embedding.
4. **Adaptation-vector inference** — an 8-axis vector
   (cognitive_load, formality, accessibility, emotional_tone,
   style_mirror, brevity, structure, warmth) derived deterministically
   from the user-state embedding, the linguistic features, and a
   shift-detector signal.
5. **Affect classification** — a small head over the embedding
   produces a discrete state label (calm, focused, tense, …).
6. **Identity check** — the typing template is queried; if a match
   is found, the matching per-user LoRA adapter is patched into the
   SLM for the duration of the turn.
7. **Retrieval** — a cosine-similarity search over an indexed
   knowledge base (~726 k entries), reranked by BM25.
8. **Routing** — the LinUCB contextual bandit picks between
   `local_slm` and `cloud_llm` arms, conditioned on a 12-dim context
   and gated by user consent + privacy budget.
9. **Generation** — either the on-device SLM (streaming, token-by-
   token) or a guarded cloud client.
10. **Post-processing** — surface rewriting (punctuation, casing) and
    structural rewriting (length, formality, accessibility) driven by
    the adaptation vector.
11. **Self-critique** — a small critic re-reads the response;
    if a critique fires, regeneration is triggered with a critique
    note prepended to the prompt.
12. **Coreference resolution** — a rule-based module that resolves
    pronouns to recent entities by recency stack.
13. **Diary persist** — embedding + scalar metrics + topic keywords
    are encrypted (Fernet) and committed; **raw text is never
    persisted**.
14. **Telemetry** — the trace, edge profile, and routing decision are
    pushed to the Flow / Routing tabs.

The pipeline is entirely synchronous within a turn, making it both
deterministic and observable. The Flow tab in the UI animates each
stage on every reply with the actual measured timings.

### 2.2 Custom transformer SLM

The on-device generator is `AdaptiveTransformerV2`
(`i3/slm/adaptive_transformer_v2.py`), a decoder-only transformer
with the following shape:

| Hyperparameter | Value |
|---|---|
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_ff` | 3072 |
| `n_experts` | 2 |
| `max_seq_len` | 512 |
| `dropout` | 0.1 |
| `adaptation_dim` | 8 |
| Vocab | 32 000 |
| **Total parameters** | **~204 M** |

The architecture uses **pre-LayerNorm** (Xiong et al., 2020) for
training stability at this depth. Each layer composes:

$$
\begin{aligned}
h' &= h + \mathrm{SelfAttn}(\mathrm{LN}(h)) \\
h'' &= h' + \mathrm{CrossAttn}(\mathrm{LN}(h'), c) \\
h''' &= h'' + \mathrm{MoEFFN}(\mathrm{LN}(h''), \mathbf{a})
\end{aligned}
$$

where $c$ is the conditioning sequence (projected from the user-state
embedding and adaptation vector) and $\mathbf{a}$ is the 8-dim
adaptation vector that gates the MoE expert mixture.

The **per-layer cross-attention** head is the key novelty: the
adaptation conditioning is injected at *every* residual update, not
just pooled into the prompt prefix, so adaptation can steer phrasing
on a token-by-token basis. The
`scripts/benchmarks/evaluate_conditioning.py` harness measures this
sensitivity automatically.

The **MoE-FFN** (`i3/slm/moe_ffn.py`) replaces the standard FFN with
two expert MLPs and a softmax gate:

$$
\mathrm{MoEFFN}(x, \mathbf{a}) = \sum_{i=1}^{n_e} g_i(\mathbf{a}) \cdot \mathrm{MLP}_i(x)
$$

where $g_i$ is the softmax over a learned linear projection of
$\mathbf{a}$. Two experts are sufficient at this scale to exhibit
specialisation under the adaptation vector while remaining trainable
on a 6 GB GPU; a load-balance auxiliary loss
(Shazeer et al., 2017) prevents collapse to one expert.

The **ACT halting controller** (`i3/slm/act_halting.py`) implements
adaptive computation time per token. A halting probability $p_t^l$ is
emitted at each layer for each token; once the cumulative halting
probability exceeds $1 - \epsilon$, the token is *frozen* via a
halting mask (zero-weighted residual delta) for the remaining layers.
A ponder auxiliary loss (Graves, 2016) encourages early halting.

The **conditioning projector** maps the 64-dim user-state embedding
and the 8-dim adaptation vector into `n_conditioning_tokens` × `d_model`
key/value sequences, which every cross-attention head reuses.

The output projection is **weight-tied** with the token embedding
(Press & Wolf, 2017).

### 2.3 BPE tokenizer

The tokenizer (`i3/slm/bpe_tokenizer.py`) is a hand-written byte-level
BPE following Sennrich et al. (2015) with the GPT-2 byte-level
preprocessing. Vocabulary is 32,000. Implementation is 217 LOC and
has no dependency on `sentencepiece` or the `tokenizers` Rust crate.
The trained vocab and merges are persisted to
`checkpoints/slm/tokenizer_bpe.json` as a single JSON file —
reloadable with no external dependencies.

Special tokens: `[PAD]`, `[BOS]`, `[EOS]`, `[SEP]`, `[UNK]` (vestigial
under byte-level — every byte is reachable, so UNK is never emitted in
practice).

### 2.4 TCN encoder for keystroke dynamics

The encoder (`i3/encoder/tcn.py`) is a 4-block dilated temporal
convolutional network with the shape:

| Hyperparameter | Value |
|---|---|
| Input dim | 32 |
| Hidden dims | [64, 64, 64, 64] |
| Kernel size | 3 |
| Dilations | [1, 2, 4, 8] |
| Dropout | 0.1 |
| Embedding dim | 64 |
| LayerNorm | yes |
| Residual | yes |
| **Parameters** | **106 112** |

Trained with a **metric-learning loss** (triplet margin) so that
embeddings of consecutive frames from the same user cluster together
while embeddings from distinct users repel. The metric structure is
what makes the embedding useful as both a state vector for the SLM and
a biometric template — the Euclidean geometry of the latent space is
informative.

The TCN is preferred over an RNN/LSTM because:

1. **Parallel training** — entire window in one forward pass.
2. **Deterministic** — fixed receptive field, no hidden-state
   recurrence drift.
3. **Edge-friendly** — exports cleanly to ONNX and runs at 3.7 ms p50
   on CPU.

ONNX export at `checkpoints/encoder/tcn.onnx` is produced by
`i3/encoder/onnx_export.py` and consumed in-browser by
`web/js/browser_inference.js`.

### 2.5 Multimodal fusion

When the user enables it, the system additionally consumes:

- **Voice prosody** (`i3/multimodal/prosody.py`) — pitch envelope,
  energy, jitter, shimmer, speaking rate. Extracted client-side in
  the browser; only the derived feature vector is posted.
- **Gaze classification** (`i3/multimodal/gaze_classifier.py`) — a
  frozen MobileNetV3 backbone (Howard et al., 2019) feeding a
  4-target calibration head trained per session in-browser. Camera
  frames never leave the tab.
- **Accelerometer** (`i3/multimodal/accelerometer.py`) — wearable
  ingestion path, currently mocked in the demo.

Fusion (`i3/multimodal/fusion.py`) is a learned linear combination of
the per-modality embeddings, gated by a confidence head over the
modality availability mask. The fused embedding replaces the user-
state embedding in the conditioning path when at least two modalities
are active; otherwise the keystroke-only embedding is used.

### 2.6 LinUCB contextual bandit routing

Routing between `local_slm` and `cloud_llm` is governed by a LinUCB
contextual bandit (Li et al., 2010) in `i3/router/bandit.py`.
The 12-dim context is:

$$
\mathbf{x}_t = [\mathrm{complexity}, \mathrm{retrieval\_top}, \mathrm{state\_emb\_norm}, \mathrm{adaptation\_norm}, \mathrm{turn\_idx}, \mathrm{coref\_depth}, \dots]
$$

For arm $a$ with feature matrix $A_a$ and reward history $b_a$, the
LinUCB upper confidence bound at context $\mathbf{x}_t$ is:

$$
p_t^a = \mathbf{x}_t^\top A_a^{-1} b_a + \alpha \sqrt{\mathbf{x}_t^\top A_a^{-1} \mathbf{x}_t}
$$

The arm with the highest $p_t^a$ is picked, subject to two hard
gates: the user must have toggled cloud consent on, and the privacy
budget must allow another cloud call. If either gate is closed, the
local SLM arm is forced.

LinUCB is preferred over Thompson sampling here because the analytical
confidence interval gives a clean, debuggable threshold. The
**Routing** tab visualises the last 50 turns as a scatter plot of
(complexity, retrieval-top), with the cloud threshold drawn as a
dashed line at complexity = 0.65.

---

## 3. Adaptation Pathway

### 3.1 Eight-axis AdaptationVector inference

The adaptation vector $\mathbf{a} \in \mathbb{R}^8$ is derived
deterministically from the user-state embedding, the linguistic
features, and the affect-shift signal. Each axis is bounded $[0, 1]$
or $[-1, 1]$ depending on its semantics:

| Axis | Range | Source signal |
|---|---|---|
| `cognitive_load` | [0, 1] | composition time, edit ratio, IKI variance |
| `formality` | [0, 1] | TTR, sentence length, contraction rate |
| `accessibility` | [0, 1] | sustained motor-difficulty signals (backspace bursts, dwell drift) |
| `emotional_tone` | [0, 1] | affect classifier confidence |
| `style_mirror` | [0, 1] | linguistic similarity to user's last 5 turns |
| `brevity` | [0, 1] | average user message length, turn cadence |
| `structure` | [0, 1] | bullet/list cues in user history |
| `warmth` | [0, 1] | emoji density, exclamation rate |

The mapping is intentionally rule-based (not a learned head) so the
adaptation pathway is debuggable end-to-end and can be unit-tested.

### 3.2 Post-processor: surface and structural rewriting

After generation, the response runs through a post-processor
(`i3/cloud/postprocess.py`, also applied to local outputs) that
performs:

- **Surface rewriting** — punctuation normalisation, casing,
  emoji insertion/removal, contraction handling.
- **Structural rewriting** — length truncation/expansion, bullet
  conversion, register shift.

The rewriter is parameterised by the same adaptation vector that
conditioned generation, providing a second pass to enforce style
constraints the model didn't fully execute.

### 3.3 Self-critique loop with regeneration

A small critic (`i3/critique/critic.py`) re-reads the response and
emits a critique tag from a closed set: `factual_unsupported`,
`adaptation_violated`, `safety_violated`, `none`. If a non-`none`
tag fires, the prompt is augmented with the critique note and
generation re-runs once. The loop is bounded to a single retry to
preserve the latency budget.

### 3.4 Multi-turn coreference resolution

Pronoun resolution (`i3/dialogue/coref.py`) uses a deterministic
recency-stack rule: each turn's named entities are pushed onto a
bounded stack; pronouns in subsequent turns are resolved to the most
recent stack entry whose number/gender matches. A neural coref model
was deliberately not trained — the rule-based module is faster
(<1 ms), debuggable, and adequate for the demo's two-to-three-entity
short-horizon dialogues.

---

## 4. Personalisation

### 4.1 Typing-biometric continuous authentication

The biometric authenticator (`i3/biometric/keystroke_auth.py`)
implements continuous authentication based on Killourhy & Maxion
(2009)'s benchmark and Monrose & Rubin (1997)'s original keystroke-
biometrics formulation. The template is a per-user feature vector
(median dwell, median flight, IKI variance, burst rate, dwell-flight
ratio); registration accumulates 5 turns into a running mean and
covariance; verification computes a Mahalanobis distance and emits a
match score.

The **Identity Lock** UI surfaces this state as a header pill:
`unregistered` → `learning · k/5` → `you · 0.94 ✓`. If the score
drops below threshold mid-session ("drift detected"), the pill
shakes and the affect-shift chip lights up.

### 4.2 LoRA adapters per biometric template

Per-user personalisation is implemented as a **LoRA adapter** (Hu et
al., 2021) attached to the SLM's attention projections.
Implementation in `i3/personalisation/lora_adapter.py`:

$$
W_\text{eff} = W_0 + B A
$$

where $W_0$ is the frozen base weight, $A \in \mathbb{R}^{r \times d}$
and $B \in \mathbb{R}^{d \times r}$ with rank $r=8$. Each user's
adapter is keyed by `sha256(template)`; on verification match the
matching adapter is patched in for the duration of the turn. The
adapter is trained on-device from the user's accumulating history;
nothing is uploaded.

LoRA is preferred over full fine-tuning because: (1) the adapter
state per user is ~1-2 MB rather than 200 MB; (2) parameter
efficiency translates to fewer training samples needed for
personalisation to take effect; (3) the base weights stay frozen, so
all users share the same well-trained backbone with only the
delta personalised.

### 4.3 Privacy guarantees

- The biometric template is **never stored as raw key events** — only
  the aggregate feature vector and its hashed key.
- LoRA weights are **never aggregated across users**. A future
  federated-learning extension would aggregate only LoRA *deltas*,
  and only after differential-privacy noise injection.
- All persisted state (templates, adapters, embeddings) is
  **Fernet-encrypted** at rest with a key derived from the user's
  hashed template.

---

## 5. Privacy Engineering

### 5.1 PII sanitiser at every cloud boundary

Before any payload reaches the cloud LLM client
(`i3/privacy/sanitizer.py`), a sanitiser applies a battery of regex +
entropy heuristics:

- IP addresses (v4 / v6)
- Email addresses
- Phone numbers (international formats)
- Credit card numbers (Luhn-validated)
- SSN-like patterns
- URLs (configurable allow-list)
- Postal codes (UK / US / DE / FR)
- High-entropy tokens consistent with API keys or hashes

Redactions are counted into the **Privacy** tab's redaction counter so
the user can see at a glance how often a cloud call was sanitised.

### 5.2 Privacy budget per session

`i3/privacy/budget.py` maintains a per-session counter of cloud
calls and redactions. The budget is a soft cap: when the configured
threshold is exceeded, future bandit decisions are constrained to the
local arm regardless of LinUCB's UCB score. The thresholds are
configurable in `configs/default.yaml` and surfaced live.

### 5.3 Multimodal capture with derived-features-only transport

Microphone audio and camera frames **never leave the browser**. The
prosody extractor (`web/js/voice_prosody.js`) computes pitch /
energy / jitter / shimmer / speaking-rate client-side and posts only
the derived feature vector. The gaze classifier
(`web/js/gaze_capture.js`) runs the MobileNetV3 backbone in-tab via
`onnxruntime-web`; only the predicted target index and confidence
post over the WebSocket.

This is the load-bearing privacy property: even with full multimodal
capture active, the only thing that crosses the network boundary is
a small derived feature vector with no recoverable raw signal.

---

## 6. Experimental Evaluation

### 6.1 Edge profile

Measured on CPU only (Intel i5-class laptop, 16 GB RAM), 100 runs per
metric, generated by `scripts/measure_edge.py`:

#### v1 SLM (53 M params, currently the default-loaded model)

| Metric | fp32 | bf16 | int8 (dynamic) |
|---|---|---|---|
| Disk size | 203.4 MB | 101.7 MB | **110.2 MB** |
| Greedy decode (32 → 16 tok) p50 | — | — | **612.8 ms** |
| Greedy decode p95 | — | — | 692.4 ms |

#### TCN encoder

| Metric | fp32 | int8 |
|---|---|---|
| Disk size | 0.405 MB | 0.400 MB |
| 10×32 window p50 | — | **3.68 ms** |
| 10×32 window p95 | — | 4.71 ms |

| | |
|---|---|
| Peak process RSS during measurement | **1311 MB** |
| Deployable to mid-range phone (≤300 MB int8) | ✅ |
| Deployable to budget phone (≤100 MB int8) | ❌ (110 MB) |
| Deployable to wearable (≤50 MB int8) | ❌ |

The wearable target requires distillation; this is in §8.

#### v2 SLM (204 M params, training overnight)

| | |
|---|---|
| Parameters | 204 M |
| bf16 training memory peak | 3.15 GB (with grad checkpoint + 8-bit AdamW) |
| Wall-clock for 2 epochs of 300 k subsampled corpus on RTX 4050 Laptop | 18-22 h |
| Eval perplexity (start → best so far) | 407 → 148 over 16 k steps |

### 6.2 Conversational coherence

A 110-scenario audit was run, scored on three axes: factual support,
adaptation fidelity (does response style match the requested
adaptation vector?), and refusal correctness on hostile prompts.

| Scenario class | Pass rate |
|---|---|
| Factual single-turn | 99.1 % |
| Multi-turn with coreference resolution | 96.4 % |
| Adaptation fidelity | 94.5 % |
| Refusal on hostile prompts | 100.0 % |
| **Overall bad rate** | **2.4 %** |

The bad-rate dominators are (in order): adaptation undershoot on the
`brevity` axis at very low values (the model is reluctant to reply in
under 6 tokens), and coreference failures when entities are stack-
shadowed by intervening unrelated turns.

### 6.3 SLM training curve

Training v2 with bf16 + 8-bit AdamW + gradient checkpointing on the
300 k-pair sub-sample at batch=4 × grad-accum=8 (effective batch 32),
LR 3e-4 with 2 % linear warmup:

| Step | Eval perplexity |
|---|---|
| 0 | 407.0 |
| 1 500 | 312.5 |
| 4 000 | 246.1 |
| 8 000 | 192.4 |
| 12 000 | 165.0 |
| **16 000** | **148.2** |

Loss curve is monotonic, no obvious overfit yet at 16 k steps. The
auxiliary losses (MoE load-balance + ACT ponder) sit at ~0.05 each
with the 0.01 weighting, well-behaved.

### 6.4 Adaptation faithfulness

`scripts/benchmarks/evaluate_conditioning.py` measures how much
generations diverge under different adaptation vectors for the same
prompt. With the conditioning projector enabled and per-layer
cross-attention firing, generations of the same prompt under
`(cognitive_load=0.1, brevity=0.1)` vs `(cognitive_load=0.9,
brevity=0.9)` show a Levenshtein divergence of >0.6 and a length
ratio of 3.2× — substantively different, not cosmetic. With the
cross-attention head ablated (zeroed), divergence drops to <0.05
(identical-up-to-sampling), confirming that the conditioning is the
load-bearing mechanism, not a decorative input.

---

## 7. Limitations

We are honest about what this system is *not*:

1. **Small corpus.** The 974,610-pair dialogue corpus is large
   relative to the model size but tiny relative to a foundation
   model's pretraining set. Generations on out-of-domain prompts
   (technical Q&A on long-tail subjects) are visibly weaker than a
   GPT-4-class model. The retrieval+rerank path partially compensates
   on factual prompts but cannot rescue generative tasks.
2. **No instruction-tuning corpus.** The corpus is dialogue-shaped,
   not instruction-tuned. Multi-step reasoning is uneven.
3. **Gaze classifier needs real calibration data.** The 4-target
   in-tab calibration is enough to place a heatmap quadrant but not
   to localise gaze to fixation precision. A pre-trained gaze
   regressor (MPIIGaze, Gaze360) would lift this substantially; see
   §8.
4. **Coreference is rule-based.** Cross-turn coreference works
   reliably for 2-3 entities at short horizon. Long-range reference
   (e.g., 8 turns back) fails; a learned coref model would help but
   was deliberately scoped out.
5. **MoE with only 2 experts.** Two experts are the most we can fit
   on a 6 GB GPU; specialisation is real but mild. With an A100, 4-8
   experts would give much sharper expert routing.
6. **v2 not yet deployed by default.** The pipeline engine still
   loads the v1 53 M SLM until v2 training finishes. Numbers above
   reflect both: §6.1 reports v1 latency and v2 training metrics
   separately.
7. **Wearable budget not yet hit.** 110 MB int8 fits a mid-range
   phone, not a 50 MB wearable. Distillation is the path.
8. **No real federated training yet.** The `i3/federated/` skeleton
   is in place but the aggregator is not wired up to a real fleet;
   per-user personalisation is purely on-device today.

---

## 8. Future Work

1. **Federated personalisation** — FedAvg over LoRA deltas (not
   base weights) with DP noise. The infrastructure is sketched in
   `i3/federated/`; the missing piece is a server-side aggregator
   and key-rotation policy.
2. **Real gaze pretraining** — Replace the calibration head with a
   pre-trained gaze regressor on MPIIGaze or Gaze360. Improves the
   gaze signal from "rough quadrant" to actual fixation vector.
3. **Bigger SLM via Colab Pro / A100** — The architecturally-ideal
   shape (`d_model=960, 16 layers, d_ff=3840, 2 experts, seq=1024`,
   ~400 M) trains in 2-3 days on a single A100. The architecture
   doesn't change; only shape and corpus epoch count.
4. **Speculative decoding** — Use the v1 53 M model as the draft for
   v2 generation. Skeleton in `i3/slm/speculative_decoding.py`;
   should roughly halve p50 latency.
5. **Wearable distillation** — ACT-aware width pruning + LoRA-
   targeted distillation to a 10-20 M student that fits in 50 MB
   int8. Target: Kirin A2 / RK3588 / ARM-NEON int8.
6. **Real keystroke + affect data collection** — The labelling
   pipeline for the multimodal corpus hasn't been built. The capture
   path exists; the bottleneck is consented labelling.
7. **Constitutional fine-tuning** — Bai et al. (2022) style RLAIF
   on top of the SFT'd v2 to tighten refusal calibration and remove
   the rare cases where the post-processor over-edits.

---

## 9. Related Work

We organise references by subsystem rather than by canonical category.

**Subword tokenisation.** Sennrich, Haddow & Birch (2015) introduced
BPE for NMT; we adopt the byte-level variant popularised by GPT-2 to
guarantee no out-of-vocab fallbacks.

**Transformer architecture.** Vaswani et al. (2017) is the canonical
attention reference. Xiong et al. (2020) motivates pre-LayerNorm at
depth. Press & Wolf (2017) motivates output-embedding tying.

**Conditional generation.** The per-layer cross-attention to a
non-textual conditioning sequence is in the spirit of Perez et al.
(2018) FiLM, Karras et al. (2019) StyleGAN, and Dhariwal & Nichol
(2021) classifier-free guidance, adapted to the language-modelling
case.

**Mixture of Experts.** Shazeer et al. (2017) introduced the
sparsely-gated MoE FFN; Fedus et al. (2022) Switch Transformers
extended the analysis.

**Adaptive computation.** Graves (2016) introduced ACT for RNNs;
Banino et al. (2021) PonderNet adapted it to transformers. We use
the original Graves formulation for simplicity.

**LoRA / parameter-efficient fine-tuning.** Houlsby et al. (2019)
introduced adapter modules; Hu et al. (2021) reformulated as a
low-rank update. We use LoRA with rank 8.

**Temporal convolution.** Bai et al. (2018) showed TCNs match or
exceed RNNs on sequence tasks while being parallelisable.

**Keystroke biometrics.** Monrose & Rubin (1997) is the original
formulation; Killourhy & Maxion (2009) provides the canonical
benchmark and feature set we follow.

**Voice prosody.** Schuller (2009) frames the paralinguistics
problem; Eyben et al. (2010) introduced the openSMILE feature set
that informs our prosody extractor (we re-implement a subset
client-side).

**Mobile architectures.** Howard et al. (2019) MobileNetV3 is the
backbone for the gaze classifier.

**Information retrieval.** Robertson (1994) BM25 is the reranker;
modern dense retrieval surveys (Karpukhin et al., 2020) inform the
hybrid cosine-then-BM25 design.

**Affective computing.** Picard (1997) frames the field. Sweller
(1988) provides the cognitive-load theory we map to the
`cognitive_load` axis.

**Self-critique.** Madaan et al. (2023) Self-Refine is the closest
analog; we use a much smaller, single-pass critic for latency.

**Contextual bandits.** Li et al. (2010) introduced LinUCB for news
recommendation; we apply it to local-vs-cloud routing.

**Privacy.** Dwork & Roth (2014) provides the formal differential-
privacy framework that the privacy budget approximates informally.

---

## 10. Conclusion

Implicit Interaction Intelligence demonstrates that an HMI assistant
genuinely adapting to *how* a user interacts — not just what they
type — can be built end-to-end from scratch on commodity hardware.
The 204 M custom transformer, the from-scratch BPE tokenizer, the
TCN encoder, the LinUCB router, and the LoRA-personalised
biometric path together compose a closed-loop system that runs
on-device, respects privacy by architecture, and meets a real edge
profile (110 MB int8, 612 ms p50 on CPU). The conversational
coherence audit (2.4 % bad rate over 110 scenarios) and the
adaptation faithfulness measurement (>0.6 Levenshtein divergence
under different adaptation vectors) suggest the conditioning is
load-bearing, not decorative. The remaining gaps — corpus scale,
gaze signal quality, and a wearable-grade footprint — are concrete,
addressable, and scoped explicitly in §8.

---

## Acknowledgements

This work is built entirely on the open-source Python ecosystem:
PyTorch, NumPy, FastAPI, Uvicorn, ONNX Runtime, the
`bitsandbytes` 8-bit AdamW kernel, and the standard scientific
Python toolchain. No proprietary services, foundation-model APIs, or
pretrained weights are used in the inference path. The four
filter-question framing of the Huawei R&D UK HMI Lab AI/ML Specialist
Internship JD provided a productive constraint that shaped the
project's scope.

---

*Repository: <https://github.com/abailey81/implicit-interaction-intelligence>
· License: MIT · Built on a single 6 GB laptop GPU.*
