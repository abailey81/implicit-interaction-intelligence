# Provisional Patent Invention Disclosure

> **Status.** This is a provisional invention-disclosure draft prepared
> by the inventor for internal review and subsequent formal drafting by
> a UK or European patent attorney. It is **not** a filed application
> and contains no representations made on behalf of any third party.
> References to Huawei products, HarmonyOS, or Kirin NPUs are provided
> solely as background-of-the-art context for the patent attorney and
> must not be read as indicating any affiliation, endorsement, or
> filing commitment by Huawei Technologies, Huawei UK, or any
> subsidiary thereof.

---

## Title of Invention

**System and Method for Architectural Conditioning of Generative
Language Models from Implicit Interaction Signals via Dedicated
Per-Layer Cross-Attention.**

## Inventor(s)

- **Tamer Atesyakar** (sole inventor)

## Priority Date (placeholder)

**2026-04-29** — placeholder date of first full disclosure. Attorney to
set the actual priority date on filing.

## Filing Jurisdiction (intended)

United Kingdom Intellectual Property Office (UKIPO) provisional
application, with PCT extension contemplated within the 12-month
priority window.

---

## 1. Field of Invention

The invention relates to the field of on-device artificial intelligence,
specifically to the personalisation of small generative language models
(SLMs) operating on memory- and power-constrained edge hardware. More
particularly, the invention concerns an architectural mechanism by
which implicit interaction signals — such as keystroke dynamics,
linguistic complexity, and session rhythm — are transformed into a
continuous conditioning tensor that is consumed by dedicated
cross-attention sub-layers within a transformer-based generator, so
as to adapt the generator's output to a current inferred user state
without consuming the generator's input context and without requiring
retraining for each user.

The invention applies to, without limitation, conversational AI
companions, accessibility-adaptive user interfaces, smart-wearable
assistants, IoT companion devices, and vehicle-cabin interactive
systems.

---

## 2. Background of the Invention

### 2.1 The problem

On-device generative language models face a structural personalisation
problem. The dominant technique in the art for personalising a
transformer-based generator is to include a natural-language
description of the user in the model's input context — commonly
referred to as a "system prompt" or "instruction prefix":

> "You are talking to a user who prefers short, informal responses
> and is currently under high cognitive load. Simplify your language."

This technique has several well-documented limitations when applied at
edge scale:

1. **Context consumption.** Every token dedicated to describing the
   user is a token that cannot be used for the actual interaction.
   For a 256-token context window, a 32-token persona prefix is
   12.5 % of capacity.
2. **Attention dilution.** As the conversation grows, the
   system-prompt tokens are positioned further from the decoding
   position, and their contribution to the attention computation
   weakens.
3. **Instruction-following limits at small scale.** Small language
   models (≤ 10M parameters typical for on-device deployment)
   follow nuanced instruction prefixes unreliably. A prompt
   requesting "simpler language" may be ignored partially or
   entirely.
4. **Lack of continuous adaptation.** System prompts are discrete;
   updating them per turn requires costly prompt regeneration and
   incurs prefill latency.

### 2.2 Prior mechanisms in the art

Several parameter-efficient personalisation techniques exist in the
art, none of which solves the edge-scale on-device personalisation
problem.

- **Prompt tuning** (Lester, Al-Rfou, Constant, 2021) and
  **P-tuning** (Liu et al., 2022) introduce *continuous* soft
  prompts as learnable prefixes to the token embedding sequence.
  These mechanisms still consume context budget (soft prompts occupy
  token positions in the attention sequence), require gradient-based
  training per task, and scale their efficacy with base-model size.
- **Prefix-tuning** (Li and Liang, 2021) prepends learnable
  key/value tensors to each transformer layer's attention keys and
  values. While this does avoid consuming input-token context, the
  prefix is *learned and static* per task — it does not condition
  dynamically per forward pass on a per-user, per-moment state
  vector.
- **Low-Rank Adaptation (LoRA)** (Hu et al., 2022) adds low-rank
  weight deltas to attention projections. LoRA requires: (a) a
  pretrained frozen base model; (b) per-user weight-delta storage
  if personalisation is user-specific; (c) re-loading of weight
  deltas at inference, which is incompatible with dynamic
  per-turn adaptation. LoRA also presumes the existence of a large
  base model that is fine-tuned; at edge scale the base may itself
  be as small as the deltas.
- **Adapter layers** (Houlsby et al., 2019) insert small trainable
  bottleneck modules between transformer sub-layers. Like LoRA,
  adapters are *learned per task* and are not dynamic per forward
  pass.

### 2.3 Why each is insufficient for on-device SLMs at edge scale

| Mechanism in art    | Limitation relative to the present invention                                                                                                                             |
| :------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| System prompt       | Consumes context; dilutes with length; relies on large-model instruction following.                                                                                      |
| Prompt tuning       | Consumes context; static per task, not per user-moment.                                                                                                                  |
| P-tuning            | Consumes context; requires gradient fitting per task.                                                                                                                    |
| Prefix tuning       | Does not consume input context but prefix is learned per task and static; not a function of a per-forward-pass user-state tensor.                                        |
| LoRA                | Requires pretrained base; per-user deltas require storage and loading; not dynamic per forward pass; unsuitable for real-time state-dependent adaptation.               |
| Adapter layers      | Learned per task; static at inference; not conditioned on a real-time implicit signal.                                                                                   |

The present invention fills this gap: it introduces a **per-layer
cross-attention mechanism** that consumes a **dynamically-computed
conditioning tensor** derived from **implicit behavioural signals**,
with the conditioning computed *from scratch at every forward pass*
and with **no base-model requirement** (the invention is compatible
with both pretrained and randomly-initialised generators).

---

## 3. Brief Summary of the Invention

The invention provides a method and system for adapting the output of
a generative language model to a current inferred user state, wherein
the user state is represented as a fixed-dimensional tensor computed
from implicit interaction signals (e.g., keystroke dynamics,
linguistic complexity, session rhythm) without requiring the user to
supply any explicit personalisation data. The user-state tensor is
projected, by a small multi-layer perceptron termed the
**Conditioning Projector**, into a small fixed number *K* of
conditioning tokens (typically *K* = 4) each of dimension equal to
the generator's hidden size. Each transformer block of the generator
is augmented with a dedicated **cross-attention sub-layer** positioned
between the self-attention and feed-forward sub-layers, wherein the
query is derived from the token sequence being generated and the keys
and values are derived from the conditioning tokens. The result is
that every layer and every token position of the generator's forward
pass is conditioned on the user state, without any tokens of the
input context being consumed for that purpose, and without any
retraining of the generator being required to adapt to a new user.

---

## 4. Detailed Description

### 4.1 System overview

With reference to **Figure 1** (see §6.1), the invention comprises:

- a **Signal Perception Module** that extracts a fixed-dimensional
  feature vector from implicit interaction signals;
- a **Representation Encoder** (e.g., a dilated causal Temporal
  Convolutional Network with contrastive training) that maps the
  feature vector to a unit-normalised user-state embedding;
- a **User-State Model** maintaining three timescales of the
  embedding (instant, session EMA, long-term EMA) and running
  per-feature statistics via an online algorithm;
- an **Adaptation Controller** producing a small fixed-length
  adaptation vector from the user-state model;
- a **Conditioning Projector** mapping the concatenation of the
  user-state embedding and the adaptation vector to *K* conditioning
  tokens; and
- a **Conditioned Generator** — a transformer-based language model
  wherein each block includes a dedicated cross-attention sub-layer
  that attends from the token sequence to the *K* conditioning
  tokens.

### 4.2 Signal perception

The Signal Perception Module computes a feature vector of dimension
*D* (in one embodiment, *D* = 32) organised into *G* interpretable
groups (in one embodiment, *G* = 4, comprising keystroke dynamics,
message content, linguistic complexity, and session dynamics). The
signal-extraction routines are computable on commodity mobile silicon
in under 2 ms per message and do not require network access.

### 4.3 Representation encoder

The Representation Encoder, in one embodiment, is a Temporal
Convolutional Network (TCN) comprising *L* residual blocks of
dilated causal 1D convolutions with kernel size *k* and dilation
schedule $d_\ell = 2^{\ell - 1}$ for $\ell \in \{1, \dots, L\}$.
The receptive field is

$$R = 1 + (k - 1)(2^L - 1).$$

In a preferred embodiment, *L* = 4, *k* = 3, yielding *R* = 31
native (≈ 61 effective with residual paths). The encoder is trained
with a contrastive objective such as NT-Xent (normalised
temperature-scaled cross entropy) over augmented views of user
sessions. Other encoders — LSTM, GRU, mini-transformer — fall within
the scope of the invention.

### 4.4 User-state model

The User-State Model maintains:

- the instantaneous embedding $z_t$;
- a session-timescale EMA $\mu_t^{\text{S}} = (1 - \alpha_\text{S})\mu_{t-1}^{\text{S}} + \alpha_\text{S} z_t$ with $\alpha_\text{S}$ typically in $[0.2, 0.5]$;
- a long-term EMA $\mu_t^{\text{LT}}$ with $\alpha_\text{LT}$ typically in $[0.05, 0.2]$;
- per-feature running mean and variance via Welford's online
  algorithm (Welford, 1962), yielding z-scores used as deviation
  features.

### 4.5 Adaptation Controller

The Adaptation Controller produces a fixed-length real-valued
vector $\mathbf{v}_\text{adapt} \in \mathbb{R}^{A}$ (in one
embodiment, *A* = 8) whose components correspond to interpretable
behavioural axes (cognitive load, style mirror, emotional tone,
accessibility simplification).

### 4.6 Conditioning Projector

The Conditioning Projector is a two-layer multi-layer perceptron
that maps the concatenation $[\mathbf{v}_\text{adapt};
\mathbf{z}_\text{user}] \in \mathbb{R}^{A + E}$ (in one embodiment,
$A + E = 8 + 64 = 72$) to a flat tensor of size $K \cdot d$
(in one embodiment, $K = 4$, $d = 256$), reshaped to
$\mathbf{C} \in \mathbb{R}^{K \times d}$:

$$\mathbf{C} = \text{Reshape}_{K \times d}\!\big(W_2 \cdot \text{GELU}(W_1 \cdot [\mathbf{v}_\text{adapt}; \mathbf{z}_\text{user}] + b_1) + b_2\big).$$

The total parameter count of the Conditioning Projector is small
(≈ 25 k in the preferred embodiment).

### 4.7 Conditioned Generator: dedicated per-block cross-attention

Each transformer block of the Conditioned Generator, with reference
to **Figure 2**, is augmented with a dedicated cross-attention
sub-layer positioned between the self-attention sub-layer and the
feed-forward sub-layer:

```
x  ←  x + SelfAttn(LN_1(x), causal_mask)      # sub-layer 1
x  ←  x + CrossAttn(LN_2(x), C, C)            # sub-layer 2 (INVENTIVE)
x  ←  x + FeedForward(LN_3(x))                # sub-layer 3
```

where `CrossAttn(Q, K, V)` is multi-head attention with the query
derived from the token sequence and the keys and values derived from
the conditioning tokens **C**:

$$\text{CrossAttn}(X_\text{tok}, \mathbf{C}) = \text{softmax}\!\left(\frac{X_\text{tok} W_Q (\mathbf{C} W_K)^\top}{\sqrt{d_k}}\right) \mathbf{C} W_V.$$

Each block has its own $W_Q, W_K, W_V$ projection matrices for the
cross-attention sub-layer. The conditioning tensor **C** is
identical across blocks within a single forward pass but is
recomputed per forward pass.

### 4.8 Algorithmic Claim Chart

The invention is claimed in three independent claims (method,
system, computer-readable medium) and seven dependent claims.

#### Independent Claim 1 (Method)

*A method for conditioning a generative language model on a user
state vector via a dedicated cross-attention sub-layer at each
transformer block, wherein the user state vector is derived from
behavioural signals without explicit user input, the method
comprising:*

1. receiving, at a signal perception module, one or more implicit
   interaction signals from a user, wherein said signals do not
   include any explicit user self-description;
2. extracting, from said signals, a feature vector of dimension *D*;
3. encoding, by a representation encoder, said feature vector to a
   user-state embedding of dimension *E*;
4. producing, by an adaptation controller reading persistent
   statistics of prior user-state embeddings, an adaptation vector
   of dimension *A*;
5. projecting, by a conditioning projector, the concatenation of
   said user-state embedding and said adaptation vector to *K*
   conditioning tokens each of dimension *d*;
6. executing a forward pass of a transformer-based generator having
   a plurality of transformer blocks, wherein at each of said
   transformer blocks a dedicated cross-attention sub-layer
   computes attention from a query derived from a token sequence
   to keys and values derived from said *K* conditioning tokens;
7. emitting, from the generator, a response token or token
   distribution conditioned on said *K* conditioning tokens;

*without any of said conditioning tokens being appended to the
input-token context of the generator.*

#### Independent Claim 2 (System)

*A system configured to perform the method of Claim 1, comprising
one or more processors, one or more memories, and one or more
non-transitory computer-readable media bearing instructions that
when executed by said one or more processors cause the system to
perform the method of Claim 1.*

#### Independent Claim 3 (Computer-readable medium)

*One or more non-transitory computer-readable media bearing
instructions that when executed by one or more processors cause said
processors to perform the method of Claim 1.*

#### Dependent Claim 4 (Conditioning projector MLP)

*The method of Claim 1, wherein the conditioning projector comprises
a two-layer multi-layer perceptron with a nonlinearity selected from
the group consisting of GELU, ReLU, and SiLU, and wherein the
projector output is reshaped to form K conditioning tokens each of
dimension equal to the hidden size d of the generator's transformer
blocks.*

#### Dependent Claim 5 (Four-token conditioning)

*The method of Claim 1, wherein K = 4.* Alternative claim bracket:
*wherein K is selected from the group consisting of 2, 4, 8, and 16.*

#### Dependent Claim 6 (Three-timescale user model)

*The method of Claim 1, wherein the adaptation controller reads
from a user-state model comprising at least three timescales,
namely (i) an instant-timescale embedding equal to the most recent
encoder output, (ii) a session-timescale exponential moving average
with smoothing coefficient in the range 0.2 to 0.5, and (iii) a
long-term-timescale exponential moving average with smoothing
coefficient in the range 0.05 to 0.2; and wherein said user-state
model additionally maintains per-feature running mean and variance
computed via Welford's online algorithm.*

#### Dependent Claim 7 (NT-Xent contrastive encoder training)

*The method of Claim 1, wherein the representation encoder is
trained with a contrastive objective comprising a normalised
temperature-scaled cross-entropy (NT-Xent) loss computed over pairs
of augmented views of user sessions, wherein augmentation comprises
at least one of feature-channel dropout, Gaussian perturbation, and
temporal-shift operations.*

#### Dependent Claim 8 (Privacy-override routing)

*The method of Claim 1, further comprising: selecting, by a
multi-armed bandit router, a generation arm from a plurality of
generation arms including at least a local on-device arm and a
remote-service arm; and wherein said selection is deterministically
overridden to the local on-device arm when a sensitivity classifier
flags the input as containing content in one or more sensitive
topic categories, said override occurring after the bandit's
sampling step and not being configurable at runtime.*

#### Dependent Claim 9 (INT8 quantisation of the conditioning path)

*The method of Claim 1, wherein the conditioning projector and each
of the per-block cross-attention sub-layers are quantised to 8-bit
integer precision for deployment, and wherein said quantisation
preserves a KL divergence ratio of at least 1.5 between generator
output distributions under distinct adaptation-vector states relative
to a noise-equivalent perturbation.*

#### Dependent Claim 10 (Multi-modal extension)

*The method of Claim 1, wherein the feature vector of dimension D
comprises features derived from at least one input modality selected
from the group consisting of keystroke timing, linguistic content,
voice prosody, touch-pressure, gaze-duration, accelerometer signals,
heart-rate signals, and skin-conductance signals.*

---

## 5. Novelty & Non-Obviousness Arguments

### 5.1 Distinction from prompt-tuning and P-tuning

Prompt-tuning (Lester et al., 2021) and P-tuning (Liu et al., 2022)
introduce learnable *soft* prompts at the token-embedding level.
These mechanisms, although parameter-efficient, **consume token
positions in the attention sequence**. The present invention's
conditioning tokens are consumed only by dedicated cross-attention
sub-layers and occupy *no position* in the token sequence being
generated. This is a structural distinction with concrete operational
consequences: an edge-scale model with a 256-token context window
retains its full 256 tokens for conversation, rather than
sacrificing tokens to a continuous prompt.

### 5.2 Distinction from LoRA

Low-Rank Adaptation (Hu et al., 2022) adds low-rank deltas to
weight matrices of a frozen pretrained base model. LoRA requires
(a) a pretrained base, (b) per-task weight-delta storage, and
(c) loading of the correct delta at inference. None of these
requirements apply to the present invention. The invention does not
presuppose a pretrained base; it is compatible with both
from-scratch-trained and pretrained generators. No per-user weight
storage is required; the conditioning tensor is *computed per
forward pass* from live implicit signal. No delta-loading step is
required at inference.

### 5.3 Distinction from adapter layers

Adapter layers (Houlsby et al., 2019) introduce learnable bottleneck
modules that, once trained, are **static at inference**. The
invention's cross-attention sub-layers consume a **dynamic**
conditioning tensor recomputed at every forward pass as a function
of the current user state; two forward passes on the same input
with two different user-state tensors will produce different outputs,
a property which adapter layers do not exhibit at inference time
(barring a retrain of the adapter).

### 5.4 Distinction from prefix-tuning

Prefix-tuning (Li and Liang, 2021) adds learnable keys and values to
each layer's attention. It resembles the present invention most
closely of all prior mechanisms, but differs in two critical
respects:

1. The prefix in prefix-tuning is **learned once and held static**
   per task; the present invention's conditioning tensor is
   **computed at every forward pass** from an implicit-signal
   user-state embedding.
2. The prefix in prefix-tuning is **concatenated to the self-attention
   keys/values** of each layer; the present invention introduces a
   **dedicated cross-attention sub-layer** between self-attention
   and feed-forward, with its own independent query/key/value
   projection matrices per block.

### 5.5 Non-obviousness of the implicit-signal-to-conditioning-tensor pipeline

It is non-obvious to a person skilled in the art of transformer
personalisation that (i) keystroke dynamics, linguistic complexity,
and session rhythm can be compressed into a small fixed-dimensional
tensor and consumed by a transformer generator's cross-attention
path as a *per-forward-pass* conditioning signal, and (ii) that this
pipeline can operate within edge-device memory and latency
constraints without retraining for each user. The combination of
implicit-signal encoding, Welford-online user modelling, and
per-block dedicated cross-attention — as an integrated pipeline — is
not suggested by any single prior reference.

---

## 6. Drawings / Figures

### 6.1 Figure 1 — System overview

```
 +---------------------+     +----------------------+
 | IMPLICIT SIGNALS    |     | LINGUISTIC FEATURES  |
 |  (keystrokes, pauses)+---->+  (complexity, style) |
 +----------+----------+     +-----------+----------+
            |                            |
            v                            v
      +-----+---------------+   +--------+---------+
      | 32-dim InteractionFeatureVector  (4 groups)|
      +-------------------+------------------------+
                          |
                          v
             +------------+-------------+
             |  TCN Encoder (causal,    |
             |  dilated, contrastive)   |
             +------------+-------------+
                          |
                   64-dim z_user
                          |
                          v
             +------------+-------------+
             |  Three-timescale User    |
             |  Model (instant, session,|
             |  long-term; Welford)     |
             +------------+-------------+
                          |
                          v
             +------------+-------------+
             |  AdaptationController    |
             |  -> 8-dim adaptation vec |
             +------------+-------------+
                          |
              [v_adapt ; z_user] = 72-dim
                          |
                          v
             +------------+-------------+
             |  Conditioning Projector  |
             |  -> K × d tokens (K=4)   |
             +------------+-------------+
                          |
                          v
     +--------------------+--------------------+
     |  Conditioned Generator (transformer)    |
     |  block 1 -> block 2 -> ... -> block L   |
     |  each block has dedicated cross-attn    |
     +--------------------+--------------------+
                          |
                          v
                     Response token

 Caption: Fig. 1. System overview. The implicit-signal perception
 pipeline (top) produces a 32-dim feature vector, which the
 contrastively-trained TCN encoder maps to a 64-dim user-state
 embedding. Three-timescale running statistics yield an adaptation
 vector. The Conditioning Projector maps [adaptation; user-state] to
 K conditioning tokens consumed by each transformer block's dedicated
 cross-attention sub-layer.
```

### 6.2 Figure 2 — Cross-attention block (INVENTIVE)

```
     +--------------------------+
     |  Token sequence x        |  (shape: T × d)
     +------------+-------------+
                  |
          LayerNorm (ln_1)
                  |
                  v
     +------------+-------------+
     |  Self-Attention (causal) |
     +------------+-------------+
                  |
              residual add
                  |
                  v
          LayerNorm (ln_2)
                  |          Conditioning tokens C (shape: K × d)
                  |           (from Conditioning Projector)
                  v          /
     +------------+---------+------+
     |  CROSS-ATTENTION sub-layer  |   <-- dedicated, per-block,
     |   Q = ln_2(x),  K = V = C   |       computed from C at every
     +------------+----------------+       forward pass
                  |
              residual add
                  |
                  v
          LayerNorm (ln_3)
                  |
                  v
     +------------+-------------+
     |  Feed-Forward (MLP ×4)   |
     +------------+-------------+
                  |
              residual add
                  |
                  v
     +--------------------------+
     |  Token sequence x'       |
     +--------------------------+

 Caption: Fig. 2. A single Conditioned Transformer Block. Between
 the conventional self-attention and feed-forward sub-layers, a
 dedicated cross-attention sub-layer attends from the token
 sequence (query) to the K conditioning tokens C (keys and values).
 C is recomputed per forward pass, making the conditioning dynamic.
```

### 6.3 Figure 3 — Conditioning Projector

```
     [v_adapt (8)  ;  z_user (64)]   (concat, 72-dim)
                   |
                   v
        +----------+----------+
        |  Linear 72 -> 128   |
        +----------+----------+
                   |
                 GELU
                   |
                   v
        +----------+----------+
        |  Linear 128 -> K*d  |  (K=4, d=256  =>  1024)
        +----------+----------+
                   |
                   v
        +----------+----------+
        |  Reshape to K × d   |
        +----------+----------+
                   |
                   v
              C  (4 × 256)  -> consumed by every block

 Caption: Fig. 3. The Conditioning Projector: a two-layer MLP with
 GELU nonlinearity that maps the 72-dim concatenation of the 8-dim
 adaptation vector and the 64-dim user-state embedding to a K × d
 conditioning tensor. Parameter count is small (~25 k in the
 preferred embodiment, K=4, d=256).
```

### 6.4 Figure 4 — Three-timescale EMA

```
  Time  t-1       t        t+1       ...
   |     |        |         |
   z_t --+--------+---------+--- (instantaneous embedding)
                  |
                  v
   session EMA:  mu_S_t = (1 - 0.3) * mu_S_{t-1} + 0.3 * z_t
                  |          horizon: ~5-10 messages
                  v
   long-term EMA: mu_LT_t = (1 - 0.1) * mu_LT_{t-1} + 0.1 * z_t
                  |           horizon: ~30 sessions
                  v
   deviation z-scores:  z_i = (x_i - mu_i_LT) / (sigma_i_LT + eps)
       (per-feature, via Welford online running mean/variance)

 Caption: Fig. 4. Three-timescale user model. The instant embedding
 z_t is folded into a session-timescale EMA (alpha_S = 0.3) and a
 long-term EMA (alpha_LT = 0.1). Per-feature running mean and
 variance, computed via Welford's online algorithm (Welford, 1962),
 yield z-scores used as deviation features within the 32-dim
 InteractionFeatureVector.
```

---

## 7. Industrial Applicability

The invention has direct applicability in the following commercial
domains. The technical feasibility of on-device deployment at the
referenced edge envelopes is supported by the inventor's prototype,
in which the combined encoder and conditioned generator occupy
approximately 7 MB INT8, with extrapolated per-turn latency in the
50–80 ms range on flagship mobile-class neural processing units.

1. **On-device AI companions and interactive assistants** operating
   within the memory and power envelopes of consumer mobile devices
   (flagship and mid-tier smartphones), wearable devices, and
   dedicated companion hardware such as the **Smart Hanhan**
   form-factor companion device (background-of-the-art reference
   only; no affiliation claimed).

2. **Accessibility-adaptive user interfaces** in which the adaptation
   vector drives simplification behaviours (shorter sentences, reduced
   vocabulary complexity, explicit structure) in response to
   implicit indicators of motor or cognitive difficulty detected
   from keystroke patterns.

3. **Smart wearables** (including AI-glasses-class devices and
   wristband companions) in which the encoder may run on the
   wearable and the conditioning tokens propagated over a
   low-bandwidth link to a paired host device for generation.

4. **Internet-of-things companions and ambient assistants** deployed
   in home, vehicle, and retail environments where network
   connectivity is intermittent or unavailable.

5. **On-device conversational agents running on mobile NPU
   substrates** comparable to the **Kirin NPU** family or other
   DSA/NPU edge accelerators (background-of-the-art reference only;
   no affiliation claimed).

---

## Appendix A — Sample Pseudocode

The following pseudocode is illustrative of the invention and is not
claim-limiting. It corresponds to the preferred embodiment with
*D* = 32, *E* = 64, *A* = 8, *K* = 4, *d* = 256, *L* = 4 transformer
blocks, and *k* = 3 causal convolution kernel size.

```python
# ---------- Representation encoder (causal TCN) ----------
def causal_conv1d(x, weight, bias, dilation):
    k = weight.shape[-1]
    x = left_pad(x, (k - 1) * dilation)
    return conv1d(x, weight, bias, dilation=dilation)

def tcn_encode(feature_vectors):
    # feature_vectors: [B, T, 32]
    h = linear(feature_vectors, W_in)            # [B, T, 64]
    for layer_idx in range(L):                   # L = 4
        d_l = 2 ** layer_idx                     # dilations: 1, 2, 4, 8
        h_new = causal_conv1d(h, W_conv[layer_idx], b_conv[layer_idx], d_l)
        h_new = gelu(layernorm(h_new))
        h = h + dropout(h_new)                   # residual
    h = h.mean(dim=1)                            # global average pool
    z = linear(h, W_out)                         # [B, 64]
    return z / (z.norm(dim=-1, keepdim=True) + 1e-6)

# ---------- Three-timescale user-state update ----------
def update_user_state(state, z_t, alpha_s=0.3, alpha_lt=0.1):
    state.instant = z_t
    state.session_ema = (1 - alpha_s) * state.session_ema + alpha_s * z_t
    state.lt_ema     = (1 - alpha_lt) * state.lt_ema    + alpha_lt * z_t
    # Welford running stats over the 32 raw features
    for i in range(32):
        n = state.n[i] + 1
        delta = state.feature_raw[i] - state.mean[i]
        state.mean[i] += delta / n
        delta2 = state.feature_raw[i] - state.mean[i]
        state.M2[i]   += delta * delta2
        state.var[i]   = state.M2[i] / max(1, n - 1)
        state.n[i]     = n

# ---------- Conditioning Projector ----------
def conditioning_projector(v_adapt, z_user):
    x = concat([v_adapt, z_user], dim=-1)   # [B, 72]
    h = gelu(linear(x, W_cp1, b_cp1))       # [B, 128]
    y = linear(h, W_cp2, b_cp2)             # [B, K * d] = [B, 1024]
    return reshape(y, (B, K, d))            # [B, 4, 256]

# ---------- Conditioned Transformer Block ----------
def adaptive_transformer_block(x, C, causal_mask, block_params):
    # x: [B, T, d]; C: [B, K, d]
    # Pre-LN self-attention
    h = layernorm(x, block_params.ln1)
    h = multihead_self_attention(h, causal_mask, block_params.self_attn)
    x = x + dropout(h)

    # Pre-LN CROSS-ATTENTION to conditioning tokens (INVENTIVE)
    h = layernorm(x, block_params.ln2)
    h = multihead_cross_attention(query=h, key=C, value=C,
                                  params=block_params.cross_attn)
    x = x + dropout(h)

    # Pre-LN feed-forward
    h = layernorm(x, block_params.ln3)
    h = feed_forward(h, block_params.ff)
    x = x + dropout(h)
    return x

# ---------- End-to-end forward pass ----------
def forward(tokens, causal_mask, v_adapt, z_user, model):
    C = conditioning_projector(v_adapt, z_user)        # [B, K, d]
    x = embed(tokens) + positional(tokens.shape[1])
    for block in model.blocks:                         # L = 4 blocks
        x = adaptive_transformer_block(x, C, causal_mask, block)
    logits = linear(layernorm(x, model.ln_final),
                    model.embed.weight.T)              # weight-tied
    return logits
```

---

## Notes for the patent attorney

The following notes are provided to assist formal drafting and are
not themselves claim-limiting. They reflect the inventor's best
understanding as of the priority date.

1. **This document is a provisional invention disclosure.** It is
   drafted by the inventor for internal review. It is not a filed
   application. A formal GB or PCT filing will require revision by
   counsel, addition of formal claim brackets, compliance with the
   relevant office's formality rules, and decisions about which
   jurisdictions to pursue.
2. **Huawei product references** — Smart Hanhan, AI Glasses,
   HarmonyOS, HMAF, Kirin NPU, MindSpore Lite, and similar — appear
   exclusively in §2 (Background of the Invention) and §7
   (Industrial Applicability) as background-of-the-art context for
   the patent attorney. They must be treated as such. **No
   representation is made about Huawei's filing intent, endorsement,
   or participation, nor should any be inferred from this document.**
   References to commercial on-device edge platforms are included to
   illustrate the industrial applicability of the invention in the
   edge-AI domain and to frame the constraints the invention
   addresses (memory ≤ 64 MB, NPU TOPS ≤ 2.0, latency ≤ 100 ms) —
   not to make product-level representations.
3. **Claim breadth and fallback language.** The independent claims
   are drafted at a level of generality the inventor believes is
   defensible — the core inventive concept is the dedicated
   per-layer cross-attention consuming a dynamically-computed
   conditioning tensor derived from implicit behavioural signals —
   but counsel may wish to draft narrower fallback claims (e.g.
   specifying concrete dimensions, specifying keystroke-dynamics as
   the signal source, specifying a TCN encoder) to survive prior-art
   rejection. Dependent claims 4–10 are provided as a starting
   point; additional dependent claims covering (a) variance-weighted
   cross-device merging, (b) federated training of the conditioning
   projector, and (c) auxiliary conditioning-consistency losses may
   be warranted after a full prior-art search.
4. **Prior-art search recommended.** The inventor has performed an
   informal survey of prompt-tuning, P-tuning, prefix-tuning, LoRA,
   and adapter-layer literature, but has not performed a formal
   patent-office search. In particular, counsel should verify
   novelty against any existing filings by large-model providers on
   dynamic cross-attention-based personalisation.
5. **Drawings.** The ASCII figures in §6 are placeholders intended
   to convey the architecture unambiguously. Counsel may wish to
   commission formal figures per the relevant office's drawing
   standards for the eventual full application.
6. **Inventor disclosure obligations.** The inventor notes that a
   research paper and prototype code (PyTorch, Python) describing
   the invention are available and may be relevant to disclosure
   timing and public-availability considerations. Counsel should
   advise on any implications for novelty before public dissemination
   of the paper or code.
7. **Treatment of synthetic training data.** The reduction-to-practice
   section of a full application may benefit from explicit framing
   of the synthetic-training-data evidence as a proof-of-concept
   measurement rather than as a generalisation claim; this is
   consistent with the inventor's own framing in the accompanying
   research paper.
8. **Internal review only.** This disclosure is intended for internal
   review by the inventor, his counsel, and — if applicable — any
   corporate assignee's legal and IP review teams. It is not for
   public distribution in its current form.

---

*End of provisional invention disclosure draft. Version 1.0,
inventor-authored, pending counsel review.*
