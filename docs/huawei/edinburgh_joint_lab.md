# I³ and the Huawei–Edinburgh Joint Lab

> **Thesis.** The Huawei–Edinburgh Joint Lab pursues personalisation from
> sparse signals and efficient on-device learning. I³'s cross-attention
> conditioning architecture is not merely compatible with that research
> direction — it is a **structural alternative to prompt-based
> personalisation** that directly extends the Lab's line of work, offering
> a path to the same goals that a prompt-engineering approach cannot
> reach at edge scale.

---

## 1. What the Joint Lab is doing

The Huawei–Edinburgh Joint Lab has focused on the intersection of
**machine learning for language**, **human-centred AI**, and **efficient
on-device deployment**. Publicly known research threads include:

- Dialogue modelling and controllable response generation.
- Personalisation of conversational agents under data-scarcity constraints.
- Continual and on-device adaptation without catastrophic forgetting.
- Efficient training and inference for small language models.

The through-line is the Lab's bet that the next generation of useful
language systems will **not** be lumbered by full-size LLMs at inference
time and will **not** learn user preferences from explicit profiles
alone. They will learn efficiently, from sparse implicit signals, on
the devices people already own.

I³'s design was shaped by this bet.

---

## 2. Where prompt-based personalisation runs out

The industry-standard path to personalisation is **prompt injection**:

> "You are talking to a user who is under high cognitive load and prefers
> terse, informal responses. Simplify your language."

This works, up to a point. Its ceiling is visible at edge scale. Three
failure modes:

1. **Attention dilution.** As the conversation grows, the fixed system
   prompt's influence on the model's attention diminishes. By turn 10 the
   personalisation instruction is competing with nine turns of dialogue for
   attention mass. The behaviour drifts back to the pre-training prior.

2. **Token cost.** A personalisation prompt costs tokens that could be
   spent on actual conversation. On a 256-token context SLM, a 50-token
   personalisation preamble is 20 % of the budget. That budget is already
   tight.

3. **Capacity ceiling.** A small language model cannot reliably follow
   subtle, nuanced system-prompt instructions. The paradox: the smaller
   the model (and thus the more valuable personalisation is for quality),
   the less able the model is to extract personalisation from prompts.

The Joint Lab's line of research — "personalisation from sparse signals"
— implicitly acknowledges these ceilings. The question becomes: *where
else can the personalisation signal enter the model?*

---

## 3. I³'s answer: architectural conditioning

I³ answers the question at a different layer of the stack entirely. It
conditions generation **architecturally**, **at every layer**, **at every
token position**.

The mechanism has two halves:

### 3.1 The ConditioningProjector

A 2-layer MLP that maps the 72-dim concatenation of
`[AdaptationVector (8); UserStateEmbedding (64)]` into a
`4 × d_model = 4 × 256` tensor — **4 conditioning tokens**.

These 4 tokens are *not* prepended to the prompt. They live in a separate
tensor that every transformer layer attends to via cross-attention.

### 3.2 Per-layer cross-attention

Every `AdaptiveTransformerBlock` has three sub-layers: self-attention,
**cross-attention to the 4 conditioning tokens**, and feed-forward.

```
for every layer ℓ ∈ {1..L}:
  x = x + self_attn(x)                            # self, causal
  x = x + cross_attn(q=x, k=C, v=C)               # cross, user state
  x = x + ff(x)                                   # ff
```

Because the conditioning tokens are only 4, the cost is
$O(T \cdot 4 \cdot d)$ per head per layer — a rounding error.

Because the same `AdaptationVector` is recomputed every forward pass, the
same trained model adapts to any user with no retraining.

Because the cross-attention weights are trained end-to-end to minimise the
generation loss, the model **learns to use the conditioning**. Unlike a
prompt, it cannot be ignored.

### 3.3 Why this is novel relative to prompt-based personalisation

| Property | Prompt-based | I³ architectural |
|:---|:---:|:---:|
| Gradient path from user signal to every token | Indirect (through attention from prompt) | Direct (cross-attention at every layer) |
| Token budget cost | O(prompt length) | Zero (separate tensor) |
| Attention dilution with conversation length | Yes | No |
| Can be retrofitted to pre-trained model | Yes | No — needs training |
| Scales to small (sub-10 M) models | Poorly | Well — relatively *more* valuable at small scale |
| Per-request user state change | Possible via re-prompting | Native (conditioning is re-projected each call) |

The last row is the most interesting for the Joint Lab's interests.
**I³'s conditioning changes *per forward pass*.** A continual-learning or
online-learning signal has a first-class place to enter the model —
directly, every step, without retraining.

---

## 4. Relationship to specific Joint Lab themes

### 4.1 Personalisation from sparse signals

The Lab's published work on personalisation under scarcity emphasises
**extracting maximum signal from limited user data**. I³'s philosophy is
kindred: never ask the user a survey question; extract everything from
the interaction itself.

The concrete mapping:

- **Joint Lab: few-shot personalisation** — I³: *zero-shot* personalisation.
  The user never provides examples; the user-state embedding is computed
  from the very first session's keystroke dynamics.
- **Joint Lab: user simulators for offline evaluation** — I³: generates
  persona-parameterised synthetic users via the
  `training/synthetic_data.py` tooling, which produces a calibrated
  distribution of personas for offline A/B of adaptation policies.
- **Joint Lab: low-resource continual learning** — I³: the three-
  timescale EMA + Welford user model is a literal online-learning
  mechanism whose memory footprint is bounded and whose update cost is
  O(1) per message.

### 4.2 Continual adaptation without forgetting

The catastrophic-forgetting problem in continual learning usually assumes
that model weights themselves change. I³ sidesteps this by **not changing
model weights**: the user-specific signal enters through the conditioning
projector's inputs, not through the model's weights. The model is trained
once on a diverse synthetic persona distribution; each user's adaptation
is parameter-free at inference.

This is a specific design decision: **personalisation is not a weight
update, it is an input tensor.** The implications:

- Zero risk of forgetting — the weights never move post-training.
- Deployment symmetry — every device runs the same weights.
- Audit-friendly — the personalisation signal is explicit, inspectable,
  and bounded.

### 4.3 Efficient small language models

The Joint Lab's interest in SLMs lines up with I³'s 6.3 M-parameter
AdaptiveSLM. Concrete properties:

- Word-level tokenizer (8 192 vocabulary) — simpler, more debuggable
  than BPE at our scale.
- Weight-tied output projection — zero extra parameter cost for decoding.
- Sinusoidal positional encoding — zero learned parameters.
- 4-layer transformer with 4-head attention, d_model = 256.

At INT8, the whole model is **7 MB**. At torchao INT4 weight-only, it is
**2.6 MB**. Either regime lives comfortably inside the Joint Lab's
efficient-SLM charter.

---

## 5. A concrete research collaboration proposal

A hypothetical collaboration with the Joint Lab has at least three
near-term publishable questions.

### 5.1 Question 1: cross-attention conditioning vs prompt-based personalisation, controlled comparison

A head-to-head comparison on a fixed benchmark (DailyDialog, EmpatheticDialogues):

- Arm A: baseline 6.3 M SLM with a personalisation prompt (50 tokens).
- Arm B: I³'s 6.3 M SLM with 4 conditioning tokens at every layer.
- Metrics: persona-alignment score, KL divergence of next-token
  distributions under varying user states, conversation-length
  persistence of personalisation (do both arms still sound personalised
  at turn 15?).

Publishable either way. Our hypothesis is Arm B wins on all three, with
the gap growing with conversation length.

### 5.2 Question 2: the conditioning-sensitivity curve

The conditioning-sensitivity test (`evaluate.py`) measures KL divergence
between next-token distributions under different `AdaptationVectors`.
Systematically sweeping this curve — varying one dimension at a time,
varying combinations, varying magnitudes — produces a **conditioning-
response surface** that characterises how the model uses the
conditioning. This is a novel evaluation artefact for any conditioned
generation model, not just I³.

### 5.3 Question 3: federated training of the conditioning projector

A natural extension: the SLM weights are trained centrally, but the
**conditioning projector** — the small MLP that maps the user-state
embedding into conditioning tokens — is federatively fine-tuned
per-user. Because the projector is ~25 K parameters, per-user fine-tuning
is cheap, and because the SLM weights are fixed, there is no cross-user
leakage through weight updates. Differential-privacy guarantees are
imposed on the projector updates only.

This is a tight fit for the Joint Lab's interests in **private federated
personalisation**.

---

## 6. Where cross-attention conditioning is *not* the right answer

Intellectual honesty section. Cross-attention conditioning has costs:

- **Requires training from scratch.** You cannot retrofit cross-attention
  conditioning to a pre-trained HuggingFace SLM without retraining from a
  random init. For teams who want to bolt personalisation onto an
  off-the-shelf model, prompt-based remains the expedient path.
- **Small compute overhead.** ~5 % parameter overhead over a bare
  transformer. Negligible, but non-zero.
- **More engineering surface.** The conditioning projector, the per-layer
  cross-attention, the adaptation pipeline, the user model — more moving
  parts than a pure prompt-injection system.

The trade is worth it at edge scale and with control over the training
run. Prompt injection is the right answer when you have neither.

---

## 7. The broader context: implicit-first HMI

The Joint Lab sits inside a wider Huawei commitment to **human-machine
interaction as a research discipline**. The London HMI Lab's charter
explicitly calls out:

> "develop custom machine learning models including traditional ML
> pipelines, small language models (SLMs), and solutions built on top of
> foundational models"[^hmi]

[^hmi]: Huawei HMI Lab (London) published research scope, 2025.

I³ hits all three categories simultaneously:

- **Traditional ML pipelines** — the 32-dim feature extractor, the Welford
  statistics, the Thompson sampling router are textbook classical ML.
- **SLMs** — the 6.3 M-parameter AdaptiveSLM is a small language model by
  any honest definition.
- **Solutions built on top of foundational models** — the cloud path
  wraps Anthropic's Claude with dynamic system prompts derived from the
  `AdaptationVector`.

A project that hits every category in the HMI Lab's scope is the kind of
proof-of-concept the Joint Lab can actually absorb.

---

## 8. What I³ would gain from the Joint Lab

This is a symmetric relationship. What I³ would gain:

1. **Evaluation at scale.** Joint-Lab-mediated access to anonymised
   dialogue corpora with verifiable persona labels would let us validate
   the conditioning-sensitivity claims beyond the synthetic-persona
   regime we have today.
2. **Baseline models.** Access to Huawei's on-device SLM checkpoints as
   the "frontier" comparison point.
3. **Privacy-preserving evaluation infrastructure.** Federated evaluation
   of personalisation quality without raw-data centralisation is a
   research problem in itself, one the Joint Lab is well-placed to solve.

---

## 9. Open research questions

These are the questions that would motivate a research visit, a
collaboration, or a PhD:

1. **What is the optimal number of conditioning tokens?** We use 4.
   Is 2 sufficient? Does 8 buy anything?
2. **Should conditioning tokens be shared across layers or per-layer?**
   Currently shared. Per-layer projections might buy expressiveness.
3. **Can the user-state embedding be *self-supervised* to predict itself
   from future keystrokes?** A predictive-coding objective on top of
   NT-Xent might tighten the embedding's use of the deviation signal.
4. **How stable is the cross-user transfer of the cross-attention
   weights?** If we train on persona A's conditioning tokens, does the
   model generalise to persona B's, or do we need diverse-persona
   curriculum training?
5. **Is there an information-theoretic bound on the bits a cross-
   attention conditioner can usefully pass to generation?** This would be
   a genuinely novel theoretical result.

---

## 10. How this shows up in the I³ codebase

- **Cross-attention conditioning:** [`i3/slm/cross_attention.py`](../../i3/slm/cross_attention.py),
  [`i3/slm/transformer.py`](../../i3/slm/transformer.py).
- **Conditioning projector:** the MLP inside `i3/slm/model.py` (called
  `ConditioningProjector`).
- **Conditioning-sensitivity test:** the evaluation script in
  [`training/evaluate.py`](../../training/evaluate.py) (or equivalent).
- **Synthetic personas:** [`training/synthetic_data.py`](../../training/generate_synthetic.py).
- **Adaptation pipeline:** [`i3/adaptation/`](../../i3/adaptation/).

---

*Next: [Smart Hanhan — encoder-only deployment for IoT companions](./smart_hanhan.md).*
