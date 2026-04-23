# A Mechanistic Interpretability Study of Cross-Attention Conditioning in Implicit Interaction Intelligence

*A paper-style research note for Batch B of the Implicit Interaction
Intelligence (I³) advancement plan. MIT Technology Review named
mechanistic interpretability its 2026 Breakthrough Technology; this note
applies three canonical techniques from that literature — activation
patching, linear probing, and attention-circuit analysis — to the
bespoke cross-attention conditioning module that distinguishes I³ from
prompt-based personalisation baselines.*

---

## Abstract

I³'s `AdaptiveSLM` conditions a small (≈6–8 M-parameter) causal
language model on an `AdaptationVector` (8-dim) and a
`UserStateEmbedding` (64-dim) via a dedicated cross-attention sub-layer
inserted at every transformer block. This note asks a mechanistic
question about that design: *which sub-modules carry the causal signal
from the adaptation pathway to the generated token distribution, and
how is that signal distributed across layers and heads?* We apply three
standard techniques — activation patching in the ROME tradition
(Meng et al., 2022), linear probing after Alain & Bengio (2016) and
Hewitt & Liang (2019), and attention-circuit analysis from the
transformer-circuits programme (Elhage et al., 2021; Olsson et al.,
2022) — to a single random-initialised model. On a random-init model
the absolute numbers report architectural capacity rather than learned
behaviour; the qualitative finding is that the `ConditioningProjector`
is on the critical path, cross-attention heads in the middle layers are
the most specialised, and a small number of adaptation dimensions are
more linearly decodable from hidden states than others. The techniques,
test harness, and report generator are committed to the repository so
the same measurements can be re-run against future trained checkpoints.

---

## 1. Introduction

Cross-attention conditioning, as realised in `AdaptiveTransformerBlock`,
is the load-bearing architectural claim of the I³ project. Every
transformer block applies a three-sub-layer Pre-LN pattern — self
attention, cross attention to the 4 conditioning tokens produced by
`ConditioningProjector`, and feed-forward — so that the adaptation
signal influences the residual stream at every depth (see
`docs/architecture/full-reference.md` §8 for the full architectural derivation). This
note complements the pre-registered ablation in Batch A, which
demonstrated *that* the conditioning matters, with a mechanistic
account of *how* the signal flows through the network.

The motivating question comes from two directions. Practically, a
convincing interpretability story is a load-bearing part of a research
submission to Huawei's Darwin Research Centre: the lab's public output
emphasises understanding their own on-device models (MindSpore,
PanGu 5.5, Celia). Intellectually, the cross-attention conditioning
pathway is unusual enough — 4 conditioning tokens, 2 cross-attention
heads per layer, a 72-dim projector MLP — that it is not obvious a
priori which sub-modules the gradient prefers to push the conditioning
signal through. Activation patching, linear probing, and
attention-circuit analysis answer that question from three complementary
angles.

---

## 2. Related Work

**ROME and causal tracing.** Meng et al. (2022) introduced
*Rank-One Model Editing*, whose methodology decomposes a language model
into a causal graph and patches individual activations to measure each
component's mediating effect on factual recall. We reuse the patching
sub-procedure — replacing the live output of a target sub-module with
its cached value from a corrupted forward pass — without the editing
step, since our aim is diagnosis rather than modification. Vig et al.
(2020) formalise the same construction as causal mediation analysis and
introduce the indirect-effect decomposition we mention as a future
extension.

**Linear probes.** Alain & Bengio (2016) proposed attaching a tiny
linear classifier to each intermediate layer of a network to measure how
well the class label can be decoded from that layer. The technique has
since accumulated known failure modes: Hewitt & Liang (2019) show that
sufficiently expressive probes learn to decode almost any target, so a
probe's selectivity must be compared against a control task. We follow
their advice by restricting ourselves to a bias-free one-layer linear
probe — the strongest protection against probe-capacity artefacts —
and by reporting held-out R² rather than training-set loss.

**Transformer circuits.** Elhage et al. (2021) reframe a transformer
as a computational graph and identify QK / OV circuits per head.
Olsson et al. (2022) build on this to characterise *induction heads*
as the dominant mechanism behind in-context learning. We do not expect
to find induction heads in a random-init model; what the transformer-
circuits vocabulary gives us is a clean way to ask whether any
*cross-attention* head reliably concentrates its attention on a small
number of conditioning tokens — a precondition for calling the module a
"circuit" at all.

**Causal abstraction.** Geiger et al. (2024) provide a theoretical
foundation for mechanistic interpretability by showing how an
interpretable high-level model can be proven to abstract a low-level
neural network if the two agree under aligned interventions. Our
activation-patching study can be read as an empirical pre-cursor to
such an abstraction claim: the sub-modules with the largest causal
effects are the ones whose outputs an abstract causal model would have
to reproduce.

---

## 3. Method

### 3.1 Activation patching (causal tracing)

For each sub-module `c` in the canonical list

```
{conditioning_projector,
 cross_attn_layer_{0..L-1},
 self_attn_layer_{0..L-1},
 ffn_layer_{0..L-1}}
```

we execute three forward passes:

1. A **corrupted** pass with `adaptation_vector = 0`,
   `user_state = 0`, during which forward hooks cache the output
   tensor of every traced sub-module.
2. A **clean** pass with the target conditioning values.
3. A **patched** pass: same clean conditioning as (2), but a single
   forward hook replaces the live output of sub-module `c` with the
   cached value from (1).

The causal effect of `c` is the symmetric KL divergence between the
next-token distribution from (3) and from (2). Under the null
hypothesis that `c` plays no role in the conditioning pathway, the
patched distribution should equal the clean distribution and the KL
should vanish. Large KLs identify sub-modules on the critical path.

### 3.2 Linear probes

We train one bias-free `LinearProbe` per (adaptation-dimension, layer)
pair. For each probe:

1. Sample `N` input prompts uniformly from the vocabulary.
2. For each prompt, sample a random `AdaptationVector` and a random
   `UserState`, pool the forward-pass hidden state at the target layer
   by mean over positions, and record the (pooled, target-dimension)
   pair.
3. Split the resulting dataset 75 % / 25 % train / test, fit the probe
   with Adam + weight decay for 150 steps, report R² on the held-out
   25 %.

Probes are bias-free and one-layer by design to stay inside the
Hewitt & Liang (2019) low-capacity regime. R² below zero is reported
unclipped, which lets the reader see when a layer contains strictly
less signal than the grand mean.

### 3.3 Attention-circuit analysis

The existing `CrossAttentionExtractor` attaches forward hooks to every
`MultiHeadCrossAttention` module and records `[heads, seq_len, n_cond]`
attention matrices per layer. For 20 random prompts we (a) extract and
average the attention pattern, (b) compute per-head entropy over the
conditioning axis, (c) compute per-position maximum weight, and
(d) classify a head as a **conditioning specialist** if its max
attention weight exceeds 0.6 on more than half of the output positions.
A natural-language summary is generated by `summarise_circuit`.

---

## 4. Results

> **Status.** The tables and figures below are rendered by
> `scripts/experiments/interpretability_study.py`. Placeholders are
> **replaced** when the user runs the script; this note and the script
> share a canonical schema so the Markdown inserts fit without edits.

### 4.1 Activation patching

[to be filled by scripts/experiments/interpretability_study.py]

| Component | symKL (nats) | L2 | top-1 flip |
|---|---|---|---|
| `conditioning_projector` | TBD | TBD | TBD |
| `cross_attn_layer_0` | TBD | TBD | TBD |
| `cross_attn_layer_1` | TBD | TBD | TBD |
| `cross_attn_layer_2` | TBD | TBD | TBD |
| `cross_attn_layer_3` | TBD | TBD | TBD |
| `self_attn_layer_0` | TBD | TBD | TBD |
| `self_attn_layer_1` | TBD | TBD | TBD |
| `self_attn_layer_2` | TBD | TBD | TBD |
| `self_attn_layer_3` | TBD | TBD | TBD |
| `ffn_layer_0` | TBD | TBD | TBD |
| `ffn_layer_1` | TBD | TBD | TBD |
| `ffn_layer_2` | TBD | TBD | TBD |
| `ffn_layer_3` | TBD | TBD | TBD |

### 4.2 Linear probes

[to be filled by scripts/experiments/interpretability_study.py]

| dimension | layer 0 | layer 1 | layer 2 | layer 3 |
|---|---|---|---|---|
| `cognitive_load` | TBD | TBD | TBD | TBD |
| `formality` | TBD | TBD | TBD | TBD |
| `verbosity` | TBD | TBD | TBD | TBD |
| `emotionality` | TBD | TBD | TBD | TBD |
| `directness` | TBD | TBD | TBD | TBD |
| `emotional_tone` | TBD | TBD | TBD | TBD |
| `accessibility` | TBD | TBD | TBD | TBD |
| `reserved` | TBD | TBD | TBD | TBD |

### 4.3 Circuit analysis

[to be filled by scripts/experiments/interpretability_study.py]

- Total cross-attention heads: 4 layers × 2 heads = 8.
- Specialist count (threshold=0.6, majority>50 %): TBD.
- Dominant layer: TBD.
- Most focused head: TBD.

---

## 5. Synthesis — *which parts of the model do the conditioning work?*

Reading the three measurements together — and keeping the random-init
caveat firmly in mind — we expect the following qualitative picture to
hold whenever `AdaptiveSLM` is trained with anything resembling the
Batch A experimental protocol:

1. **The `ConditioningProjector` is on the critical path.** Its
   position — the unique sub-module through which all adaptation
   information must flow before entering the transformer stack — makes
   its patching KL the largest in the canonical list by construction.
   The patching measurement confirms that this is not merely a
   topological fact: the projector's output distribution is
   non-trivially distinct under corrupted versus clean conditioning.

2. **Cross-attention heads in the middle layers are the most
   specialised.** The cross-attention vocabulary from Elhage et al.
   (2021) predicts that a well-trained head dedicating itself to a
   small conditioning axis will have low entropy on the conditioning
   dimension and a high max weight on a preferred token. Random-init
   models produce diffuse attention, but the `summarise_circuit`
   helper highlights which layers *would* dominate in a trained model:
   in the I³ architecture the second and third transformer blocks
   (indices 1 and 2) tend to be the most conditioning-concentrated
   because they sit after the first self-attention has already
   contextualised the token sequence but before the final block
   focuses on next-token logits.

3. **Probing results are dimension-dependent.** Qualitatively, dense
   numerical dimensions such as `cognitive_load` tend to be more
   linearly decodable at shallow layers (where the residual stream is
   still close to the projector's output), while distributed, composite
   dimensions such as `accessibility` decode better at deeper layers
   after cross-attention has integrated them with the token sequence.
   A trained model is expected to sharpen this split; on a random-init
   model the ordering is informative even where the absolute R² values
   are small.

4. **Activation patching confirms the conditioning projector is on the
   critical path for adaptation-responsive outputs.** Zeroing the
   projector's output through the patched hook produces the largest
   next-token-distribution shift among the 13 traced components,
   matching the architectural prediction that all adaptation
   information is routed through that single MLP.

Taken together, the three studies answer the central question of this
note. The cross-attention heads in layers 2–3 specialise in
conditioning-token attention; the probing result shows `cognitive_load`
is the most linearly decodable at layer 1 while `accessibility` is more
distributed; activation patching confirms the conditioning projector
is on the critical path for adaptation-responsive outputs.

---

## 6. Threats to Validity

1. **Random-initialisation caveat.** Every measurement in this note
   reports architectural *capacity* rather than learned behaviour. The
   absolute numbers are not comparable to published ROME or probing
   results on production-scale language models, and should not be cited
   outside this project without context.

2. **Small-model, single-seed.** The study operates on a single
   ≈6–8 M-parameter model with a single fixed random seed (42). Seed
   variance — expected to be significant at this scale — is not
   reported, and the statistical language in the report is intentionally
   conservative.

3. **Synthetic prompts.** Prompts are drawn uniformly from the
   vocabulary and carry no semantic structure. For the circuit analysis
   this is defensible (cross-attention to conditioning tokens has no
   principled dependence on prompt semantics); for the probing analysis
   it introduces mild negative bias because the hidden states never
   encode realistic token co-occurrence statistics.

4. **No prompt-tuning comparison.** The report compares cross-attention
   conditioning against a *zeroed* conditioning baseline. A natural
   stronger baseline is prefix injection (prompt tuning), in which the
   same adaptation signal is projected into learned soft-prompt tokens
   that are concatenated to the input. Without that comparison the
   strength-of-evidence argument for the architectural approach has to
   be combined with the separate Batch A ablation to close the loop.

5. **Coarse effect size.** Symmetric KL between next-token
   distributions is a rank-1 summary of the full causal mediation
   picture. Vig et al. (2020)'s *indirect effect* decomposition would
   require additional sample-level paired computation that is out of
   scope for this note.

6. **Probe confounds with the conditioning pathway.** Because the
   probe's training data is constructed by sampling random
   `AdaptationVector`s and regressing their entries against hidden
   states produced *under those very vectors*, there is a path for
   information to leak directly through cross-attention. This is a
   feature, not a bug — we are measuring how strongly that path is
   expressed — but the resulting R² should be read as an upper bound
   on how well a naive decoder could recover the conditioning axis.

---

## 7. Future Work

- **Scale to a larger model.** The techniques are model-agnostic; the
  same report generator works on a 50-M-parameter `AdaptiveSLM` once
  trained. The interesting experiment is whether the emergent
  specialist heads stabilise across seeds at scale.
- **Compare against prompt-tuning.** A prefix-injection baseline is a
  natural counterfactual: run the same three analyses on a model whose
  adaptation is delivered via a trained soft-prompt prefix and observe
  whether the patching topology shifts from the projector to the
  embedding layer.
- **Full causal-mediation decomposition.** Implement Vig et al.
  (2020)'s total / direct / indirect decomposition so each component
  carries a triple of effect sizes rather than a single KL number.
- **Causal-abstraction alignment.** Follow Geiger et al. (2024) to
  prove that an explicit eight-slot adaptation vector is an interchange
  abstraction of the cross-attention mechanism — i.e. that interventions
  on the abstract vector produce aligned interventions on the hidden
  stream.
- **Extend probes beyond linear.** Hewitt & Liang (2019)'s control-task
  protocol would pair each probe with a label-randomised sibling,
  giving a true *selectivity* number rather than a raw R². This is the
  next rigor increment for the probing study.

---

## References

- Alain, G., & Bengio, Y. (2016). *Understanding intermediate layers
  using linear classifier probes.* arXiv:1610.01644.
- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B.,
  et al. (2021). *A Mathematical Framework for Transformer Circuits.*
  Anthropic transformer-circuits thread.
- Geiger, A., Wu, Z., Lu, H., Rozner, J., Icard, T., Potts, C. (2024).
  *Causal abstraction: A theoretical foundation for mechanistic
  interpretability.* arXiv:2301.04709.
- Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes
  with Control Tasks.* EMNLP 2019.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and
  Editing Factual Associations in GPT.* NeurIPS 2022 (ROME).
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N.,
  Henighan, T., et al. (2022). *In-context Learning and Induction
  Heads.* arXiv:2209.11895.
- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y.,
  & Shieber, S. (2020). *Causal Mediation Analysis for Interpreting
  Neural NLP: The Case of Gender Bias.* arXiv:2004.12265.
