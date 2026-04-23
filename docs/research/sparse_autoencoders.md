# Sparse Autoencoders for Cross-Attention-Conditioning Interpretability

*A paper-style research note for Batch G3 of the Implicit Interaction
Intelligence (I³) advancement plan (v3). MIT Technology Review named
mechanistic interpretability its 2026 Breakthrough Technology, and
Anthropic's 2023–2024 sparse-autoencoder programme is the technique at
the heart of that citation. This note applies the same machinery to the
bespoke cross-attention conditioning pathway that distinguishes I³ from
prompt-based personalisation baselines.*

---

## Abstract

We train dictionary-learning sparse autoencoders (SAEs) on the residual
stream of each cross-attention block in I³'s `AdaptiveSLM`, a
≈6–8 M-parameter causal language model conditioned on an eight-dimensional
`AdaptationVector` (cognitive load, formality, verbosity, emotionality,
directness, emotional tone, accessibility, reserved) and a
64-dimensional `UserStateEmbedding`. For every one of the four cross-
attention sub-layers we train a per-layer SAE with an overcomplete basis
(`d_dict = 8·d_model`) following Bricken et al. (2023), enforce the
decoder-column unit-norm constraint after every optimiser step, and then
correlate each of the `d_dict` learned features with each of the eight
adaptation dimensions. Features whose Pearson correlation with a single
dimension exceeds a 0.7 threshold are labelled *monosemantic* and become
the basic units of downstream interpretation. The trained SAEs are
immediately reusable as activation-addition steering vectors (Turner et
al., 2023). Because the SAE training pipeline is fully deterministic and
requires no trained SLM checkpoint, the whole analysis can be re-run
against the random-init baseline or — when available — a production
checkpoint, which preserves the Batch B architectural-capacity framing.

## 1. Introduction

Cross-attention conditioning, as realised in `AdaptiveTransformerBlock`,
is the load-bearing architectural claim of the I³ project. Every
transformer block applies a three-sub-layer Pre-LN pattern — self-
attention, cross-attention to the 4 conditioning tokens produced by
`ConditioningProjector`, and feed-forward — so that the adaptation
signal influences the residual stream at every depth. Batch A's ablation
showed *that* the conditioning matters (Cohen's d > 0.8 on KL-divergence
against prompt-based baselines); Batch B's activation patching and
linear probes gave a layer-level account of *where* the signal flows
(cross-attention heads in mid-depth layers and the `ConditioningProjector`
being on the critical path). Batch G3 closes the remaining gap: *what
interpretable concepts does the model learn to encode at each layer, and
how do those concepts align with the eight adaptation dimensions?*

The interpretability frontier moved decisively in that direction in
2023. Bricken et al. showed that a one-hidden-layer autoencoder with an
overcomplete basis and an L1 sparsity penalty could decompose a small
language model's activations into a dictionary of monosemantic features,
each activating only on a narrow slice of the input distribution.
Templeton et al. (2024) scaled the approach to Claude 3 Sonnet and
uncovered tens of thousands of interpretable features, some with direct
safety and steering applications. Cunningham et al. (2023) confirmed the
result on GPT-2 and Pythia, and the ensuing year of work (Marks et al.,
2024; Gao et al., 2024; Rajamanoharan et al., 2024) has turned SAEs into
the dominant tool for circuit discovery. Applying the same tool to a
conditioned SLM is the natural next experiment: if cross-attention is
the mechanism by which user-state signals modulate generation, the SAEs
trained on the cross-attention residual stream should recover features
that align with the adaptation axes we engineered, and they should do so
at layers whose depth matches the Batch B linear-probe peak.

## 2. Related work

**Monosemanticity and dictionary learning.** Bricken et al. (2023)
*Towards Monosemanticity: Decomposing Language Models With Dictionary
Learning* established the SAE recipe this note follows: a ReLU
autoencoder with `d_dict ≈ 8·d_model`, an L1 penalty on the hidden
activations, a shared decoder bias that doubles as an input-centring
term, and a decoder-column unit-norm projection applied after every
optimiser step (their §5.2). Templeton et al. (2024) *Scaling
Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*
pushed `d_dict` into the tens of millions on production-scale models and
published a gallery of feature-level safety interventions. Cunningham et
al. (2023) *Sparse Autoencoders Find Highly Interpretable Features in
Language Models* independently confirmed the recipe on GPT-2 and Pythia
and introduced the now-standard per-feature activation-correlation
protocol that we follow in §4.

**SAE-based circuits.** Marks et al. (2024) use SAEs as the primitive
units for end-to-end circuit discovery, extending the transformer-
circuits tradition of Elhage et al. (2021) and Olsson et al. (2022)
beyond induction heads to user-facing behavioural circuits. Their
*feature-level ablation* and *feature-level patching* protocols are
natural follow-ons to Batch B's component-level patching, and are an
explicit target of the *Future work* section below.

**Activation steering.** Turner et al. (2023) *Activation Addition:
Steering Language Models Without Optimisation* (ActAdd) showed that
adding a scaled direction to a transformer's residual stream at a single
layer is sufficient to bias generation in a controlled way; Zou et al.
(2023) *Representation Engineering* generalised the move to a broader
programme. We implement ActAdd with the SAE's decoder columns as the
directions, which lets us combine interpretation and intervention in a
single pipeline (`ActivationSteerer` in this batch).

**Superposition.** Elhage et al. (2022) *Toy Models of Superposition*
is the theoretical basis for why dictionary learning with an
overcomplete basis is the right architectural choice: a small model
with a bottleneck of width `d_model` routinely packs many more than
`d_model` concepts into its activations by exploiting nearly-orthogonal
directions; an SAE with a larger hidden dimension can recover those
directions one at a time.

## 3. Method

### 3.1 SAE architecture

For a residual-stream activation `x ∈ R^{d_model}`, an SAE with feature
dictionary size `d_dict` computes

```
f = ReLU(W_e (x - b) + b_e)          # features, shape [d_dict]
x̂ = W_d f + b                         # reconstruction, shape [d_model]
```

where `b ∈ R^{d_model}` is the decoder bias (shared with the input
centring term), `W_e ∈ R^{d_dict × d_model}`, `W_d ∈ R^{d_model × d_dict}`,
and each column of `W_d` is projected back to unit L2 norm after every
optimiser step (Bricken 2023 §5.2). The input-centring and unit-norm
constraints jointly remove the trivial sparsity degeneracy in which the
encoder's L1 penalty is absorbed by scaling the decoder columns up.

### 3.2 Training objective

The per-sample loss is the standard MSE reconstruction error plus a
scaled L1 on the hidden activations:

```
L(x) = ||x - x̂||_2^2 + λ · ||f||_1
```

with `λ = 1e-3` matching Bricken's default. We optimise with Adam at
`lr = 1e-3`, batch size 256, for 50 epochs on the cached residual-
stream activations of each cross-attention block. Training is
deterministic under a fixed seed and uses `torch.compile` when available
for a modest speed-up.

### 3.3 Overcomplete basis

We set `d_dict = 8 · d_model`, the Bricken 2023 default. This is large
enough that superposition (Elhage et al., 2022) can be unpacked into
largely orthogonal directions, but small enough that the resulting SAE
fits comfortably on a laptop CPU. For the I³ default (`d_model = 64`)
this yields 512 features per layer and 2 048 features across the full
four-layer sweep.

### 3.4 Per-layer training

One SAE is trained per cross-attention block, hooked onto
`model.layers[i].cross_attn`. The activation cache is built by running
the 50 canonical prompts from Batch A against each of the 8 archetype
`AdaptationVector`s (400 prompt/adaptation combinations), which gives
~6 400 flattened activation rows per layer after the default seq-length
of 16.

### 3.5 Feature-semantics correlation

After training we correlate each feature's activation across the cache
with each of the eight `AdaptationVector` dimensions using both Pearson
and (when SciPy is available) Spearman coefficients. A feature is
labelled *monosemantic* if its strongest Pearson correlation exceeds a
configurable threshold (default 0.7, Bricken 2023 §4.2). The
`FeatureSemantics` Pydantic dataclass stores the feature index, the
top-three correlations, mean and max activations, sparsity, and the
auto-assigned label.

### 3.6 Activation steering

Trained SAEs double as steering-vector libraries: each decoder column is
a direction in residual space that "points at" the corresponding
feature. `ActivationSteerer` attaches a forward hook to a target layer
and adds a scalar multiple of the direction to the residual stream at
inference time, implementing Turner et al. (2023)'s ActAdd in the SAE-
basis basis advocated by Templeton et al. (2024).

## 4. Results (placeholder)

Produced by running

```
python scripts/training/train_sae.py --seed 42
python scripts/experiments/analyse_sae.py --seed 42
```

`reports/sae_training_<ts>.md` tabulates the per-layer MSE, sparsity,
and loss trajectory; `reports/sae_analysis_<ts>.md` tabulates the
monosemantic-feature count per layer, the top-k features per
`AdaptationVector` dimension, and the decoder-column cosine-similarity
heatmap summary. On a random-init SLM the raw numbers reflect
architectural capacity (per the Batch B disclaimer); on a checkpoint
they report learned behaviour.

## 5. Discussion

The SAE dictionary provides the missing vocabulary layer for the claims
that the ablation (Batch A) and probe (Batch B) studies already made at
a coarser granularity. A probe R² of 0.6 for *cognitive load* at layer 2
is a summary statistic; a monosemantic feature with a +0.83 Pearson
correlation with the *cognitive load* dimension is a *thing*, and the
corresponding decoder column is a manipulable object.

For the I³ use case the most actionable output is the dimension-feature
mapping table: a short list of "cognitive-load features", "formality
features", "accessibility features", each with an activation histogram,
a list of top-activating prompts, and a decoder column ready to be fed
into `ActivationSteerer`. That table is the concrete embodiment of the
claim that cross-attention conditioning is a differentiable interface
between the perception pipeline and the language-generation pipeline: if
the interface were purely prompt-based, no such feature would appear on
the residual stream, because the residual stream would not carry the
adaptation signal in a coherent direction.

Connecting back to the Batch B probe picture: the probe is a linear
*read-out* of the residual stream, and it implicitly assumes that the
target dimension is encoded in a single direction. An SAE recovers that
direction explicitly, and also recovers the secondary directions the
probe averages over. The number of monosemantic features per layer is a
finer-grained substitute for the probe R² and should peak at the same
depth.

## 6. Threats to validity

- **SAE overfit.** With `d_dict = 8·d_model` the SAE has ample capacity
  to memorise the training cache; nothing prevents it from
  reconstructing every single activation with a unique feature. The L1
  penalty combats this but does not eliminate it. A held-out cache and
  cross-validated MSE would harden the claim.
- **Polysemantic features.** Not every feature will clear the 0.7
  threshold. Polysemantic features are still useful — their top-three
  correlations show which dimensions they blend — but they are not
  reported as interpretable concepts.
- **Random-init baseline.** On an untrained SLM the activations are the
  outputs of a random non-linear mixing of the conditioning tokens; any
  feature-dimension correlation above chance reflects the architectural
  bandwidth for the dimension to be represented, not learned behaviour.
  Running the same pipeline against a checkpoint removes this caveat.
- **Ground-truth feature labels.** We rely on correlation against the
  engineered adaptation dimensions for labelling. The Templeton 2024
  programme supplements this with human-judge and LLM-judge evaluations
  of top-activating examples; we leave that for future work.

## 7. Future work

- **SAE-based circuit discovery** (Marks et al., 2024): use the SAE
  features as the primitive units of a feature-level extension of Batch
  B's activation-patching harness, so that "patch feature 17 at layer 2"
  replaces "patch the entire cross-attention output at layer 2".
- **Top-K SAEs** (Gao et al., 2024): replace the L1 penalty with a hard
  top-k selection, which empirically yields a sharper monosemanticity /
  reconstruction Pareto front.
- **Jumping ReLU SAEs** (Rajamanoharan et al., 2024): an activation-
  function tweak that further reduces dead features and increases the
  fraction of interpretable features per layer.
- **Automated feature labelling** with a judge LLM (Claude Sonnet is the
  natural choice given Batch G4's commitment to the same tool for
  response evaluation): replace or complement the correlation-with-
  dimension labels with natural-language summaries of the top-activating
  prompts.
- **Cross-layer feature tracking**: use the decoder-column cosine
  heatmap to identify feature families that recur across layers, a
  pre-requisite for multi-layer steering.

---

## References

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A.,
  Conerly, T., et al. (2023). *Towards Monosemanticity: Decomposing
  Language Models With Dictionary Learning.* Anthropic Circuits
  Thread.
- Templeton, A., Conerly, T., Marcus, J., Bricken, T., et al. (2024).
  *Scaling Monosemanticity: Extracting Interpretable Features from
  Claude 3 Sonnet.* Anthropic Circuits Thread.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L.
  (2023). *Sparse Autoencoders Find Highly Interpretable Features in
  Language Models.* arXiv:2309.08600.
- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., &
  Mueller, A. (2024). *Sparse Feature Circuits: Discovering and
  Editing Interpretable Causal Graphs in Language Models.*
  arXiv:2403.19647.
- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R.,
  Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and
  Evaluating Sparse Autoencoders.* arXiv:2406.04093.
- Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V.,
  Kramár, J., Shah, R., & Nanda, N. (2024). *Improving Dictionary
  Learning with Gated Sparse Autoencoders / Jumping ReLU SAEs.*
  arXiv:2404.16014.
- Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., &
  MacDiarmid, M. (2023). *Activation Addition: Steering Language
  Models Without Optimisation.* arXiv:2308.10248.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., et al.
  (2023). *Representation Engineering: A Top-Down Approach to AI
  Transparency.* arXiv:2310.01405.
- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T.,
  Kravec, S., et al. (2022). *Toy Models of Superposition.*
  Anthropic Circuits Thread.
- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann,
  B., et al. (2021). *A Mathematical Framework for Transformer
  Circuits.* Anthropic Circuits Thread.
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N.,
  Henighan, T., et al. (2022). *In-context Learning and Induction
  Heads.* arXiv:2209.11895.
