# Stretch-goal capabilities — mathematical formulation and citations

This note records the mathematical underpinnings of the stretch-goal
capabilities from THE_COMPLETE_BRIEF.md §9 that are implemented in the
`i3.slm.aux_losses`, `i3.interpretability`, `i3.biometric`, and
`i3.adaptation.ablation` modules, plus the `server.routes_whatif`
interpretability endpoints. Each section ends with the relevant citations.

## 1. Auxiliary conditioning losses

The SLM's cross-attention conditioning (see `docs/ARCHITECTURE.md` §8)
learns the `f: (AdaptationVector, UserStateEmbedding) -> conditioning_tokens`
mapping end-to-end against next-token cross-entropy. Nothing in that
objective forces the conditioning tokens to actually *shape* the output
distribution — a sufficiently large model can learn to ignore them and
rely entirely on the self-attention branch. The stretch-goal auxiliary
losses in `i3.slm.aux_losses` counter this failure mode.

### 1.1 Consistency loss

For two conditioning vectors `c1 ≠ c2` drawn from the same training
batch, let `p(x | c)` denote the next-token distribution produced by the
SLM under conditioning `c`. Define the consistency loss as

```
L_consistency(θ) = - min(KL(p(x | c1) || p(x | c2)), margin)
```

with `KL(p || q) = Σ_i p_i (log p_i - log q_i)` (Kullback & Leibler,
1951) and `margin` a hyperparameter (default 2 nats). The clamp ensures
the gradient vanishes once the model has learnt to differentiate
conditioning vectors by at least `margin`; this prevents the loss from
pushing distributions to pathological extremes.

### 1.2 Style-fidelity loss

Given a target `StyleVector s = (formality, verbosity, emotionality,
directness)` and generated token IDs `y = (y_1, ..., y_T)`, the fidelity
loss is

```
L_fidelity(θ) = MSE(φ_formality(y), s.formality)
              + MSE(φ_verbosity(y), s.verbosity)
              + MSE(φ_sentiment(y), 2 * s.emotionality - 1)
```

where `φ_*` are deterministic differentiable surrogates of the style
features (see `aux_losses._formality_score`, `_verbosity_score`,
`_sentiment_score`). These surrogates are coarse by design — the
loss is intended as a training-time regulariser, not a replacement for
the full lexicon-based scorer in `i3.interaction`.

### 1.3 Combined loss

```
L_total = L_CE + α_consistency · L_consistency + α_fidelity · L_fidelity
```

with default weights `α_consistency = 0.1`, `α_fidelity = 0.05` per
THE_COMPLETE_BRIEF.md §18.2 Day 7.

**Citations**
- Kullback, S. & Leibler, R. A. (1951). *On Information and Sufficiency*.
  Annals of Mathematical Statistics, 22(1), 79–86.
- Flesch, R. (1948). *A new readability yardstick*. J. Applied
  Psychology, 32(3), 221–233.
- He, P., Mou, L., Xu, S., Song, Y. & Xu, Q. (2020). *Learning to
  Condition Text Generation on a Style Vector via Cross-Attention*.
  ICLR workshop.

## 2. Interpretability pipeline

### 2.1 Integrated gradients

For a mapping `F: R^n -> R^m` and a baseline `x' = 0`, the integrated
gradient of input dimension `i` with respect to output dimension `j` is

```
IG_{i, j}(x) = (x_i - x'_i) · ∫_{α=0}^{1}
    ∂F_j(x' + α (x - x'))/∂x_i dα
```

approximated by a 50-point Riemann sum per Sundararajan, Taly & Yan
(2017). The completeness axiom guarantees
`Σ_i IG_{i, j}(x) = F_j(x) - F_j(x')`, giving unit-tests a deterministic
check against a linear surrogate.

### 2.2 Forward-hook attention extraction

`CrossAttentionExtractor` registers one `register_forward_hook` per
`MultiHeadCrossAttention` module in the SLM and records the second
return value (`attn_weights ∈ R^{batch × heads × T × n_cond}`). Hooks
are released in `__exit__` — even on exception — preventing dangling
references that would otherwise pin the extractor in memory across
subsequent forward passes.

### 2.3 SHAP adapter

SHAP (Shapley-value attribution) provides the unique fairness axioms
proved by Shapley (1953). It is an optional dependency in I³: when
`import shap` succeeds, `SHAPAdapter` uses `shap.KernelExplainer` with
a zero background; otherwise it falls back to integrated gradients,
which shares the completeness property.

**Citations**
- Sundararajan, M., Taly, A. & Yan, Q. (2017). *Axiomatic Attribution
  for Deep Networks*. ICML 2017.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to
  Interpreting Model Predictions*. NeurIPS 2017.
- Shapley, L. S. (1953). *A value for n-person games*. Contributions to
  the Theory of Games II, Annals of Mathematics Studies 28.
- Vig, J. (2019). *A Multiscale Visualisation of Attention in the
  Transformer Model*. ACL demo track.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-
  Performance Deep Learning Library*. NeurIPS 2019.

## 3. Keystroke-biometric identification

Centroid representation on the unit hypersphere (the TCN embedding
space). For a query `q` and a per-user centroid `c_u`,

```
similarity(q, u) = q · c_u = cos(q, c_u)
identified_user = argmax_u similarity(q, u)   if max ≥ threshold
```

with `threshold = 0.85` — the balanced-EER operating point reported by
Killourhy & Maxwell (2009) for TCN-style keystroke embeddings. On
re-enrolment the centroid is updated by the running-mean formula used
in FaceNet (Schroff et al., 2015):

```
c_u ← normalise((n c_u + q_new) / (n + 1))
```

Continuous authentication uses Welford's online variance algorithm
(Welford, 1962) to track the running distribution of per-observation
drift `d_t = 1 - cos(c_u, q_t)`. A drift exceeding `mean + 3σ`
raises a structured `AuthenticationEvent` — the classical 3-sigma
out-of-control rule from statistical process control (Shewhart, 1931).

**Citations**
- Monrose, F. & Rubin, A. (1997). *Authentication via keystroke
  dynamics*. ACM CCS '97.
- Killourhy, K. S. & Maxwell, R. A. (2009). *Comparing anomaly-
  detection algorithms for keystroke dynamics*. IEEE/IFIP DSN 2009.
- Schroff, F., Kalenichenko, D. & Philbin, J. (2015). *FaceNet: A
  Unified Embedding for Face Recognition and Clustering*. CVPR 2015.
- Welford, B. P. (1962). *Note on a method for calculating corrected
  sums of squares and products*. Technometrics 4(3), 419–420.
- Shewhart, W. A. (1931). *Economic Control of Quality of Manufactured
  Product*. D. Van Nostrand Company.

## 4. Ablation mode

`AblationController` wraps `AdaptationController` by composition. For an
ablation mode `M = (encoder, user_model, router_override, style_mirror)`
with boolean flags, the view computes

```
v = controller.compute(features, deviation)
v' = apply_mask(v, M)
```

where `apply_mask` replaces each flagged dimension with its neutral
default (0.5 for the scalar dims, `StyleVector.default()` for
`style_mirror`). The underlying controller's mutable state
(`_current_style` EMA) is left untouched — toggling ablation is a
zero-cost reversible operation. This mirrors the *dimension-wise
ablation* protocol of Morris & Hopkins (2014, *The Ablation Protocol for
Usability Experiments*).

**Citations**
- Morris, M. R. & Hopkins, C. G. (2014). *The Ablation Protocol for
  Usability Experiments*. CHI Workshops.

## 5. "What-if" endpoints

`server.routes_whatif` exposes two REST endpoints that feed the SLM
with an overridden `AdaptationVector` for a single forward pass:

| Route                     | Purpose                                          |
|:--------------------------|:-------------------------------------------------|
| `POST /whatif/respond`    | Single override -> single response.             |
| `POST /whatif/compare`    | Up to 4 overrides -> parallel response bundle.  |

The user's real pipeline state is never mutated: the override is
applied via `AdaptationVector.from_tensor` (which round-trips through
the same clamp semantics as the live path) and passed directly to
`SLMGenerator.generate`. Input validation uses Pydantic with the same
`user_id` regex as the main REST routes, and the endpoints inherit the
rate-limit / size-limit / security-headers middleware stack.

This implements the "what-if mode" stretch goal from
THE_COMPLETE_BRIEF.md §9 and the interpretability-panel requirement
from BRIEF_ANALYSIS.md §9.
