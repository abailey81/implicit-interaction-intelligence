# Pre-Registration — Cross-Attention Conditioning Ablation (Batch A)

- **Title:** Empirical Ablation of Cross-Attention Conditioning vs. Prompt-Based
  and No-Adaptation Baselines in the I³ Adaptive Small Language Model.
- **Author:** Tamer Atesyakar (`t.ates232004@gmail.com`).
- **Date:** 2026-04-22.
- **Pre-registration convention:** Nosek, B. A., Ebersole, C. R., DeHaven, A. C.,
  & Mellor, D. T. (2018). *The preregistration revolution.* PNAS 115(11).

This document is registered **before** running `scripts/experiments/ablation_study.py`.
No analysis has been performed on the outputs at the time of writing; the
analysis plan below is therefore fully confirmatory.

---

## 1. Research Question

> Does cross-attention conditioning produce next-token distributions that are
> more responsive to changes in an `AdaptationVector` than prompt-based
> conditioning or no conditioning at all, as measured by pairwise KL
> divergence of the next-token distribution under a random-init I³ SLM?

The empirical claim embedded in
`docs/architecture/full-reference.md` §8 ("the Novel Cross-Attention Conditioning") is that
cross-attention can **steer** generation where prompt-based conditioning
cannot. The claim has not previously been tested numerically in this
repository; Batch A closes that gap.

---

## 2. Hypotheses (directional)

Let `KL_mean(C)` be the mean symmetric KL divergence between next-token
probability distributions produced by condition `C` when the conditioning
input is varied (two different `AdaptationVector`s, `c1` and `c2`,
with the prompt held fixed), averaged over the experimental grid.

- **H1 (primary).**
  `KL_mean(cross_attn) > KL_mean(prompt) > KL_mean(none)`.
  Directional prediction: strictly decreasing from cross-attention to
  prompt-based to no conditioning.

- **H2 (style fidelity).**
  Style-fidelity match score (token-length-distribution entropy under target
  `verbosity`) is ordered
  `fidelity(cross_attn) > fidelity(prompt) > fidelity(none)`.

- **H3 (latency).**
  CPU latency overhead of cross-attention conditioning relative to
  no-conditioning is `< 15 %` at the 50th percentile on a single-threaded
  CPU forward pass over 32-token prompts.

No two-sided hypotheses are registered; the alternative hypothesis for each
is one-tailed in the stated direction, with the null being equality.

---

## 3. Experimental Conditions

| Code          | Condition                 | Mechanism                                                                                                 |
|:--------------|:--------------------------|:----------------------------------------------------------------------------------------------------------|
| `none`        | No conditioning           | `AdaptiveSLM.forward(input_ids, adaptation_vector=zeros, user_state=zeros)` — the model's default.         |
| `prompt`      | Prompt-based conditioning | A short natural-language description of the `AdaptationVector` is prepended to the prompt; `adaptation_vector=zeros`. |
| `cross_attn`  | Cross-attention           | The `AdaptationVector` is passed through the `ConditioningProjector` into all four `AdaptiveTransformerBlock` cross-attention sub-layers. |

Each condition is evaluated on the **same random-init model** — no separate
training runs. We are measuring architectural *responsiveness*, not quality.

---

## 4. Data

### 4.1 Prompts

A fixed list of **50 prompts** covering four registers (conversational,
technical, emotional-support, task-oriented) is hard-coded inside
`i3/eval/ablation_experiment.py::_CANONICAL_PROMPTS`. The list is fully
enumerated in the accompanying `ablation_report.md` Appendix A for
reproducibility.

### 4.2 Adaptation Vectors

**8 archetype `AdaptationVector`s**, each representing a distinct user
profile (e.g., *low-load-warm*, *high-load-technical*, *urgent-formal*,
*accessibility-first*). Full numerical values listed in
`ablation_report.md` Appendix B. They are hard-coded inside
`i3/eval/ablation_experiment.py::_CANONICAL_ARCHETYPES`.

### 4.3 Grid and Stopping Rule

Total runs per condition: `8 archetypes × 50 prompts = 400`.
Total runs overall: `400 × 3 conditions = 1200`. The experiment
**terminates after exactly 1200 forward passes.** No early stopping, no
interim looks, no sequential-testing correction.

---

## 5. Primary and Secondary Metrics

### 5.1 Primary — KL divergence

For each prompt `p` and each pair of archetype vectors `(c_i, c_j)` with
`i < j`, compute `KL_sym(P(· | p, c_i) || P(· | p, c_j))` where `KL_sym`
is the symmetric KL divergence in nats and `P` is the softmax of the final
next-token logits. The per-condition primary statistic is the **mean over
all `(p, i, j)` triples** where the condition is held fixed.

### 5.2 Secondary — Style fidelity

Under each condition we greedy-decode 16 tokens and compute the
**length-distribution entropy** of the generated continuation grouped by
archetype `verbosity` bucket (low, mid, high). Fidelity is the negative
cross-entropy between the observed length histogram and the target
verbosity-conditioned prior (a Gaussian centred at the target bucket).

### 5.3 Secondary — Latency

For each condition we measure wall-clock time for a single forward pass
with prompt length 32 on CPU. Report P50, P95, P99 in milliseconds over
all 400 (archetype, prompt) runs per condition.

---

## 6. Statistical Plan

- **Bootstrap 95 % confidence intervals.** For each condition's
  per-pair KL sample, draw `n_resamples = 10_000` bootstrap resamples of
  the per-pair KL values and report the 2.5th–97.5th percentile.
- **Effect sizes.** Cohen's `d` between each pair of conditions
  (`cross_attn` vs `prompt`, `cross_attn` vs `none`, `prompt` vs `none`).
  Interpretation thresholds (Cohen 1988): `|d| < 0.2` negligible,
  `0.2 ≤ |d| < 0.5` small, `0.5 ≤ |d| < 0.8` medium, `|d| ≥ 0.8` large.
- **Paired sign test.** For the (prompt × archetype-pair) grid we obtain
  per-cell paired samples of `KL_cross_attn - KL_prompt` and
  `KL_prompt - KL_none`. Run a paired sign test and report the exact
  binomial p-value. Significance threshold: α = 0.05, one-tailed in the
  direction predicted by H1.
- **Correction.** Three pairwise comparisons are made; we apply Bonferroni
  correction (α_family = 0.05 / 3 ≈ 0.0167) when reporting joint
  significance for H1.

---

## 7. Seeds and Random Sources

- **Global seed:** `42`.
- **Random source list (documented):**
  - `torch.manual_seed(42)` — model weights (random-init), dropout (disabled
    at eval time).
  - `numpy.random.default_rng(42)` — bootstrap resampling.
  - `random.seed(42)` — any other Python-level randomness.
- **Determinism settings:** `torch.use_deterministic_algorithms(True)` where
  supported, `torch.backends.cudnn.deterministic = True` on CUDA
  (CPU default is deterministic). Any platform-specific non-determinism is
  documented in the final report as a threat to reproducibility.

---

## 8. Exclusions

**None.** The dataset is fully synthetic and fully enumerated; there is no
outlier removal, no missing-data handling, and no post-hoc filtering of any
kind. All 1200 runs contribute to all reported statistics.

---

## 9. Analysis Code Hash

The Markdown report produced by `scripts/experiments/ablation_study.py` includes
the git SHA of the repository HEAD at the time the script was run, via
`git rev-parse HEAD`. The SHA is embedded in the report directly under
the title so that the reader can verify this pre-registration describes
the exact code that produced the reported results.

Any deviation from this plan will be flagged in the report under a
**"Deviations from Pre-Registration"** section and justified.

---

## 10. Author's Declaration

I confirm that no preliminary analysis has been performed on the 1200-run
grid at the time this document is committed. The only preparatory
measurement was a single-prompt sanity check on a separate prompt not
included in the experimental set.

— Tamer Atesyakar, 2026-04-22.
