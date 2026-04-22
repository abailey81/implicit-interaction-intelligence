# Cross-Attention Conditioning: A Pre-Registered Ablation

**Author:** Tamer Atesyakar
**Date:** 2026-04-22
**Status:** Method + data frozen; empirical results placeholder until
`scripts/run_ablation_study.py` is executed.

---

## Abstract

The Implicit Interaction Intelligence (I³) system personalises a small
language model's output by injecting a compact adaptation vector into
every transformer layer through cross-attention. Although this
*architectural* conditioning has been the signature novelty claim of the
I³ architecture since the earliest design notes, no numerical evidence
for its responsiveness advantage over prompt-based personalisation has
previously been recorded in the repository. This report documents the
Batch A ablation study that closes that gap. We contrast three
conditions — no conditioning, prompt-based conditioning, and
cross-attention conditioning — across eight archetype adaptation
vectors and fifty prompts, computing pairwise symmetric KL divergence
between next-token distributions as the primary responsiveness metric.
The study is pre-registered, seeded, and runnable on a random-init
model.

---

## 1. Introduction

The core design hypothesis of I³ is that a small language model (SLM)
can be made genuinely adaptive by structurally wiring a user-state
representation into its attention mechanism, rather than by describing
the user in a system prompt. Section 8 of
`docs/ARCHITECTURE.md` lays out the argument in full: prompt-based
personalisation suffers from attention dilution, consumes context, and
is especially weak on tiny models that cannot follow subtle system
prompts. Cross-attention to a projected conditioning tensor avoids all
three problems in principle. *In principle* is where the claim has
stopped, however: no quantitative comparison had been performed in the
repository prior to Batch A.

This report addresses that gap. We do **not** claim to demonstrate that
the cross-attention pathway produces *better* output than a prompt — such
a claim requires a trained model and a quality benchmark, which is the
scope of Batch C. Instead we measure **responsiveness**: how much do the
next-token probabilities shift when the adaptation input changes, under
each conditioning regime? Under a random-init SLM the absolute
responsiveness values will be modest, because the weights have never been
tuned to amplify the conditioning signal. But even at random
initialisation the cross-attention pathway has *more architectural
capacity* to respond than a prompt prefix does, and the pre-registered
directional prediction is that this capacity manifests as strictly
ordered mean KL divergences.

### 1.1 Contributions

1. A fully pre-registered three-condition ablation
   (`docs/experiments/preregistration.md`), including seeds, stopping
   rule, exclusions, and analysis plan, per the convention of
   Nosek et al. (2018).
2. A runnable experiment class `AblationExperiment` in
   `i3/eval/ablation_experiment.py` with three `_run_condition_*`
   methods, producing a Pydantic `AblationResult` with per-pair records
   and bootstrap-CI summaries.
3. A statistical-helpers module `i3/eval/ablation_statistics.py`
   providing `bootstrap_ci`, `cohens_d`, `paired_sign_test`, and
   `effect_size_interpretation`, each 100 %-typed and unit-tested.
4. A CLI driver `scripts/run_ablation_study.py` that emits both
   machine-readable JSON and a human-readable Markdown report stamped
   with the run-time git SHA.

---

## 2. Method

### 2.1 Conditions

**None.** The model's `adaptation_vector` argument is set to zeros and
its `user_state` argument is set to zeros. The prompt is the bare user
input. Under this condition, swapping archetypes cannot move the
next-token distribution at all — any non-zero KL is floating-point
noise. This is the H1 floor.

**Prompt.** The archetype adaptation vector is verbalised into a short
natural-language prefix (for example, `"[System: respond elaborate,
formal, neutrally, with technical depth.] "`) and prepended to the
prompt. The architectural conditioning path is held neutral. This is
the modern-LLM standard for personalisation.

**Cross-attention.** The archetype adaptation vector is passed as an
8-dimensional tensor to `AdaptiveSLM.forward(...)`, routed through the
`ConditioningProjector` into four conditioning tokens, and attended to
at every transformer layer via the cross-attention sub-layer defined in
`i3/slm/transformer.py`. The prompt is the bare user input.

### 2.2 Data

The experimental grid is the Cartesian product of **8 archetype
`AdaptationVector`s** (Appendix B) and **50 prompts** (Appendix A),
evaluated under each of the **3 conditions**, giving 1 200 forward
passes total. For KL computation we then form all
`C(8, 2) = 28` archetype pairs per prompt per condition, yielding
1 400 per-pair KL values per condition (28 × 50).

### 2.3 Metrics

**Primary — symmetric KL divergence.** For each archetype pair
`(c_i, c_j)` and prompt `p`, we compute the symmetric KL
`KL_sym(P(· | p, c_i) || P(· | p, c_j))` in nats from the softmax of
the final next-token logits.

**Secondary — style fidelity.** Greedy-decode 16 tokens under each
condition and score the actual continuation length against a
target length derived from the archetype's `verbosity` bucket (terse,
balanced, verbose). The score is a Gaussian log-likelihood penalty
`−(L_target − L_actual)² / (2σ²)` with `σ = 4`.

**Secondary — latency.** Wall-clock time for a single forward pass
with a 32-token prompt on CPU, reported as P50 / P95 / P99 over all
400 (prompt × archetype) runs per condition.

### 2.4 Statistical plan

Per the pre-registration: percentile-bootstrap 95 % confidence
intervals with 10 000 resamples; Cohen's `d` between every pair of
conditions (three comparisons); paired sign test on the per-pair KL
samples with an exact binomial null. A Bonferroni correction to
α = 0.05 / 3 is applied when reporting joint significance for H1.

### 2.5 Seeding and reproducibility

Global seed 42. Random sources documented in
`docs/experiments/preregistration.md` §7. The output Markdown report
embeds the repository HEAD SHA at run time so that any result table can
be matched to the exact code.

### 2.6 Pre-registration link

The full pre-registration is at
[`docs/experiments/preregistration.md`](./preregistration.md). The
study design recorded there fixes: research question, three directional
hypotheses, three conditions, data grid (8 × 50), primary + two
secondary metrics, bootstrap resampling count, effect-size thresholds,
seeding, stopping rule, exclusions, and the requirement to embed the
code SHA in each report. **We have not deviated from the pre-registration.**

---

## 3. Results

> Results below are placeholders. They will be populated by
> `scripts/run_ablation_study.py`.

### 3.1 Condition summary

| Condition    | KL mean (nats)                                          | 95 % CI                                                  | Style fidelity                                          | Latency P50 (ms)                                        | P95 (ms) | P99 (ms) |
|:-------------|:--------------------------------------------------------|:---------------------------------------------------------|:--------------------------------------------------------|:--------------------------------------------------------|:---------|:---------|
| `none`       | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled]                                          | [tbd]    | [tbd]    |
| `prompt`     | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled]                                          | [tbd]    | [tbd]    |
| `cross_attn` | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled by scripts/run_ablation_study.py]          | [to be filled]                                          | [tbd]    | [tbd]    |

### 3.2 Pairwise comparisons

| Comparison               | Cohen's d     | Interpretation | Sign-test p   |
|:-------------------------|:--------------|:---------------|:--------------|
| `cross_attn_vs_prompt`   | [tbd]         | [tbd]          | [tbd]         |
| `cross_attn_vs_none`     | [tbd]         | [tbd]          | [tbd]         |
| `prompt_vs_none`         | [tbd]         | [tbd]          | [tbd]         |

### 3.3 H3 latency overhead

The H3 hypothesis predicts that the P50 latency overhead of the
cross-attention condition relative to the no-conditioning condition is
less than 15 %. The fill-in value is produced by the CLI at run time.

---

## 4. Discussion

### 4.1 Interpretation under random-init

The architectural prediction is directional: `cross_attn` should
strictly dominate `prompt`, which should strictly dominate `none`.
The `none` condition is a definitional floor — swapping the archetype
has no effect on the model input, so any non-zero KL there is
floating-point noise. The `prompt` condition feeds a different prefix
into the embedding table, so the downstream representations will
differ non-trivially, and KL will be clearly above zero even on
random-init weights. The `cross_attn` condition feeds a different
8-dimensional tensor through a dense MLP into four conditioning
tokens that are then attended to at every layer. Even without training,
this provides *more parameters* along which the input can move the
output, and we expect cross-attention KL to be the highest.

Crucially, on a random-init model we do **not** expect the
cross-attention responsiveness to be useful in any downstream sense;
we only expect it to be *large*. The usefulness depends on training,
which is the scope of Batch C.

### 4.2 Comparison with the existing conditioning-sensitivity scaffolding

The repository already contains a related but smaller-scope module at
`i3/eval/conditioning_sensitivity.py`. That module measures the
cross-attention KL matrix across four canonical archetypes and a
user-supplied prompt list, and emits a per-prompt table. It is useful
for spot-checking a trained model during development. It does **not**
evaluate a prompt-based or no-conditioning baseline, it does not carry
pre-registered hypotheses, it does not compute effect sizes or
confidence intervals, and it does not time the forward passes. Batch A
supersedes it for inferential purposes while leaving it in place as a
lightweight sensitivity probe.

### 4.3 Limitations

1. **Random-init weights.** The absolute KL numbers are not
   interpretable as evidence that cross-attention learns a useful
   conditioning signal — only that it is *capable* of one. A trained
   model is required to convert this into a quality claim.
2. **Synthetic archetypes.** The eight `AdaptationVector`s are
   hand-authored corners of the adaptation space, not empirically
   sampled from real user behaviour. A model trained on one distribution
   of archetypes may be more or less responsive than the Batch A
   numbers suggest.
3. **Short contexts.** All prompts are capped at 32 tokens. The
   principal *motivation* for architectural conditioning —
   attention dilution over long contexts — is not itself measured.
4. **Single seed.** Variance across seeds is not reported. The
   bootstrap CIs are within-sample and do not quantify cross-run
   variance.
5. **CPU-only latency claim.** The 15 % overhead threshold in H3 is
   set for CPU; on GPU the relative overhead of cross-attention is
   different because the conditioning is much cheaper than self-
   attention per layer.

### 4.4 Future work

- Repeat on a trained checkpoint (`I3_CHECKPOINT_PATH`) — the
  experiment harness accepts one without modification.
- Extend to a long-context regime (256 or 512 prompt tokens) to
  directly test the attention-dilution motivation.
- Multi-seed sweep (seeds `42, 43, 44, 45, 46`) to produce genuine
  between-run confidence intervals.
- Measure responsiveness *after* instruction-tuning vs. before to
  quantify how much of the architectural capacity gets used during
  training.

---

## References

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
  *A simple framework for contrastive learning of visual
  representations (SimCLR).* In ICML.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
  Sciences* (2nd ed.). Routledge.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018).
  *The preregistration revolution.* PNAS 115(11), 2600–2606.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
  Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
  *Attention is all you need.* In NeurIPS.
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H.,
  Lan, Y., Wang, L., & Liu, T.-Y. (2020). *On layer normalization in
  the transformer architecture.* In ICML.

---

## Appendix A — The 50 prompts

The 50 prompts are hard-coded in
`i3/eval/ablation_experiment.py::_CANONICAL_PROMPTS` and accessible via
`canonical_prompts()`. They are reproduced verbatim below.

### A.1 Conversational (13)

1. Tell me about your weekend.
2. What is your favourite season, and why?
3. How have you been feeling lately?
4. Do you enjoy reading novels?
5. What did you have for breakfast today?
6. If you could travel anywhere, where would you go?
7. Describe your ideal Sunday morning.
8. What music do you listen to when working?
9. Tell me a short story about a cat and a lighthouse.
10. What makes you laugh?
11. Who is someone you admire, and why?
12. Describe a place that feels calm to you.
13. What is the best meal you have had recently?

### A.2 Technical (13)

14. Explain how a transformer self-attention layer works.
15. What is the difference between variance and standard deviation?
16. Summarise the key idea behind reinforcement learning.
17. What is the purpose of layer normalisation in a neural network?
18. Explain what a closure is in Python.
19. What is the time complexity of merge sort?
20. Why is cross-validation important in machine learning?
21. How does a hash table resolve collisions?
22. Explain gradient descent to a beginner.
23. What is the CAP theorem?
24. How does TCP differ from UDP?
25. Describe how a B-tree index speeds up database lookups.
26. What is overfitting, and how can it be mitigated?

### A.3 Emotional support (12)

27. I feel overwhelmed by work lately.
28. I cannot seem to focus today.
29. I had an argument with a close friend.
30. I am anxious about an upcoming interview.
31. I lost a pet recently and I miss them.
32. I am struggling to sleep this week.
33. I feel guilty for taking a day off.
34. I doubt my own decisions a lot.
35. I am lonely even when people are around.
36. I feel stuck in my current job.
37. I worry that I am disappointing my family.
38. I keep procrastinating on something important.

### A.4 Task-oriented (12)

39. Please write a two-sentence apology email.
40. Draft a polite request to reschedule a meeting.
41. Give me three tips for writing cleaner code.
42. List four healthy lunch ideas.
43. Help me plan a 30-minute home workout.
44. Outline the agenda for a project kick-off meeting.
45. Write a thank-you note for a job interview.
46. Suggest a safe icebreaker for a new team.
47. Summarise the key steps of the scientific method.
48. Generate a packing list for a three-day work trip.
49. Draft a bullet-point weekly status update.
50. Write a short review of a book you enjoyed.

---

## Appendix B — The 8 archetype AdaptationVectors

Layout: `[cognitive_load, formality, verbosity, emotionality,
directness, emotional_tone, accessibility, reserved=0.0]` — matching
`AdaptationVector.to_tensor()`.

| Archetype              | cog_load | form. | verb. | emo.  | direct. | tone | access. |
|:-----------------------|:--------:|:-----:|:-----:|:-----:|:-------:|:----:|:-------:|
| `neutral`              | 0.50     | 0.50  | 0.50  | 0.50  | 0.50    | 0.50 | 0.00    |
| `low_load_warm`        | 0.10     | 0.20  | 0.30  | 0.80  | 0.30    | 0.10 | 0.60    |
| `high_load_technical`  | 0.90     | 0.80  | 0.80  | 0.20  | 0.80    | 0.70 | 0.10    |
| `urgent_formal`        | 0.60     | 0.90  | 0.20  | 0.20  | 0.95    | 0.50 | 0.20    |
| `accessible_simple`    | 0.15     | 0.30  | 0.30  | 0.60  | 0.50    | 0.30 | 0.95    |
| `casual_verbose`       | 0.40     | 0.10  | 0.90  | 0.70  | 0.40    | 0.20 | 0.30    |
| `direct_terse`         | 0.50     | 0.50  | 0.10  | 0.20  | 0.95    | 0.60 | 0.20    |
| `reflective_neutral`   | 0.50     | 0.50  | 0.70  | 0.50  | 0.30    | 0.40 | 0.40    |

The archetypes cover low/high cognitive load, warm vs. neutral
emotional tone, high/low verbosity, accessibility on and off, and the
`urgent_formal` mix that stresses the directness dimension without
emotional expressiveness.

---

*End of report.*
