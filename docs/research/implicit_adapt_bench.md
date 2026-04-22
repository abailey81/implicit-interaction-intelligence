# ImplicitAdaptBench: A Benchmark for Adaptive Generation from Implicit Behavioural Signals

*Paper-style spec. Version 0.1.0. Companion to
`benchmarks/implicit_adapt_bench/` and
`benchmarks/implicit_adapt_bench/LEADERBOARD.md`.*

---

## Abstract

We introduce **ImplicitAdaptBench**, the first public benchmark measuring
response-generation adaptation from implicit behavioural signals —
keystroke timing, correction rate, linguistic complexity trends, baseline
deviations — rather than from an explicit user profile. The 2025–2026 wave
of personalisation benchmarks (PersonaLens 2025, AlpsBench 2025, PersoBench
2024) assumes the system is told "who the user is" in natural language and
judges faithful execution of that profile. Real assistants on device rarely
receive such a profile. They must infer user state from the interaction
itself and adapt in real time. We formalise that adaptation task, ship a
reproducible synthetic data generator, define five rule-based metrics that
do not depend on a trained judge model, provide three runnable baselines
(including a full cross-attention conditioning path), and outline an
IRB-lite protocol for collecting the small human-preference validation set
the scoring pipeline expects. The benchmark is intended to sit next to the
explicit-profile benchmarks, not replace them.

---

## 1. Motivation

The 2024–2026 explicit-profile wave:

* **PersoBench** (2024) probes whether a generator can produce responses
  consistent with a natural-language persona description.
* **AlpsBench** (2025) grounds profile-conditioned generation across a
  larger task mix.
* **PersonaLens** (2025) couples a user profile with a task transcript to
  evaluate longitudinal consistency.

All three supply the assistant with *what the user is*. They do not measure
what on-device assistants actually do: observe behaviour, infer state,
adapt mid-conversation. The closest public artefacts are the
keystroke-dynamics datasets from the HCI literature (Epp et al. 2011;
Vizer 2009; Zimmermann 2014), but those are classification datasets, not
generation benchmarks — they stop at "the user is frustrated", they never
ask "what should the model say back".

ImplicitAdaptBench is the generation-side half of that question.

The niche has three concrete implications for the design:

1. The benchmark input is a **behavioural window**, not a profile
   sentence. The generator is expected to *infer* style targets from
   features, not read them off.
2. The benchmark must be **small and CPU-friendly** because the target
   deployment is on-device. A benchmark that only runs on a GPU cluster
   would miss the population of systems it's meant to measure.
3. The metrics must **not require a trained judge**. Rule-based metrics
   are noisier than an LLM-as-judge, but they are cheap, reproducible
   across hardware, and legible — every scoring decision can be
   explained line-by-line.

---

## 2. Task formalisation

Let $\mathbf{x} \in \mathbb{R}^{32}$ be the canonical 32-dim interaction
feature vector at the end of a short sliding window
(`i3.interaction.types.InteractionFeatureVector`), let
$\mathbf{k} \in \mathbb{R}_{\geq 0}^{L}$ be the raw inter-key interval
sequence for the most recent message, and let
$\boldsymbol{\delta}\in \mathbb{R}^{8}$ be the baseline-deviation
z-scores. The tuple
$w = (\mathbf{x}, \mathbf{k}, \boldsymbol{\delta})$ is the **behavioural
window**.

Let $p$ be the user's prompt (UTF-8 text). A benchmark record is

$$r = (w, p, a^{\ast}, \ell^{\ast}, \mathbf{v}^{\ast}, h^{\ast}),$$

where $a^{\ast}$ is the target archetype label, $\ell^{\ast}$ is the
structured reference style label, $\mathbf{v}^{\ast} \in [0,1]^{8}$ is the
target AdaptationVector, and $h^{\ast} \in [0,1] \cup \{\varnothing\}$ is
the optional held-out human-preference score.

A **system** is a function
$f : (w, p) \mapsto y$ where $y$ is free text. The benchmark ranks
systems by a weighted mean of metric scalars
$m_{i}(y, r) \in [0,1]$:

$$S(f) = \sum_{i} \hat{w}_{i} \; \mathbb{E}_{r \sim \mathcal{D}_{\text{test}}}\!\left[m_{i}(f(w,p), r)\right],$$

with $\sum_i \hat{w}_i = 1$.

The three crucial constraints are:

1. $f$ may *not* read $a^{\ast}$, $\ell^{\ast}$, or $\mathbf{v}^{\ast}$ —
   only $(w, p)$ is surfaced.
2. $f$'s p95 latency is part of the score (see $m_5$ in §3).
3. $f$ must produce output on **every** record; abstention is scored as
   an empty string.

---

## 3. Metrics

All metrics return a scalar in $[0, 1]$.

### 3.1 Style match ($m_1$)

The structured reference label $\ell^{\ast}$ encodes three axes — tone,
formality, and length. Let $F(y)$ be the Flesch-based formality score,
let $N(y)$ be the length of $y$ in tokens, and let $V(y) \in [-1,1]$ be
the lexicon-based valence. For each axis the benchmark defines a target
value $\tau$; the axis sub-score is $\max(0, 1 - d)$ where $d$ is a
distance normalised to $[0, 1]$:

* Formality: $|F(y) - \tau_{f}|$ for $\tau_{f} \in \{0.2, 0.5, 0.9\}$.
* Length: piecewise-linear buckets with knots at $40$ and $120$ tokens.
* Tone: $|V(y) - \tau_{t}|/2$ for $\tau_{t} \in \{-0.1, 0.0, 0.5\}$.

The full score is the equal-weight mean. Axes the label is silent on
contribute the neutral $0.5$ so short labels don't skew the score.

### 3.2 Cognitive-load fidelity ($m_2$)

Let $G(y)$ be the Flesch-Kincaid grade of $y$. Given target load
$\lambda \in [0,1]$ the benchmark maps $\lambda$ to a grade band:

$$
[\ell, h](\lambda) =
\begin{cases}
[3, 6] & \lambda \leq 0.33, \\
[6, 10] & 0.33 < \lambda \leq 0.66, \\
[10, 14] & \lambda > 0.66.
\end{cases}
$$

The score is $1$ if $G(y) \in [\ell, h]$ and decays as
$\max(0, 1 - d/4)$ where $d$ is the one-sided grade distance to the band.

### 3.3 Accessibility appropriateness ($m_3$)

When the target accessibility $\alpha > 0.7$, we check three things:

1. *Short-sentence ratio*: fraction of sentences with at most 15 words.
2. *Idiom absence*: at most zero matches against a small curated idiom
   list. Each match deducts $1/3$.
3. *Yes/no confirmation*: at least one sentence ending in `?` whose
   first token is an English auxiliary/copula (``is``, ``can``, ...).

The score is the mean of the three sub-scores. When $\alpha \leq 0.7$ we
return $1.0$ — the benchmark does not penalise rich language when
accessibility was not requested.

### 3.4 Preference rate ($m_4$)

Only records in the held-out-human split carry $h^{\ast} \in [0,1]$. The
metric is the mean of those values over submissions whose record id
matches a scored gold record. Records with $h^{\ast} = \varnothing$ are
silently skipped.

### 3.5 Runtime budget compliance ($m_5$)

$m_5(f) = \frac{1}{|R|} \sum_{r \in R} \mathbb{1}\!\left[\text{runtime}^{p95}_r(f) \leq B\right]$,

with the default budget $B = 200$ ms matching the Batch A responsiveness
pre-registration. This guards against submissions that buy metric gains
with unrealistic latency.

### 3.6 Aggregation

The default weight vector is

$$\hat{w} = (0.35,\, 0.25,\, 0.15,\, 0.15,\, 0.10)$$

for $(m_1, m_2, m_3, m_4, m_5)$. Style match dominates (it is the core
responsiveness signal); preference is the strongest validity anchor we
have; runtime is a small brake.

---

## 4. Data

### 4.1 Synthetic splits

`benchmarks/implicit_adapt_bench/data_generator.py` deterministically
emits four splits — train, dev, test, held-out-human — from a single
base seed. For each of eight canonical archetypes (shared with the
Batch A ablation) the generator emits `n_per_archetype` records. Each
record's behavioural window is synthesised from the archetype's
AdaptationVector with per-channel Gaussian noise, so two records from
the same archetype are never identical.

The 32-dim feature vector is filled as follows:

* **Keystroke dynamics** (IKI, bursts, backspaces, composition speed) —
  scale with the archetype's cognitive-load and accessibility axes.
* **Message content** (length, TTR, word length, FK, formality, emoji,
  sentiment) — scale with the style-mirror axes.
* **Session dynamics** (trends, engagement velocity, topic coherence) —
  scale with verbosity, formality, and emotionality.
* **Deviation features** — Gaussian noise centred at zero so they
  behave as warm-up-baseline z-scores.

The generator also populates the raw IKI sequence with per-record lengths
10–60 and a mean scaled by the archetype's cognitive-load + accessibility
load.

Synthetic data is the right starting point for this benchmark because:

1. It removes subject-consent and licensing barriers.
2. It guarantees coverage of extreme corners of the archetype space.
3. It is cheap to regenerate at any new benchmark version.

The cost is acknowledged in §6: synthetic data cannot validate any
real-world claim. The held-out-human split exists specifically to close
that loop.

### 4.2 Held-out human-preference protocol (IRB-lite)

The held-out-human split is populated by annotating at most 64 synthetic
records with pairwise human preference (system A vs system B over the
same prompt and behavioural window). The protocol is the following
IRB-lite version of Mitchell et al.'s model-card-style disclosure:

1. **Purpose**: collect preference scores that let us validate
   `preference_rate` against a reasonable ground truth.
2. **Population**: adult English-speaking volunteers drawn from the
   project maintainers' direct network. Number fixed in advance, not
   opportunistic.
3. **Consent**: explicit written consent via a single-page form
   describing the task, the anonymous storage scheme (no identifiers
   retained beyond a per-session random id), and the right to withdraw.
4. **Task**: show each annotator a prompt and two candidate responses
   (A, B), ask which is "more appropriate given this user's typing
   behaviour". Typing behaviour is shown as plain-English bullet
   points, not raw numbers, to keep the task accessible.
5. **Compensation**: none; the pool is small enough to be a goodwill
   ask.
6. **Storage**: preference scores are stored in
   `benchmarks/implicit_adapt_bench/data/held_out_human.jsonl` as the
   `human_preference_score` field. No annotator identifier is ever
   committed.
7. **Re-collection**: at each major benchmark version the set is
   regenerated from scratch — old scores are not carried forward
   because the prompts and behaviours may have drifted.
8. **Opt-out**: submissions that do not want their output shown to
   annotators may opt out of the held-out row explicitly (they will
   appear on the dev and test rows only).

This is not a replacement for a full IRB review; it is the minimum
documented process that an academic supervisor would accept for a
benchmark of this size. Any real deployment (i.e. > 64 annotators,
compensated labour, data retained beyond the release) must escalate to
a full IRB.

---

## 5. Baselines

Three runnable baselines ship with the benchmark. All three run against
a random-init `AdaptiveSLM`; this is deliberate and mirrors the Batch A
ablation design — the benchmark measures *responsiveness* of the
conditioning path, not generation quality.

| baseline | conditioning | entry point |
|----------|--------------|-------------|
| `baseline_none` | none — zero AdaptationVector + zero user state | `benchmarks.implicit_adapt_bench.baselines.baseline_none.run_baseline_none` |
| `baseline_prompt` | prompt prefix verbalisation (the "ChatGPT way"); architectural path held neutral | `...baseline_prompt.run_baseline_prompt` |
| `baseline_cross_attention` | full I³ cross-attention conditioning | `...baseline_cross_attention.run_baseline_cross_attention` |

Any submitter who claims implicit-signal adaptation should beat
`baseline_none` on `style_match` and `cognitive_load_fidelity`, and
should ideally beat `baseline_prompt` on at least one of those two
metrics.

---

## 6. Threats to validity

### 6.1 Synthetic-data effects

The data generator is a simulator. It encodes our beliefs about how a
given archetype produces keystroke behaviour, so a system that learns
the generator's inverse will score highly for reasons that have nothing
to do with real users. Mitigations:

* The generator is public so systems can be audited against it.
* The held-out-human split, once populated, anchors at least one metric
  to real preferences.
* Submissions that materially beat the baselines will be asked to
  disclose whether they trained on the dev split.

### 6.2 Single-architecture bias

All three baselines use `i3.slm.model.AdaptiveSLM`. The style-match
metric is architecture-agnostic (it runs on plain text), but the
latency metric is not — a large GPU-only model will look bad on
`runtime_budget_compliance`, and that is by design. Submitters who
want to argue "my 70B-parameter server-side model wins" should report
numbers on the non-latency metrics and flag the latency deficit
explicitly.

### 6.3 Cultural and linguistic bias in the prompt set

The canonical prompt set is 50 English prompts drawn from the Batch A
ablation. It is culturally skewed towards Western work-and-wellbeing
contexts. A future version (v0.2) will add (a) prompts translated and
culturally adapted for CJK languages (this is a Huawei-relevant
axis), and (b) a prompt-set checksum so submissions can't be gamed by
selecting a favourable subset.

### 6.4 Rule-based metric brittleness

Rule-based metrics can be gamed by systems that target the rules
(e.g. always end with a yes/no question to max `accessibility`). The
mitigations are:

* The aggregate weight vector dilutes any single metric to at most 35 %.
* `preference_rate` (human) cannot be gamed by metric tricks.
* The moderation rules in the LEADERBOARD allow the maintainer to
  delist submissions that obviously overfit.

### 6.5 Metric coverage

The benchmark intentionally does not measure factual accuracy or
toxicity — both are covered better by existing benchmarks and would
add an unrelated quality-of-generation axis that random-init models
cannot contribute to. An `implicit_adapt_bench_ii` follow-up may add
toxicity and factuality axes once the base benchmark has stabilised.

---

## 7. Governance

### 7.1 Versioning

See the README. Metric redefinitions and split-layout changes are major
version bumps. Minor version bumps add optional metrics or larger
default splits and remain scoreable by old submissions.

### 7.2 Contribution

Contributions are welcomed through pull requests. Three classes of
contribution are accepted:

1. **Submissions** — a new leaderboard row (see the README §7).
2. **Metric extensions** — a new optional metric. Must include a
   closed-form definition, tests (see §8), and a paragraph in this
   document under §3.
3. **Data contributions** — a new split or a held-out-human rerun.
   Must include the IRB-lite protocol disclosure from §4.2.

### 7.3 Leaderboard moderation

The maintainer may delist submissions that:

* claim dev-split results but were trained on dev;
* decline to share enough runtime metadata to verify the latency
  numbers;
* exhibit obvious metric-rule overfitting as described in §6.4.

Delisting is recorded in the commit log; it is never silent.

### 7.4 Code of conduct

This benchmark follows the project-wide `CODE_OF_CONDUCT.md`. Reports
of abuse or harassment should go to the contact listed there.

---

## 8. Tests

Unit tests live at `tests/test_implicit_adapt_bench.py`. They cover:

* Pydantic validation on malformed records (feature-vector length, bad
  adaptation-vector ranges, bad runtime invariants).
* Metric math correctness via known-answer tests (perfect match → 1.0,
  worst-case → close to 0.0).
* Monotonicity of the aggregate score under metric improvements.
* Reproducibility of the data generator under fixed seed.
* Runnability of each of the three baselines against a random-init SLM.

---

## 9. References

* Bender, E. M. & Friedman, B. (2018). *Data Statements for Natural
  Language Processing: Toward Mitigating System Bias and Enabling
  Better Science.* TACL.
* Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications
  of the ACM 64 (12).
* Mitchell, M. et al. (2019). *Model Cards for Model Reporting.*
  FAT\* 2019.
* Luo, Y. et al. (2025). *PersonaLens: Evaluating Longitudinal User
  Modelling in LLM Assistants.* arXiv preprint.
* Shi, L. et al. (2025). *AlpsBench: Profile-Conditioned Assistant
  Evaluation.* arXiv preprint.
* Li, C. et al. (2024). *PersoBench: Measuring Persona-Consistent
  Response Generation.* arXiv preprint.
* Epp, C., Lippold, M., & Mandryk, R. L. (2011). *Identifying Emotional
  States using Keystroke Dynamics.* CHI 2011.
* Vizer, L. M. (2009). *Detecting Cognitive and Physical Stress through
  Typing Behavior.* CHI EA 2009.
* Zimmermann, R. (2014). *Keystroke Dynamics for Biometric
  Authentication.* survey article.
* Nosek, B. A. et al. (2018). *The preregistration revolution.*
  PNAS 115 (11).

---

## 10. Changelog

* **0.1.0** (2026-04-22) — Initial release. Five metrics, three
  baselines, four synthetic splits, IRB-lite protocol documented.
