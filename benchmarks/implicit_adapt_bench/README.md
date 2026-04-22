# ImplicitAdaptBench

A benchmark for adaptive response generation from **implicit behavioural
signals** — keystroke dynamics, linguistic complexity, baseline deviations
— rather than from an explicit user profile.

> Full paper-style spec: `docs/research/implicit_adapt_bench.md`.
> Leaderboard: `LEADERBOARD.md`.

---

## 1. Motivation

The 2025–2026 personalisation-benchmark wave (PersonaLens 2025, AlpsBench
2025, PersoBench 2024) measures adaptation from **explicit** user profiles:
an agent is told the user is a 34-year-old nurse who prefers concise
answers and is then judged on how faithfully the response matches that
profile. Real deployed assistants rarely receive such a profile. Instead,
they must **infer** the user's state from interaction behaviour —
keystroke timing, correction rate, message length, verbosity trends —
and adapt in real time.

No public benchmark currently measures this adaptation mode. ImplicitAdaptBench
fills that gap with:

* **4 synthetic splits** (train / dev / test / held-out human) with
  paired behavioural windows and reference-style targets;
* **5 rule-based metrics** that do not depend on a trained judge model;
* **3 runnable reference baselines** (including a full I³ cross-attention
  baseline);
* **a leaderboard submission format** and moderation policy.

The benchmark is intentionally rule-based and random-init-friendly: it
measures *responsiveness* of the generator to the implicit signal, not
generation quality at scale.

---

## 2. Task definition

### Input

For each record the submitter receives:

* ``behavioural_window`` — a ``BehaviouralWindow`` combining:
  - ``feature_vector`` — the canonical 32-dim
    ``InteractionFeatureVector`` (order matches
    ``i3.interaction.types.FEATURE_NAMES``);
  - ``raw_keystroke_intervals_ms`` — the last message's IKI sequence;
  - ``baseline_deviation_metrics`` — sparse z-score deviations from the
    user's warm-up baseline.
* ``prompt`` — the text the user has just submitted.

### Output

A :class:`BenchmarkSubmission` with:

* ``record_id`` — matches the gold record;
* ``generated_text`` — the candidate response (free text, UTF-8);
* ``method_name`` — short identifier of the submitting method;
* ``runtime_ms_p50`` and ``runtime_ms_p95`` — per-record forward latency.

### Not provided to the submitter

* ``target_archetype`` / ``target_adaptation_vector`` — these are held on
  the scorer side.
* ``human_preference_score`` — only the held-out split has it, and only
  the scorer can see it.

---

## 3. Data format

Each split is a JSONL file with one
:class:`benchmarks.implicit_adapt_bench.data_schema.BenchmarkRecord` per
line:

```json
{
  "record_id": "dev-0007",
  "behavioural_window": {
    "feature_vector": [0.52, 0.48, ...],
    "raw_keystroke_intervals_ms": [134.2, 220.1, ...],
    "baseline_deviation_metrics": {"iki_deviation": 0.12, ...}
  },
  "prompt": "Explain gradient descent to a beginner.",
  "target_archetype": "high_load_technical",
  "target_adaptation_vector": [0.9, 0.8, 0.8, 0.2, 0.8, 0.7, 0.1, 0.0],
  "reference_style_label": "objective_formal_long",
  "reference_length_bucket": "long",
  "reference_formality_bucket": "formal",
  "human_preference_score": null
}
```

The 32-dim ``feature_vector`` order is the dataclass-field order of
``i3.interaction.types.InteractionFeatureVector``.

### Splits

| split | purpose | size (default generator) |
|-------|---------|--------------------------|
| ``train`` | method training / fitting, if any | 8 × ``n_per_archetype`` |
| ``dev`` | public development, leaderboard submission dev score | 8 × ``n_per_archetype`` |
| ``test`` | held-back test split | 8 × ``n_per_archetype`` |
| ``held_out_human`` | human-preference annotated micro-set | 8 × ``n_per_archetype`` |

The default generator uses ``n_per_archetype = 4`` (32 records per split).
Submitters may request larger splits by invoking
``generate_synthetic_split`` directly; the leaderboard fixes a size at
submission time.

---

## 4. Metrics

All metrics return a scalar in ``[0, 1]`` (higher is better). Formal
definitions live in ``docs/research/implicit_adapt_bench.md`` §3; a short
form follows.

1. **``style_match``** — equal-weight mean of three sub-scores measuring
   formality match, length-bucket match, and tone-bucket match against
   the structured ``reference_style_label``.
2. **``cognitive_load_fidelity``** — whether the Flesch-Kincaid grade of
   the response falls in the target band for the record's
   ``cognitive_load`` target (``0.0`` → FK 3–6, ``0.5`` → FK 6–10,
   ``1.0`` → FK 10–14).
3. **``accessibility_appropriateness``** — when
   ``accessibility > 0.7``, tests short-sentence ratio, idiom absence,
   and presence of a yes/no confirmation question.
4. **``preference_rate``** — mean human-preference score over the
   held-out human split (``None``-scored records are skipped).
5. **``runtime_budget_compliance``** — fraction of submissions whose
   ``runtime_ms_p95`` is under the 200 ms budget.

The aggregate is a weighted mean with default weights

```python
{
    "style_match": 0.35,
    "cognitive_load_fidelity": 0.25,
    "accessibility_appropriateness": 0.15,
    "preference_rate": 0.15,
    "runtime_budget_compliance": 0.10,
}
```

---

## 5. Baselines

Three runnable baselines are shipped under
``benchmarks/implicit_adapt_bench/baselines/``:

| baseline | description | entry point |
|----------|-------------|-------------|
| ``baseline_none`` | No conditioning; zero AdaptationVector + zero user state. | ``run_baseline_none`` |
| ``baseline_prompt`` | Verbalises the AdaptationVector into a ``[System: ...]`` prompt prefix. | ``run_baseline_prompt`` |
| ``baseline_cross_attention`` | Full I³ cross-attention conditioning path. | ``run_baseline_cross_attention`` |

All three run against a random-init ``AdaptiveSLM``; the benchmark
measures *responsiveness*, not quality.

Run them with:

```bash
python scripts/run_implicit_adapt_bench.py --run-baselines
```

---

## 6. Leaderboard

See ``LEADERBOARD.md``. Placeholder rows are reserved for the three
reference baselines; community submissions are appended below.

---

## 7. Submission format

A submission is a JSONL file with one
:class:`benchmarks.implicit_adapt_bench.data_schema.BenchmarkSubmission`
per line. Submit by opening a pull request that adds:

1. ``submissions/<method_name>/<timestamp>.jsonl`` — the submission file.
2. ``submissions/<method_name>/<timestamp>.card.md`` — a short model
   card (training data, compute, intended use, known failure modes).

The CI pipeline runs ``scripts/run_implicit_adapt_bench.py --score``
against the agreed split and posts the per-metric breakdown as a PR
comment.

### Local scoring

```bash
python scripts/run_implicit_adapt_bench.py \
    --score submissions/my_method/20260422T090000Z.jsonl \
    --records-path benchmarks/implicit_adapt_bench/data/dev.jsonl
```

---

## 8. Versioning

The benchmark is versioned by the ``__version__`` constant in
``benchmarks/implicit_adapt_bench/__init__.py``. Breaking changes (metric
redefinitions, split layout, archetype set) bump the major version.
Backward-compatible additions (new optional metrics, larger default
splits) bump the minor version. The leaderboard retains a version column
so old submissions remain interpretable.

---

## 9. Citation

```bibtex
@misc{implicitadaptbench2026,
  title        = {ImplicitAdaptBench: A Benchmark for Adaptive Generation
                   from Implicit Behavioural Signals},
  author       = {Atesyakar, Tamer},
  year         = {2026},
  howpublished = {Implicit Interaction Intelligence (I\^3) project},
  note         = {Version 0.1.0. See
                   docs/research/implicit\_adapt\_bench.md.},
}
```
