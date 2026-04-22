# ImplicitAdaptBench — leaderboard

Rankings are by the aggregate scalar defined in
`benchmarks/implicit_adapt_bench/metrics.py::aggregate_score` with the
default weights listed in the README. All three reference baselines are
reserved placeholder rows; fill them in by running

```bash
python scripts/run_implicit_adapt_bench.py --run-baselines
```

and copying the per-metric numbers from
`reports/implicit_adapt_bench_<ts>/<method>.score.json`.

---

## Dev split (public)

| rank | method | aggregate | style_match | cognitive_load_fidelity | accessibility_appropriateness | preference_rate | runtime_p95_ms | bench version | submitted |
|-----:|--------|----------:|------------:|------------------------:|------------------------------:|----------------:|---------------:|:-------------:|:---------:|
|  —   | `baseline_none` (placeholder) |   —   |   —   |   —   |   —   |   —   |   —   | 0.1.0 | reserved |
|  —   | `baseline_prompt` (placeholder) |   —   |   —   |   —   |   —   |   —   |   —   | 0.1.0 | reserved |
|  —   | `baseline_cross_attention` (placeholder) |   —   |   —   |   —   |   —   |   —   |   —   | 0.1.0 | reserved |

## Test split (held back)

Populated only by the maintainer at each benchmark version bump.

| rank | method | aggregate | bench version | notes |
|-----:|--------|----------:|:-------------:|:------|
|      |        |           |               |       |

## Held-out human-preference split

Populated only after an IRB-lite preference-collection run. See
`docs/research/implicit_adapt_bench.md` §5 for the protocol.

| rank | method | preference_rate | n_annotators | bench version | notes |
|-----:|--------|----------------:|-------------:|:-------------:|:------|
|      |        |                 |              |               |       |

---

## How to submit

1. Fork this repository.
2. Run your method to produce a submission JSONL following the schema in
   `benchmarks/implicit_adapt_bench/data_schema.py::BenchmarkSubmission`.
3. Add your file under `submissions/<method_name>/<timestamp>.jsonl`.
4. Add a short model card at
   `submissions/<method_name>/<timestamp>.card.md` (intended use, data,
   compute, failure modes).
5. Open a pull request titled
   `ImplicitAdaptBench submission: <method_name>`. CI runs
   `scripts/run_implicit_adapt_bench.py --score` and comments the
   per-metric breakdown. The maintainer reviews, merges, and appends the
   row to this leaderboard.

## Moderation rules

* Only one submission per method per week to prevent leaderboard spam.
* Submissions must disclose whether any training data overlaps with the
  public dev split. Leaderboard entries that trained on dev are listed
  separately.
* Submissions that claim preference-rate improvements must provide raw
  annotator logs or opt out of the held-out row.
* The maintainer reserves the right to remove submissions found to
  overfit on metric artefacts (see the threats-to-validity section of
  `docs/research/implicit_adapt_bench.md`).
