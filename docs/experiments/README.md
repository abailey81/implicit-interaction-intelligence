# `docs/experiments/` — empirical studies

This directory contains pre-registrations, protocols, and write-ups for
empirical studies of the Implicit Interaction Intelligence (I³) system.

| File                        | Purpose                                                                 |
|:----------------------------|:------------------------------------------------------------------------|
| `preregistration.md`        | Pre-registration for Batch A (cross-attention conditioning ablation).   |
| `ablation_report.md`        | Paper-style write-up of the Batch A results.                            |
| `README.md`                 | This index.                                                             |

## Running the Batch A study

The Batch A study compares three conditioning regimes — no conditioning,
prompt-based conditioning, and architectural cross-attention — on a
random-init (or optionally checkpoint-loaded) `AdaptiveSLM`. See
[`preregistration.md`](./preregistration.md) for the full protocol.

### Quick start

```bash
# Default: random-init model, seed 42, all 50 prompts, reports/ output.
python scripts/experiments/ablation_study.py --verbose

# Choose custom paths.
python scripts/experiments/ablation_study.py \
  --seed 42 --n-prompts 50 \
  --out reports/ablation.json \
  --out-md reports/ablation.md

# Optional: load a trained checkpoint via environment variable.
I3_CHECKPOINT_PATH=checkpoints/slm/slm.pt \
  python scripts/experiments/ablation_study.py --verbose
```

The CLI emits:

- A JSON dump with the full per-pair dataframe plus summary statistics.
- A Markdown report with tables for KL means + bootstrap CIs, style
  fidelity, latency P50/P95/P99, pairwise Cohen's *d*, and paired-sign
  test p-values. The report is stamped with the repository HEAD SHA so
  it can be matched to exact code.

### Components

- Experiment class: [`i3/eval/ablation_experiment.py`](../../i3/eval/ablation_experiment.py)
- Statistical helpers: [`i3/eval/ablation_statistics.py`](../../i3/eval/ablation_statistics.py)
- CLI driver: [`scripts/experiments/ablation_study.py`](../../scripts/experiments/ablation_study.py)
- Unit tests: [`tests/test_ablation_statistics.py`](../../tests/test_ablation_statistics.py)

### Reproducibility checklist

- [x] Pre-registration exists and is committed before any analysis run.
- [x] Global seed fixed at 42; all other random sources enumerated in
      the pre-registration.
- [x] Test set (50 prompts, 8 archetypes) hard-coded inside the
      experiment module rather than a mutable config file.
- [x] Bootstrap resample count fixed at 10 000.
- [x] Analysis code hash (git SHA) embedded in every report.
- [x] Exclusions policy: none — all 1 200 runs contribute to all stats.

## Adding a future experiment

Subsequent batches (B, C, D, E) should follow the same layout: one
Markdown pre-registration, one paper-style report, and an entry in the
table at the top of this file. Keep prompts, archetypes, and any other
"data" inside the Python experiment module so that the pre-registration
can commit to an exact grid without a drift-prone YAML file.
