# Closed-Loop Persona Evaluation -- Quickstart

One-page quickstart for the persona-simulation harness that scores the
I3 pipeline's user-modelling correctness against eight canonical
synthetic users.

## What it does

For each of eight HCI personas (fresh user, fatigued developer, motor
impairment, second-language speaker, high cognitive load, dyslexia,
energetic user, low vision) it:

1. Runs `N_sessions x N_messages` simulated dialogue turns through the
   full I3 pipeline.
2. Compares the inferred `AdaptationVector` against the persona's
   ground-truth `expected_adaptation`.
3. Reports 1-NN persona recovery, per-message L2 error, convergence
   speed, a confusion matrix, and a router-bias check, all with 95 %
   bootstrap confidence intervals.

## Run it

```bash
python scripts/run_closed_loop_eval.py \
  --config configs/default.yaml \
  --n-sessions 5 \
  --n-messages 15 \
  --seed 42 \
  --out reports/closed_loop_eval.json \
  --out-md reports/closed_loop_eval.md
```

Useful flags:

* `--personas fresh_user,fatigued_developer,motor_impaired_user` --
  restrict to a subset (comma-separated `HCIPersona.name` values).
* `--threshold 0.3` -- L2-error threshold for the convergence metric.
* `-v` -- verbose logging to stderr.

## Read the output

* `reports/closed_loop_eval.json` -- machine-readable dump; schema
  version `1`. Top-level keys: `run_metadata`, `summary`,
  `per_message_records`.
* `reports/closed_loop_eval.md` -- 7-section human-readable report:
  1. Methodology
  2. Per-persona recovery rates (with CIs)
  3. Per-persona adaptation error by message (ASCII sparklines)
  4. Convergence speeds
  5. Persona confusion matrix (table + ASCII heatmap)
  6. Router-bias check (accessibility vs. baseline)
  7. Threats to validity

## Code entry points

| File | Role |
|---|---|
| `i3/eval/simulation/personas.py` | The 8 canonical `HCIPersona` instances. Do not modify without bumping the library version. |
| `i3/eval/simulation/user_simulator.py` | Deterministic keystroke + text generation. |
| `i3/eval/simulation/closed_loop.py` | `ClosedLoopEvaluator` + `ClosedLoopResult`. |
| `scripts/run_closed_loop_eval.py` | CLI runner (JSON + Markdown output). |
| `docs/research/closed_loop_evaluation.md` | Full paper-style writeup. |
| `tests/test_simulation_personas.py` | Persona + simulator unit tests. |
| `tests/test_closed_loop_evaluator.py` | Evaluator tests with a mocked pipeline. |

## Determinism

Every number in the report is reproducible given:

* `--seed` (controls both the per-session `UserSimulator` seeds and
  the bootstrap RNG).
* The commit SHA (pinned in `run_metadata.git_sha`).
* The YAML config (pinned in `run_metadata.config`).

If two runs with identical triples produce different numbers, that is a
bug -- please file one.
