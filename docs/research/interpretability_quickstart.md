# Batch B — Mechanistic Interpretability (README)

One-page overview of the interpretability study committed in Batch B of
the Implicit Interaction Intelligence (I³) advancement plan.

## What is here

Three new analyses, applied to I³'s cross-attention conditioning
pathway:

| Analysis | File | What it measures |
|---|---|---|
| Activation patching (ROME-style) | `i3/interpretability/activation_patching.py` | Symmetric KL between the clean and component-patched next-token distributions; identifies sub-modules on the causal path. |
| Linear probing | `i3/interpretability/probing_classifiers.py` | Held-out R² of a bias-free linear probe trained to recover each adaptation dimension from each transformer layer's pooled hidden state. |
| Attention-circuit analysis | `i3/interpretability/attention_circuits.py` | Per-head max-weight & entropy profiles; flags "conditioning specialist" heads (max weight > 0.6 on a majority of positions). |

The study orchestrator and paper-style write-up live at:

- `scripts/experiments/interpretability_study.py` — CLI that runs all three
  analyses on a single random-init `AdaptiveSLM` and emits a Markdown
  report plus a JSON sibling.
- `docs/research/mechanistic_interpretability.md` — abstract,
  related-work, method, placeholder result tables, synthesis,
  threats to validity, future work.
- `tests/test_interpretability_circuits.py` — 10+ unit tests covering
  hook management, probe recovery of a pass-through dimension,
  attention-pattern shape invariants, and specialist detection on a
  hand-crafted synthetic circuit.

## How to run

```bash
# From the repo root, with the i3 package importable:
python scripts/experiments/interpretability_study.py \
    --seed 42 --n-prompts 20 \
    --out reports/interpretability_study.md
```

The script prints the paths of the written Markdown + JSON files.
Typical wall time on a 2023-class laptop CPU: **≈2–5 seconds** for the
default small-model configuration.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--seed` | `42` | Global RNG seed. |
| `--n-prompts` | `20` | Number of prompts for the circuit-analysis average. |
| `--n-probe-samples` | `64` | Examples per probed adaptation dimension. |
| `--d-model` | `64` | Hidden dim of the random-init `AdaptiveSLM`. |
| `--n-layers` | `4` | Number of transformer blocks. |
| `--n-heads` | `4` | Self-attention heads per layer. |
| `--n-cross-heads` | `2` | Cross-attention heads per layer. |
| `--seq-len` | `16` | Prompt length (tokens). |
| `--out` | timestamped under `reports/` | Output Markdown path. |

If `matplotlib` is available the script saves two PNG figures
(`*_patching.png`, `*_probes.png`) next to the Markdown report. Without
matplotlib it falls back to ASCII heat-maps embedded in fenced code
blocks so the report is always self-contained. Similarly, if `pandas`
is available the JSON sibling gains a rendered selectivity table;
without pandas the dict-of-dicts fallback is used.

## How to extend

The three analyses each expose a small public API:

```python
from i3.interpretability.activation_patching import trace_causal_effect
from i3.interpretability.probing_classifiers import ProbingSuite
from i3.interpretability.attention_circuits import (
    extract_attention_patterns, identify_conditioning_specialists,
    summarise_circuit,
)
```

To run them against a trained checkpoint, load the state dict into an
`AdaptiveSLM` before passing the model into each function; no other
code changes are required. The study orchestrator
(`scripts/experiments/interpretability_study.py`) pins the commit SHA via
`git rev-parse HEAD` so every run is reproducible.

## New dependencies

**Zero.** Both `matplotlib` and `pandas` are soft-imported; the study
runs end-to-end on the existing I³ dependency set (torch + numpy).

## Tests

```bash
pytest tests/test_interpretability_circuits.py -v
```

All 10+ tests run on CPU and complete in under ~5 seconds.

## References

Meng et al. 2022 (ROME); Vig et al. 2020 (Causal Mediation);
Alain & Bengio 2016 (Linear probes); Hewitt & Liang 2019 (Control
tasks); Elhage et al. 2021 (Transformer Circuits); Olsson et al. 2022
(Induction Heads); Geiger et al. 2024 (Causal Abstraction).
