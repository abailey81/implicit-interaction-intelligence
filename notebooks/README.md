# I3 Teaching Notebooks

These notebooks are teaching artefacts, not part of the runtime. They accompany the Implicit Interaction Intelligence (I3) package and walk through the theoretical foundations and reference implementations behind each major component.

## Running

```bash
poetry install --with dev && poetry run jupyter lab notebooks/
```

All notebooks are self-contained. They depend only on the `i3.*` packages plus `numpy`, `matplotlib`, and `torch`, all of which are already listed as project dependencies. No external datasets are required.

## Contents

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_perception_keystroke_dynamics.ipynb` | The 32-dim `InteractionFeatureVector` and its four feature groups |
| 02 | `02_tcn_encoder_from_scratch.ipynb`      | Temporal Convolutional Network encoder with NT-Xent objective     |
| 03 | `03_three_timescale_user_model.ipynb`    | Welford online stats and EMAs across three timescales             |
| 04 | `04_cross_attention_conditioning_centrepiece.ipynb` | Conditioning projector + KL-divergence sensitivity test |
| 05 | `05_contextual_thompson_sampling.ipynb`  | Bayesian logistic regression bandit with Laplace approximation    |
| 06 | `06_privacy_by_architecture.ipynb`       | No-raw-text guarantee, PII sanitizer, Fernet round-trip           |
| 07 | `07_edge_profiling_and_quantisation.ipynb` | Parameter counts, INT8 sizing, Kirin extrapolation              |

Author: Tamer Atesyakar
