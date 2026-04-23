# Federated Learning for I³

> *"Cross-device federated averaging of long-term profile (HarmonyOS
> Distributed Data Management; MindSpore Federated)."* —
> THE_COMPLETE_BRIEF §11

This package sketches the federated-learning path for the I³ TCN encoder.
The SLM is **deliberately excluded** — too large to ship per-client, and the
memorisation-attack surface on federated generative models is poorly
understood.

## Scope

| Component | Federated? | Rationale |
|:---|:---:|:---|
| TCN encoder (~220k params INT8) | Yes | Small; benefits from cross-user generalisation. |
| User model (EMA, Welford stats) | No | Per-user by definition; see L3 design in `docs/huawei/l1_l5_framework.md`. |
| Bandit router posterior | Per-user | Each user's Thompson posterior is local; DP-SGD sketched separately in `i3/privacy/differential_privacy.py`. |
| SLM | **No** | Too large per-client; memorisation risk. |

## Contents

- `client.py` — `I3FederatedClient` — a `flwr.client.NumPyClient` subclass
  wrapping the encoder. Supports optional Opacus DP-SGD.
- `server.py` — `I3FederatedServer` — thin `FedAvg` wrapper with I³ defaults
  (`fraction_fit=0.3`, `min_available_clients=5`). Ships a pure-numpy
  `weighted_fedavg` for test harnesses.
- `aggregator.py` — `SecureAggregator` — **pedagogical** XOR-mask additive
  secure aggregation. Not production crypto. See module docstring.

## Running the demo

```bash
# Install the future-work extras first.
poetry install --with future-work

# Start a 3-client simulation on the local machine.
python scripts/demos/federated.py --num-clients 3 --num-rounds 5
```

If `flwr` is missing, the script prints a clear install hint and exits with
status 2.

## References

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., Arcas, B. A. (2017).
  *Communication-efficient learning of deep networks from decentralized
  data.* AISTATS. — The seminal FedAvg paper.
- Kairouz, P. et al. (2021). *Advances and open problems in federated
  learning.* Foundations and Trends in Machine Learning 14(1-2). — The
  canonical survey; shapes how we think about heterogeneity, adversaries,
  and privacy budgets.
- Yousefpour, A. et al. (2021). *Opacus: User-friendly differential privacy
  library in PyTorch.* arXiv:2109.12298. — The DP-SGD wrapper used by the
  client when `noise_multiplier` is set.
- Bonawitz, K. et al. (2017). *Practical secure aggregation for privacy-
  preserving machine learning.* CCS. — The real secure-aggregation
  protocol; the sketch in `aggregator.py` is a stripped-down illustration.
- Dwork, C., Roth, A. (2014). *The algorithmic foundations of differential
  privacy.* Foundations and Trends in Theoretical Computer Science 9(3-4).
- MindSpore Federated, Huawei —
  <https://www.mindspore.cn/mindspore-federated/docs/en/master/index.html>.
  The Huawei-native counterpart to Flower; a MindSpore port of this module
  is a natural follow-on.
