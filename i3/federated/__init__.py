"""Federated-learning sketch for the I³ long-term profile.

This package stubs out the federated-averaging path for the TCN encoder
only.  The SLM is **explicitly excluded** — its parameter count and
generative risk profile make it a poor federated target.  The encoder, at
~220 k parameters after quantisation, is well within the sweet spot for
FedAvg-class methods.

References
----------
* McMahan, H. B., Moore, E., Ramage, D., Hampson, S., Arcas, B. A. (2017).
  *Communication-efficient learning of deep networks from decentralized
  data.*  AISTATS.
* Kairouz, P. et al. (2021). *Advances and open problems in federated
  learning.*  Foundations and Trends in Machine Learning 14(1-2).
* Yousefpour, A. et al. (2021). *Opacus: User-friendly differential privacy
  library in PyTorch.*  arXiv:2109.12298.
* MindSpore Federated, Huawei:
  https://www.mindspore.cn/mindspore-federated/docs/en/master/index.html
"""

from __future__ import annotations

from i3.federated.aggregator import SecureAggregator
from i3.federated.client import I3FederatedClient
from i3.federated.server import I3FederatedServer, weighted_fedavg

__all__ = [
    "I3FederatedClient",
    "I3FederatedServer",
    "SecureAggregator",
    "weighted_fedavg",
]
