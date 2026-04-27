"""On-device per-biometric LoRA personalisation (I3 flagship novelty).

This package contains the per-user low-rank adaptation residual that
layers onto the base :class:`AdaptationVector` produced by
:class:`i3.adaptation.AdaptationController`.  The adapter is keyed by a
SHA-256 hash of the typing-biometric template embedding and trained
online from the A/B preference picker — never federated, never leaves
the device.

Public surface::

    from i3.personalisation import (
        AdaptationLoRA,
        LoRAUpdate,
        PersonalisationManager,
    )

Cites Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language
Models" (arXiv:2106.09685) and Houlsby et al. 2019 "Parameter-Efficient
Transfer Learning for NLP" (ICML 2019) — see the docstring of
:mod:`i3.personalisation.lora_adapter` for full citations.
"""

from i3.personalisation.lora_adapter import (
    AdaptationLoRA,
    LoRAUpdate,
    PersonalisationManager,
)

__all__ = [
    "AdaptationLoRA",
    "LoRAUpdate",
    "PersonalisationManager",
]
