"""Huawei integration package for Implicit Interaction Intelligence (I³).

This package is a **leaf integration surface**: nothing in the core I³
pipeline imports it. It collects reference implementations and
configuration artefacts that describe how I³ plugs into Huawei's
on-device AI ecosystem — HarmonyOS 6's Harmony Multi-Agent Framework
(HMAF), the Kirin NPU family, and Huawei's L1–L5 device intelligence
ladder.

Contents:
    - :mod:`i3.huawei.hmaf_adapter` — reference HMAF agent-protocol adapter.
    - :mod:`i3.huawei.kirin_targets` — Pydantic v2 target-device models and
      a ``select_deployment_profile()`` helper.
    - :mod:`i3.huawei.executorch_hooks` — stubs showing where the
      ``torch.export → to_edge → to_executorch → .pte`` pipeline hooks in.

See :doc:`docs/huawei/README.md` for the integration dossier.
"""

from __future__ import annotations

__all__ = [
    "executorch_hooks",
    "hmaf_adapter",
    "kirin_targets",
]
