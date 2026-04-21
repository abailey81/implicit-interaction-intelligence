"""On-device ("edge") export pipelines for I3 models.

This package hosts ExecuTorch (``.pte``) export pipelines for the
Adaptive SLM and the TCN encoder. ExecuTorch is the PyTorch 2026
on-device runtime (https://pytorch.org/executorch/) and is the
recommended deployment target for Huawei Kirin SoCs, Apple Silicon,
Android NNAPI, and embedded Linux.

Public API:

* :func:`i3.edge.executorch_export.export_slm_to_executorch` — export
  the SLM to ``.pte``.
* :func:`i3.edge.tcn_executorch_export.export_tcn_to_executorch` —
  export the TCN encoder to ``.pte``.

All submodules soft-import ``executorch`` so the package remains
importable when the dependency is absent.
"""

from __future__ import annotations

__all__: list[str] = []
