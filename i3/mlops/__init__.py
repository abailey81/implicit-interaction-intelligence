"""MLOps infrastructure for Implicit Interaction Intelligence (I3).

This package provides the production-grade MLOps plumbing used by the I3
training, evaluation and deployment pipelines:

* :mod:`i3.mlops.tracking` -- MLflow experiment tracking wrapper that
  soft-imports MLflow and no-ops if it is not installed.
* :mod:`i3.mlops.registry` -- Local-filesystem model registry with
  optional MLflow + W&B mirrors.
* :mod:`i3.mlops.checkpoint` -- SHA-256 + optional cosign-style signature
  integrity checks and a safe-load wrapper around ``torch.load``.
* :mod:`i3.mlops.export` -- ONNX export entry point that dispatches to
  per-model exporters.

All imports are deliberately lazy so that the rest of the I3 codebase
never pays an import-time cost for optional MLOps dependencies.
"""

from __future__ import annotations

from i3.mlops.checkpoint import ChecksumError, load_verified, save_with_hash
from i3.mlops.registry import ModelRegistry, RegistryEntry
from i3.mlops.tracking import ExperimentTracker

__all__ = [
    "ChecksumError",
    "ExperimentTracker",
    "ModelRegistry",
    "RegistryEntry",
    "load_verified",
    "save_with_hash",
]
