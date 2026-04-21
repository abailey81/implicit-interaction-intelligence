"""Per-subgroup fairness evaluation for I³.

Implements the future-work item in THE_COMPLETE_BRIEF §11
("Per-subgroup fairness evaluation surfacing confidence"): for each of the
8 Epp/Vizer/Zimmermann archetypes, measure whether the adaptation layer
behaves equitably, and report the disparity with bootstrap confidence
intervals (Efron 1979).

The modules are intentionally lightweight — the interesting work lives in
:mod:`i3.fairness.subgroup_metrics` (metric computation) and
:mod:`i3.fairness.confidence_intervals` (bootstrap CI helpers).
"""

from __future__ import annotations

from i3.fairness.biometric_bias import BiometricBiasReport, compute_biometric_bias
from i3.fairness.confidence_intervals import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_mean_ci,
)
from i3.fairness.subgroup_metrics import (
    ArchetypeMetrics,
    FairnessReport,
    compute_per_archetype_adaptation_bias,
)

__all__ = [
    "ArchetypeMetrics",
    "BiometricBiasReport",
    "BootstrapResult",
    "FairnessReport",
    "bootstrap_ci",
    "bootstrap_mean_ci",
    "compute_biometric_bias",
    "compute_per_archetype_adaptation_bias",
]
