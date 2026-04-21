"""Per-archetype FAR / FRR computation for the keystroke-biometric module.

A keystroke biometric is only useful if it performs equitably across
archetypes.  If the false-reject rate spikes for the *motor_difficulty*
archetype — users with low typing speed and high correction rate — the
biometric is systematically locking out the exact population the
accessibility adapter is supposed to help.  That is the failure mode this
module surfaces.

The biometric module itself lives in ``i3/biometric/`` (owned by another
agent); this file is the **fairness evaluation** of whatever biometric
that package produces.  It depends only on a duck-typed
:class:`BiometricScoreProvider` interface.

References
----------
* Mansfield, A. J., Wayman, J. L. (2002). *Best practices in testing and
  reporting performance of biometric devices.*  NIST SP 500-245.  The
  canonical source for FAR / FRR definitions.
* Epp, C., Lippold, M., Mandryk, R. L. (2011).  *Identifying emotional
  states using keystroke dynamics.*  CHI.
* Barocas, S., Hardt, M., Narayanan, A. (2019).  *Fairness and Machine
  Learning.*  fairmlbook.org — the framework for equalised-odds
  subgroup analysis applied here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from i3.fairness.confidence_intervals import bootstrap_ci
from i3.fairness.subgroup_metrics import EPP_VIZER_ZIMMERMANN_ARCHETYPES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed biometric provider
# ---------------------------------------------------------------------------

@runtime_checkable
class BiometricScoreProvider(Protocol):
    """Interface every biometric module must expose for fairness evaluation.

    Returns a similarity score in ``[0, 1]`` where higher means more likely
    to be the claimed user.  A score ≥ ``threshold`` triggers an *accept*
    decision.
    """

    def score(self, sample: np.ndarray, claimed_user_id: str) -> float: ...


# ---------------------------------------------------------------------------
# Evaluation datum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiometricTrial:
    """A single biometric verification attempt.

    Attributes:
        sample: Feature vector (shape depends on the biometric; 32-dim for
            keystroke dynamics).
        claimed_user_id: The user the sample claims to be.
        true_user_id: The user the sample *actually* came from.
        archetype: The user's archetype at the time of capture.
    """

    sample: np.ndarray
    claimed_user_id: str
    true_user_id: str
    archetype: str


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

class ArchetypeBiometricPerformance(BaseModel):
    """Per-archetype FAR / FRR summary.

    Attributes:
        archetype: Archetype label.
        n_genuine: Count of genuine attempts (same claimed / true user).
        n_impostor: Count of impostor attempts (different claimed / true).
        far: False-accept rate (impostors wrongly accepted).
        frr: False-reject rate (genuines wrongly rejected).
        far_ci: 95% CI on FAR (lower, upper).
        frr_ci: 95% CI on FRR (lower, upper).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    archetype: str
    n_genuine: int
    n_impostor: int
    far: float
    frr: float
    far_ci: tuple[float, float]
    frr_ci: tuple[float, float]


class BiometricBiasReport(BaseModel):
    """Aggregate biometric fairness report.

    Attributes:
        threshold: Acceptance threshold used by the biometric.
        per_archetype: List of :class:`ArchetypeBiometricPerformance`.
        far_disparity: ``max - min`` of FAR across archetypes.
        frr_disparity: ``max - min`` of FRR across archetypes.
        flagged_archetypes: Archetypes whose FAR or FRR is more than
            ``disparity_threshold`` above the median.
        disparity_threshold: The threshold used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float
    per_archetype: list[ArchetypeBiometricPerformance] = Field(default_factory=list)
    far_disparity: float = 0.0
    frr_disparity: float = 0.0
    flagged_archetypes: list[str] = Field(default_factory=list)
    disparity_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def compute_biometric_bias(
    provider: BiometricScoreProvider,
    trials: list[BiometricTrial],
    *,
    threshold: float = 0.5,
    archetypes: list[str] | None = None,
    disparity_threshold: float = 0.05,
    num_resamples: int = 1000,
    seed: int | None = None,
) -> BiometricBiasReport:
    """Compute per-archetype FAR / FRR for a biometric scorer.

    Args:
        provider: Any object exposing ``score(sample, claimed_user_id)``.
        trials: List of verification trials.  At least one genuine and
            one impostor attempt per archetype is required for that
            archetype's CI to be non-degenerate.
        threshold: Acceptance threshold.  Scores ``>= threshold`` accept.
        archetypes: Archetype whitelist.  Defaults to the 8 canonical
            Epp/Vizer/Zimmermann labels.
        disparity_threshold: Archetypes whose FAR or FRR exceeds the median
            by more than this are flagged.
        num_resamples: Bootstrap resamples for the CIs.
        seed: Optional RNG seed.

    Returns:
        A :class:`BiometricBiasReport`.
    """
    if archetypes is None:
        archetypes = list(EPP_VIZER_ZIMMERMANN_ARCHETYPES)

    per_archetype: list[ArchetypeBiometricPerformance] = []

    for archetype in archetypes:
        sub = [t for t in trials if t.archetype == archetype]
        genuine_flags: list[int] = []   # 1 => wrongly rejected
        impostor_flags: list[int] = []  # 1 => wrongly accepted
        for trial in sub:
            accepted = float(provider.score(trial.sample, trial.claimed_user_id)) >= threshold
            if trial.claimed_user_id == trial.true_user_id:
                genuine_flags.append(0 if accepted else 1)
            else:
                impostor_flags.append(1 if accepted else 0)

        def _rate_with_ci(flags: list[int]) -> tuple[float, tuple[float, float]]:
            if not flags:
                return 0.0, (0.0, 0.0)
            arr = np.asarray(flags, dtype=np.float64)
            rate = float(arr.mean())
            ci = bootstrap_ci(
                arr,
                statistic=lambda x: float(x.mean()),
                num_resamples=num_resamples,
                level=0.95,
                seed=seed,
            )
            return rate, (ci.lower, ci.upper)

        frr, frr_ci = _rate_with_ci(genuine_flags)
        far, far_ci = _rate_with_ci(impostor_flags)
        per_archetype.append(
            ArchetypeBiometricPerformance(
                archetype=archetype,
                n_genuine=len(genuine_flags),
                n_impostor=len(impostor_flags),
                far=far,
                frr=frr,
                far_ci=far_ci,
                frr_ci=frr_ci,
            )
        )

    fars = [p.far for p in per_archetype if (p.n_impostor > 0)]
    frrs = [p.frr for p in per_archetype if (p.n_genuine > 0)]

    far_disp = float(max(fars) - min(fars)) if len(fars) > 1 else 0.0
    frr_disp = float(max(frrs) - min(frrs)) if len(frrs) > 1 else 0.0

    median_far = float(np.median(fars)) if fars else 0.0
    median_frr = float(np.median(frrs)) if frrs else 0.0
    flagged: list[str] = []
    for p in per_archetype:
        if p.far - median_far > disparity_threshold:
            flagged.append(p.archetype)
        elif p.frr - median_frr > disparity_threshold:
            flagged.append(p.archetype)

    return BiometricBiasReport(
        threshold=threshold,
        per_archetype=per_archetype,
        far_disparity=far_disp,
        frr_disparity=frr_disp,
        flagged_archetypes=flagged,
        disparity_threshold=disparity_threshold,
    )
