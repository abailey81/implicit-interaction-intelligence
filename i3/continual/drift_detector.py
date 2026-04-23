"""Concept-drift detection on the encoder's embedding-error distribution.

The continual-learning loop needs a trigger: when should the pipeline
call :meth:`~i3.continual.ewc.ElasticWeightConsolidation.consolidate`?
Kirkpatrick 2017 assumes discrete task boundaries; the I3 deployment
sees a *stream* of interactions where task boundaries must be inferred.
We infer them via two complementary detectors:

1. :class:`ConceptDriftDetector` -- Bifet & Gavaldà 2007 ADWIN
   ("Learning from Time-Changing Data with Adaptive Windowing", SDM)
   style adaptive windowing on the encoder error stream. ADWIN keeps a
   variable-length window of recent observations and, whenever the
   window can be split into two sub-windows with significantly different
   means, it drops the older half. A drop event is raised as drift.
2. :func:`population_stability_index` -- PSI, a classical statistical
   distance used in credit scoring (Siddiqi 2006) for detecting input-
   distribution shift. PSI > 0.2 is "moderate drift"; > 0.25 is commonly
   treated as "significant".

References
----------
* Bifet, A., & Gavaldà, R. (2007). "Learning from Time-Changing Data
  with Adaptive Windowing". *SDM*, 443-448.
* Gama, J., et al. (2014). "A Survey on Concept Drift Adaptation".
  *ACM Comput. Surv.* 46 (4): 44.
* Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# SEC: ADWIN confidence parameter; Bifet & Gavaldà 2007 typically use
# 0.002 for responsive detection on noisy streams.
_DEFAULT_DELTA: float = 0.002

# SEC: Minimum sub-window size for an ADWIN split. Prevents triggering
# on single-point outliers.
_DEFAULT_MIN_SUB_WINDOW: int = 8

# SEC: Maximum retained history; caps memory in an unbounded stream.
_DEFAULT_MAX_WINDOW: int = 2048

# SEC: Cooldown in observations between successive drift alarms so the
# callback is not hammered while the window rebuilds.
_DEFAULT_COOLDOWN: int = 32


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriftDetectionResult:
    """Summary returned by :meth:`ConceptDriftDetector.update`.

    Attributes:
        drift_detected: ``True`` iff the current update raised an alarm.
        window_size: Size of the retained window after any truncation.
        old_mean: Mean of the dropped (older) sub-window when drift
            fired; ``None`` otherwise.
        new_mean: Mean of the retained (newer) sub-window when drift
            fired; ``None`` otherwise.
        cut_point: Index of the split within the pre-trim window.
        psi: Optional PSI score against the baseline, when a baseline
            was supplied.
    """

    drift_detected: bool
    window_size: int
    old_mean: float | None = None
    new_mean: float | None = None
    cut_point: int | None = None
    psi: float | None = None


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------


def population_stability_index(
    baseline_dist: Sequence[float] | torch.Tensor,
    current_dist: Sequence[float] | torch.Tensor,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute the Population Stability Index between two samples.

    PSI (Siddiqi 2006) is defined as::

        PSI = Σ_i (p_i − q_i) · log(p_i / q_i)

    where ``p_i`` and ``q_i`` are the bin frequencies of the baseline
    and current distributions respectively. Values are interpreted as:

    ========= ========================
    PSI       Interpretation
    ========= ========================
    < 0.1     No significant change
    0.1-0.25  Moderate drift
    > 0.25    Significant drift
    ========= ========================

    The bins are defined by quantiles of the *baseline* so that the
    baseline distribution is uniform across bins; this is the standard
    credit-scoring convention and makes PSI scale-free.

    Args:
        baseline_dist: Samples from the reference distribution.
        current_dist: Samples from the distribution to test.
        n_bins: Number of equal-frequency bins. Must be >= 2.
        epsilon: Frequency floor to avoid ``log(0)`` / division by zero.

    Returns:
        Non-negative float. Returns ``0.0`` when either sample is empty.

    Raises:
        ValueError: If ``n_bins < 2`` or ``epsilon <= 0``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    # SEC: detach + move to CPU float for numerical stability across
    # devices; keep it defensive for accidental gradient-carrying inputs.
    baseline = _to_cpu_1d(baseline_dist)
    current = _to_cpu_1d(current_dist)
    if baseline.numel() == 0 or current.numel() == 0:
        return 0.0

    # Equal-frequency quantile bins from the baseline.
    qs = torch.linspace(0.0, 1.0, n_bins + 1)
    edges = torch.quantile(baseline, qs)
    # SEC: ensure strict monotonicity of edges; a degenerate baseline
    # (constant samples) collapses quantiles and would produce zero-
    # width bins. Nudge duplicates upward by epsilon.
    for i in range(1, edges.numel()):
        if edges[i].item() <= edges[i - 1].item():
            edges[i] = edges[i - 1] + epsilon
    # SEC: use bucketize + bincount instead of torch.histogram for broader
    # torch-version compatibility; the interior bin boundaries are the
    # middle edges (indices 1..n_bins-1) since bucketize assigns indices
    # based on the right edges.
    interior = edges[1:-1]
    base_idx = torch.bucketize(baseline, interior)
    curr_idx = torch.bucketize(current, interior)
    base_hist = torch.bincount(base_idx, minlength=n_bins).float()
    curr_hist = torch.bincount(curr_idx, minlength=n_bins).float()
    base_p = base_hist / max(float(base_hist.sum().item()), 1.0)
    curr_p = curr_hist / max(float(curr_hist.sum().item()), 1.0)

    # Avoid log(0).
    base_safe = torch.clamp(base_p, min=epsilon)
    curr_safe = torch.clamp(curr_p, min=epsilon)
    psi = ((curr_safe - base_safe) * torch.log(curr_safe / base_safe)).sum()
    return float(max(0.0, psi.item()))


# Alias with the exact name requested in the task description.
PSI = population_stability_index


# ---------------------------------------------------------------------------
# ConceptDriftDetector (ADWIN-style)
# ---------------------------------------------------------------------------


class ConceptDriftDetector:
    """ADWIN-style adaptive-window detector for streaming embedding errors.

    Maintains a deque of the last ``max_window`` scalar observations
    (typically the encoder's reconstruction / contrastive-loss error for
    each user message). On every call to :meth:`update`:

    1. The new value is appended.
    2. The window is scanned for a split point ``k`` such that the means
       of the two sub-windows differ by more than the ADWIN Hoeffding
       bound ``ε_cut`` at confidence ``δ``.
    3. If found, the older sub-window is discarded, a
       :class:`DriftDetectionResult` with ``drift_detected=True`` is
       returned, and an optional :attr:`on_drift_detected` callback is
       invoked.

    The Hoeffding bound used is::

        ε_cut = sqrt( (1 / (2 · m)) · ln(4 / δ) )

    where ``m`` is the harmonic mean of the two sub-window sizes
    (Bifet & Gavaldà 2007 Theorem 2).

    Args:
        delta: Confidence parameter ``δ ∈ (0, 1)``. Smaller values give
            fewer false positives but slower detection.
        min_sub_window: Minimum size of each sub-window considered for
            a split.
        max_window: Upper bound on the retained window; older values
            fall off automatically.
        cooldown: Minimum number of observations between successive
            drift alarms.
        on_drift_detected: Optional callback ``(DriftDetectionResult)``
            invoked on every drift event. Typically used to trigger
            :meth:`~i3.continual.ewc.ElasticWeightConsolidation.
            consolidate`.
    """

    def __init__(
        self,
        *,
        delta: float = _DEFAULT_DELTA,
        min_sub_window: int = _DEFAULT_MIN_SUB_WINDOW,
        max_window: int = _DEFAULT_MAX_WINDOW,
        cooldown: int = _DEFAULT_COOLDOWN,
        on_drift_detected: Callable[[DriftDetectionResult], None] | None = None,
    ) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if min_sub_window < 2:
            raise ValueError(
                f"min_sub_window must be >= 2, got {min_sub_window}"
            )
        if max_window < 2 * min_sub_window:
            raise ValueError(
                "max_window must be >= 2 * min_sub_window; "
                f"got max_window={max_window}, min_sub_window={min_sub_window}"
            )
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")

        self.delta: float = float(delta)
        self.min_sub_window: int = int(min_sub_window)
        self.max_window: int = int(max_window)
        self.cooldown: int = int(cooldown)
        self.on_drift_detected: Callable[[DriftDetectionResult], None] | None = on_drift_detected

        self._window: deque[float] = deque(maxlen=self.max_window)
        self._baseline_samples: list[float] = []
        self._cooldown_remaining: int = 0
        self._drift_count: int = 0

    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Current size of the retained window."""
        return len(self._window)

    @property
    def drift_count(self) -> int:
        """Total number of drift events since construction."""
        return self._drift_count

    def snapshot_baseline(self) -> None:
        """Freeze the current window as the PSI baseline."""
        self._baseline_samples = list(self._window)

    def reset(self) -> None:
        """Clear the window, baseline, and counters."""
        self._window.clear()
        self._baseline_samples = []
        self._cooldown_remaining = 0
        self._drift_count = 0

    # ------------------------------------------------------------------

    def update(self, value: float) -> DriftDetectionResult:
        """Ingest a new observation and return the detection result.

        Args:
            value: Scalar observation (typically an embedding-level
                reconstruction error).

        Returns:
            :class:`DriftDetectionResult` describing the window after
            this update.
        """
        if not math.isfinite(value):
            # SEC: Silently skip non-finite observations rather than
            # polluting the window. The result still reports the
            # current window size for symmetry with normal updates.
            logger.warning("Drift detector dropped non-finite value: %r", value)
            return DriftDetectionResult(
                drift_detected=False, window_size=self.window_size
            )
        self._window.append(float(value))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return DriftDetectionResult(
                drift_detected=False, window_size=self.window_size
            )

        cut = self._find_adwin_cut()
        if cut is None:
            return DriftDetectionResult(
                drift_detected=False, window_size=self.window_size
            )

        old = list(self._window)[:cut]
        new = list(self._window)[cut:]
        old_mean = sum(old) / len(old)
        new_mean = sum(new) / len(new)

        # SEC: Drop the older sub-window. The deque's maxlen still
        # applies, so the new window is the retained half.
        self._window.clear()
        self._window.extend(new)

        psi_val: float | None = None
        if self._baseline_samples:
            psi_val = population_stability_index(
                self._baseline_samples, new
            )

        self._drift_count += 1
        self._cooldown_remaining = self.cooldown

        result = DriftDetectionResult(
            drift_detected=True,
            window_size=len(self._window),
            old_mean=old_mean,
            new_mean=new_mean,
            cut_point=cut,
            psi=psi_val,
        )
        logger.info(
            "Drift detected: old_mean=%.4f new_mean=%.4f cut=%d psi=%s",
            old_mean,
            new_mean,
            cut,
            f"{psi_val:.4f}" if psi_val is not None else "n/a",
        )
        if self.on_drift_detected is not None:
            try:
                self.on_drift_detected(result)
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "on_drift_detected callback raised %s: %s",
                    type(exc).__name__,
                    exc,
                )
        return result

    def update_batch(self, values: Iterable[float]) -> list[DriftDetectionResult]:
        """Ingest a batch of observations in order.

        Args:
            values: Iterable of scalar observations.

        Returns:
            List of :class:`DriftDetectionResult`, one per ingested
            value, in order.
        """
        return [self.update(v) for v in values]

    # ------------------------------------------------------------------
    # PSI helpers
    # ------------------------------------------------------------------

    def psi_against_baseline(self) -> float | None:
        """Return the PSI between the baseline and current window.

        Returns ``None`` if no baseline has been snapshotted or the
        current window is empty.
        """
        if not self._baseline_samples or not self._window:
            return None
        return population_stability_index(
            self._baseline_samples, list(self._window)
        )

    # ------------------------------------------------------------------
    # Internals -- ADWIN cut search
    # ------------------------------------------------------------------

    def _find_adwin_cut(self) -> int | None:
        """Return the smallest split index at which ADWIN would cut.

        Bifet & Gavaldà 2007 argue: for every partition of the window
        into an older sub-window of size ``n_0`` and a newer of size
        ``n_1``, compute::

            ε_cut = sqrt( (1 / (2 m)) · ln(4 / δ) ),
            m     = 1 / (1 / n_0 + 1 / n_1)

        If ``|μ_0 − μ_1| > ε_cut`` the older sub-window is incompatible
        with the newer one under the null hypothesis that both were
        drawn from the same distribution. We return the first such
        index so the retained window is the longest newer suffix.

        Returns:
            Cut point ``k`` such that ``window[:k]`` is discarded, or
            ``None`` if no split satisfies the bound.
        """
        n = len(self._window)
        if n < 2 * self.min_sub_window:
            return None
        arr = list(self._window)
        # Cumulative sums let us compute any sub-window mean in O(1).
        cumsum = [0.0]
        for v in arr:
            cumsum.append(cumsum[-1] + v)

        delta_prime = self.delta / max(1, (n - 1))  # Bonferroni over splits
        ln_term = math.log(4.0 / delta_prime)

        for k in range(self.min_sub_window, n - self.min_sub_window + 1):
            n0 = k
            n1 = n - k
            mu0 = cumsum[k] / n0
            mu1 = (cumsum[n] - cumsum[k]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon_cut = math.sqrt((1.0 / (2.0 * m)) * ln_term)
            if abs(mu0 - mu1) > epsilon_cut:
                return k
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_cpu_1d(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Coerce *values* to a 1-D CPU float tensor."""
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    return torch.tensor(list(values), dtype=torch.float32)


__all__ = [
    "PSI",
    "ConceptDriftDetector",
    "DriftDetectionResult",
    "population_stability_index",
]
