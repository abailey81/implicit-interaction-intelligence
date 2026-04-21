"""Bootstrap confidence intervals (Efron 1979) for small-sample metrics.

Used by :mod:`i3.fairness.subgroup_metrics` to put error bars on each
archetype's mean adaptation vector.  Per-archetype subgroup sizes in the
diary are typically small (tens to low-hundreds of sessions), which makes
parametric CIs fragile and the bootstrap the appropriate choice.

References
----------
* Efron, B. (1979). *Bootstrap methods: another look at the jackknife.*
  The Annals of Statistics 7(1), 1–26.
* Davison, A. C., Hinkley, D. V. (1997). *Bootstrap methods and their
  application.*  Cambridge University Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """The output of a bootstrap CI computation.

    Attributes:
        point_estimate: The statistic computed on the original sample.
        lower: Lower bound of the confidence interval at the requested level.
        upper: Upper bound of the confidence interval at the requested level.
        level: The confidence level used (e.g. 0.95).
        num_resamples: The number of bootstrap resamples performed.
    """

    point_estimate: float
    lower: float
    upper: float
    level: float
    num_resamples: int

    @property
    def width(self) -> float:
        """The width of the interval (``upper - lower``)."""
        return self.upper - self.lower


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    num_resamples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Compute a percentile bootstrap CI for an arbitrary statistic.

    This is the canonical non-parametric bootstrap (Efron 1979): draw *B*
    resamples with replacement from *data*, compute *statistic* on each,
    and report the percentile interval.  For small samples with unknown
    distributions — which is exactly the per-archetype regime — this is
    the right choice.

    Args:
        data: 1-D numpy array of observations.
        statistic: Callable mapping a 1-D array of observations to a scalar.
        num_resamples: Number of bootstrap resamples.  The default 2000 is
            conservative for percentile CIs at the 95% level; go higher for
            tighter extreme-quantile estimates.
        level: Confidence level (default 0.95).
        seed: Optional RNG seed for reproducibility.

    Returns:
        A :class:`BootstrapResult`.

    Raises:
        ValueError: If ``data`` is empty or not 1-D, or ``level`` is not in
            ``(0, 1)``.
    """
    if data.ndim != 1:
        raise ValueError(f"bootstrap_ci expects a 1-D array, got shape {data.shape}")
    if data.size == 0:
        raise ValueError("bootstrap_ci: data is empty")
    if not (0.0 < level < 1.0):
        raise ValueError(f"bootstrap_ci: level must be in (0, 1), got {level}")
    if num_resamples < 1:
        raise ValueError(f"bootstrap_ci: num_resamples must be >= 1, got {num_resamples}")

    rng = np.random.default_rng(seed)
    point = float(statistic(data))
    n = data.size
    samples = np.empty(num_resamples, dtype=np.float64)
    for b in range(num_resamples):
        idx = rng.integers(0, n, size=n)
        samples[b] = float(statistic(data[idx]))
    alpha = (1.0 - level) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        level=level,
        num_resamples=num_resamples,
    )


def bootstrap_mean_ci(
    data: np.ndarray,
    num_resamples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Convenience wrapper for a mean CI.

    Equivalent to ``bootstrap_ci(data, np.mean, ...)``.
    """
    return bootstrap_ci(
        data,
        statistic=lambda arr: float(np.mean(arr)),
        num_resamples=num_resamples,
        level=level,
        seed=seed,
    )
