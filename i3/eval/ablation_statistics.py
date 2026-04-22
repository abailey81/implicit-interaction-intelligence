"""Statistical helpers for the cross-attention conditioning ablation study.

This module provides small, focused, fully-typed primitives for the Batch A
ablation experiment pre-registered in ``docs/experiments/preregistration.md``.

Functions:
    bootstrap_ci: Percentile bootstrap confidence interval for a sample mean.
    cohens_d: Pooled-standard-deviation Cohen's d effect size.
    paired_sign_test: Exact-binomial paired sign test.
    effect_size_interpretation: Textual bucket for a Cohen's d magnitude
        using the standard thresholds from Cohen (1988).

Design notes:
    * All functions accept `Sequence[float]` | `np.ndarray`; they are coerced
      to ``np.ndarray[float64]`` internally for numerical stability.
    * No broad ``except:`` clauses — every error path is a specific exception
      with an actionable message.
    * Deterministic seeding is accepted via a ``rng`` parameter; if ``None``
      a ``numpy.random.default_rng()`` (OS-entropy-seeded) is used, so the
      caller is responsible for passing a seeded RNG when reproducibility
      matters. The ablation experiment always passes ``np.random.default_rng(42)``.

References:
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
        2nd ed. Routledge. Chapters 2 & 7 for d thresholds.
    Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
        CRC Press. Chapter 13 for percentile-bootstrap CIs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from scipy.stats import binomtest as _binomtest
    _HAVE_SCIPY: bool = True
except ImportError:  # pragma: no cover — scipy is a transitive dep via sklearn
    _HAVE_SCIPY = False


_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the sample mean.

    Args:
        samples: Observed sample values. Must contain at least one element.
        n_resamples: Number of bootstrap resamples to draw. Defaults to
            ``10_000`` per the pre-registration.
        ci: Nominal coverage level in ``(0, 1)``. Defaults to ``0.95``.
        rng: Optional ``numpy.random.Generator``. If ``None``, a fresh
            entropy-seeded generator is used.

    Returns:
        Two-element tuple ``(lower, upper)`` of float bounds in the same
        units as ``samples``.

    Raises:
        ValueError: If ``samples`` is empty, ``n_resamples < 1``, or
            ``ci`` is not in the open interval ``(0, 1)``.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"samples must be 1-D, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    if rng is None:
        rng = np.random.default_rng()

    n = arr.size
    # Vectorised resample: [n_resamples, n] indices then reduce by row-mean.
    indices = rng.integers(low=0, high=n, size=(n_resamples, n))
    resampled_means = arr[indices].mean(axis=1)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.percentile(resampled_means, lower_pct))
    upper = float(np.percentile(resampled_means, upper_pct))
    return lower, upper


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def cohens_d(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Compute Cohen's d (pooled-sd) effect size between two samples.

    The sign convention is ``mean(a) - mean(b)``: a positive ``d`` means
    sample ``a`` has a larger mean than ``b``.

    Args:
        a: First sample (typically the "treatment").
        b: Second sample (typically the "control").

    Returns:
        Cohen's d as a float. Returns ``0.0`` when both samples are
        constant with the same mean (undefined effect collapses to zero
        rather than ``NaN``).

    Raises:
        ValueError: If either sample is empty or has fewer than 2 elements
            (the pooled standard deviation is undefined in that case).
    """
    arr_a = np.asarray(a, dtype=np.float64).ravel()
    arr_b = np.asarray(b, dtype=np.float64).ravel()
    if arr_a.size < 2 or arr_b.size < 2:
        raise ValueError(
            f"each sample must have at least 2 observations; "
            f"got |a|={arr_a.size}, |b|={arr_b.size}"
        )

    mean_diff = arr_a.mean() - arr_b.mean()

    var_a = arr_a.var(ddof=1)
    var_b = arr_b.var(ddof=1)
    n_a = arr_a.size
    n_b = arr_b.size

    # Pooled sample standard deviation (Cohen 1988, eq. 2.5.2).
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= _EPS:
        if abs(mean_diff) <= _EPS:
            return 0.0
        # Non-zero mean difference with zero variance: effect is formally
        # infinite; report a very large but finite value so downstream
        # formatting does not emit `inf`.
        return float(np.sign(mean_diff) * 1e6)
    pooled_sd = float(np.sqrt(pooled_var))
    return float(mean_diff / pooled_sd)


# ---------------------------------------------------------------------------
# Paired sign test
# ---------------------------------------------------------------------------


def paired_sign_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Exact-binomial paired sign test.

    Tests the null hypothesis that the median of ``a - b`` is zero under a
    symmetry assumption on signs. Pairs where ``a_i == b_i`` are discarded
    (the "two-sided"/"exclude zeros" convention).

    Args:
        a: First paired sample (length must equal ``b``).
        b: Second paired sample (length must equal ``a``).

    Returns:
        Two-sided p-value from the exact binomial with parameter ``0.5``.

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths or are empty.
    """
    arr_a = np.asarray(a, dtype=np.float64).ravel()
    arr_b = np.asarray(b, dtype=np.float64).ravel()
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"paired samples must have the same shape; "
            f"got {arr_a.shape} vs {arr_b.shape}"
        )
    if arr_a.size == 0:
        raise ValueError("paired samples must be non-empty")

    diffs = arr_a - arr_b
    # Exclude exact ties (difference of zero) — conventional handling.
    non_zero = diffs[np.abs(diffs) > _EPS]
    n = int(non_zero.size)
    if n == 0:
        # All pairs tied: no evidence against the null.
        return 1.0

    n_positive = int((non_zero > 0.0).sum())

    if _HAVE_SCIPY:
        result = _binomtest(n_positive, n=n, p=0.5, alternative="two-sided")
        return float(result.pvalue)

    # Fallback exact two-sided p-value via log-space binomial CDF.
    # We compute P(|X - n/2| >= |n_positive - n/2|) under Binomial(n, 0.5).
    k = n_positive
    deviation = abs(k - n / 2.0)
    log_half = np.log(0.5)
    # Work in log space to avoid overflow for large n.
    log_coeffs = np.zeros(n + 1, dtype=np.float64)
    # log C(n, i) via lgamma
    from math import lgamma
    log_n_fact = lgamma(n + 1)
    for i in range(n + 1):
        log_coeffs[i] = log_n_fact - lgamma(i + 1) - lgamma(n - i + 1)
    log_pmf = log_coeffs + n * log_half
    pmf = np.exp(log_pmf)
    mask = np.abs(np.arange(n + 1) - n / 2.0) >= deviation
    p_value = float(pmf[mask].sum())
    # Clamp to [0, 1] against floating-point drift.
    return max(0.0, min(1.0, p_value))


# ---------------------------------------------------------------------------
# Effect-size interpretation
# ---------------------------------------------------------------------------


def effect_size_interpretation(d: float) -> str:
    """Bucket a Cohen's d magnitude into a human-readable descriptor.

    Thresholds follow Cohen (1988):

    * ``|d| < 0.2`` → ``"negligible"``
    * ``0.2 ≤ |d| < 0.5`` → ``"small"``
    * ``0.5 ≤ |d| < 0.8`` → ``"medium"``
    * ``|d| ≥ 0.8`` → ``"large"``

    Args:
        d: Cohen's d value (any finite float; sign is ignored).

    Returns:
        One of ``"negligible"``, ``"small"``, ``"medium"``, ``"large"``.
    """
    magnitude = abs(float(d))
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


__all__ = [
    "bootstrap_ci",
    "cohens_d",
    "paired_sign_test",
    "effect_size_interpretation",
]
