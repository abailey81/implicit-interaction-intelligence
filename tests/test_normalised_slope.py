"""Tests for the ``_normalised_slope`` helper used in session-feature
trend computation.

Iter 8: replace the previous ``slope / abs(y_mean)`` normaliser with
``slope * (n - 1)`` (= total change across the window) — bounded for
[0, 1] inputs without dependence on the mean magnitude.

Old behaviour blew up at small y_mean: e.g. a slope of 0.01 with
y_mean=0.001 produced slope/y_mean = 10, which then clamped to the
[-1, 1] ceiling regardless of actual signal.  Multiple low-magnitude
features all pinned at ±1 looked like saturated trends to downstream
models, hiding the real signal.

New behaviour: total change across the window stays naturally bounded
in [-1, 1] for any [0, 1] input.  Continuous, monotonic in slope, no
mean-dependent instability.
"""

from __future__ import annotations

import math

import pytest

from i3.interaction.features import _normalised_slope


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_empty_or_single_value_returns_zero() -> None:
    assert _normalised_slope([]) == 0.0
    assert _normalised_slope([0.5]) == 0.0


def test_flat_sequence_returns_zero() -> None:
    assert _normalised_slope([0.5, 0.5, 0.5, 0.5]) == 0.0
    assert _normalised_slope([0.0, 0.0]) == 0.0
    assert _normalised_slope([1.0, 1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Monotonic sequences for [0, 1] inputs land near ±1.
# ---------------------------------------------------------------------------


def test_full_range_rising_slope_returns_one() -> None:
    """A perfectly linear rise from 0 to 1 across the window
    should produce slope=1.0 (full positive trend)."""
    n = 6
    values = [i / (n - 1) for i in range(n)]  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    assert _normalised_slope(values) == pytest.approx(1.0, abs=1e-9)


def test_full_range_falling_slope_returns_negative_one() -> None:
    n = 6
    values = [(n - 1 - i) / (n - 1) for i in range(n)]  # 1.0, 0.8, ..., 0.0
    assert _normalised_slope(values) == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Small-mean stability — the core iter 8 fix.
# ---------------------------------------------------------------------------


def test_small_y_mean_does_not_explode() -> None:
    """Iter 8: the result for a tiny-but-real upward trend is in [-1, 1]
    without blowing up on the y_mean=epsilon case.  Old behaviour:
    slope / |y_mean| with y_mean ~ 0.001 produced ~10 (clamped to 1)."""
    # A slow rise from 0.001 to 0.005 — y_mean is ~0.003.
    n = 5
    values = [0.001 + i * 0.001 for i in range(n)]  # 0.001, 0.002, 0.003, 0.004, 0.005
    result = _normalised_slope(values)
    assert math.isfinite(result)
    # Total change = 0.004 across the window; the new formula gives
    # exactly 0.004 (slope * (n-1) = (0.001/step) * 4 = 0.004).
    assert result == pytest.approx(0.004, abs=1e-9)
    # Crucially the result is small — proportional to the actual
    # change — not saturated at 1.0 from mean-division blow-up.
    assert abs(result) < 0.01


def test_zero_y_mean_with_real_slope_is_handled() -> None:
    """A slope through zero (negative-to-positive) shouldn't trip the
    old `if abs(y_mean) > 1e-9` guard and lose precision."""
    # values = [-0.4, -0.2, 0.0, 0.2, 0.4] — y_mean = 0, slope = 0.2/step.
    values = [-0.4, -0.2, 0.0, 0.2, 0.4]
    result = _normalised_slope(values)
    # Total change across window = 0.8.  New formula returns this
    # cleanly; old returned slope (= 0.2/step) when y_mean was below
    # the guard, which mixed semantics.
    assert math.isfinite(result)
    assert result == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# Continuity — small input perturbations produce small output changes.
# ---------------------------------------------------------------------------


def test_continuous_under_small_perturbation() -> None:
    base = [0.3, 0.4, 0.5, 0.6, 0.7]
    pert = [0.31, 0.41, 0.51, 0.61, 0.71]  # uniform shift
    a = _normalised_slope(base)
    b = _normalised_slope(pert)
    # Shifting all values by a constant doesn't change the slope.
    assert abs(a - b) < 1e-9


def test_continuous_under_slope_perturbation() -> None:
    base = [0.3, 0.4, 0.5, 0.6, 0.7]      # slope = 0.1/step
    pert = [0.3, 0.4, 0.5, 0.6, 0.71]     # slightly steeper at the end
    a = _normalised_slope(base)
    b = _normalised_slope(pert)
    # Small perturbation should produce a small output difference.
    assert abs(a - b) < 0.05


# ---------------------------------------------------------------------------
# Determinism + idempotence
# ---------------------------------------------------------------------------


def test_deterministic_under_repeated_calls() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    a = _normalised_slope(values)
    b = _normalised_slope(values)
    c = _normalised_slope(list(values))
    assert a == b == c


# ---------------------------------------------------------------------------
# Robustness to extreme inputs
# ---------------------------------------------------------------------------


def test_handles_large_negative_values() -> None:
    """Large negative magnitudes mustn't produce inf or NaN."""
    values = [-1e6, -1e6 + 1, -1e6 + 2, -1e6 + 3]
    result = _normalised_slope(values)
    assert math.isfinite(result)


def test_handles_floating_point_jitter() -> None:
    """Float jitter near a flat sequence shouldn't produce huge slopes."""
    values = [0.5 + 1e-15 * (-1) ** i for i in range(10)]
    result = _normalised_slope(values)
    assert math.isfinite(result)
    assert abs(result) < 1e-13
