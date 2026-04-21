"""Hypothesis property tests for Welford's online mean/variance tracker
inside :class:`i3.user_model.model.UserModel`.

Invariants:
    * **Mean matches numpy** — after ``n`` observations, the tracker's
      running mean equals ``numpy.mean(xs)`` up to float tolerance.
    * **Sample variance matches numpy** — after ``n > 1`` observations
      the std equals ``numpy.std(xs, ddof=1)`` up to float tolerance.
    * **NaN skips the update** — feeding NaN never corrupts the running
      stats (matches the internal SEC guard).
    * **Reset clears state** — ``reset_statistics()`` returns mean/std
      dictionaries to empty.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings, strategies as st

from i3.interaction.types import InteractionFeatureVector


# Import the UserModel + config lazily so we can skip cleanly on a
# minimal install that lacks the user-model subpackage.
user_model_mod = pytest.importorskip("i3.user_model.model")
config_mod = pytest.importorskip("i3.config")


# Which feature names does the baseline tracker track?  We introspect
# rather than hard-code so the test keeps working if the list evolves.
_BASELINE_NAMES = list(getattr(user_model_mod, "_BASELINE_FEATURE_NAMES", []))
# Fallback: use a safe subset that must exist on every feature vector.
if not _BASELINE_NAMES:
    _BASELINE_NAMES = [
        f.name
        for f in fields(InteractionFeatureVector)
        if f.name
        in {
            "mean_iki",
            "std_iki",
            "composition_speed",
            "message_length",
            "type_token_ratio",
            "mean_word_length",
            "flesch_kincaid",
            "formality",
        }
    ]

_FEATURE_NAME = _BASELINE_NAMES[0] if _BASELINE_NAMES else "mean_iki"


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_user_model():
    """Build a UserModel with a minimal fake config."""
    UserModel = user_model_mod.UserModel
    try:
        cfg = config_mod.load_config("configs/default.yaml", set_seeds=False)
        user_cfg = cfg.user_model
    except Exception:
        pytest.skip("default config unavailable for Welford tests")
    return UserModel("test-user", user_cfg)


def _feed(model, name: str, values: list[float]) -> None:
    """Push each value through ``_update_feature_statistics`` via a
    minimal feature vector where only *name* is set."""
    for v in values:
        features = InteractionFeatureVector()
        setattr(features, name, float(v))
        model._update_feature_statistics(features)


# A strategy of realistic float values for per-feature tests.
_STAT_FLOAT = st.floats(
    min_value=-5.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


# ─────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────


class TestWelfordMean:
    @given(
        values=st.lists(_STAT_FLOAT, min_size=1, max_size=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_running_mean_matches_numpy(self, values: list[float]) -> None:
        """Running mean after n pushes == numpy.mean(xs)."""
        model = _make_user_model()
        _feed(model, _FEATURE_NAME, values)
        expected = float(np.mean(values))
        got = model._feature_mean.get(_FEATURE_NAME, 0.0)
        assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-6), (
            f"got={got} expected={expected} n={len(values)}"
        )


class TestWelfordVariance:
    @given(
        values=st.lists(_STAT_FLOAT, min_size=2, max_size=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_running_variance_matches_numpy(self, values: list[float]) -> None:
        """Running sample-variance std == numpy.std(xs, ddof=1)."""
        model = _make_user_model()
        _feed(model, _FEATURE_NAME, values)
        expected_std = float(np.std(values, ddof=1))
        # Pull from the profile which is the place std is persisted.
        stored_std = model.profile.baseline_features_std or {}
        got_std = stored_std.get(_FEATURE_NAME, 0.0)
        # Welford's clamps M2 >= 0, so the std can differ by 1 ulp on
        # constant inputs — tolerance accommodates that.
        assert math.isclose(got_std, expected_std, rel_tol=1e-4, abs_tol=1e-6)


class TestWelfordNaNGuard:
    @given(
        values=st.lists(_STAT_FLOAT, min_size=1, max_size=10),
    )
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_nan_input_does_not_corrupt(self, values: list[float]) -> None:
        """Feeding NaN between real samples preserves mean/variance."""
        model = _make_user_model()
        _feed(model, _FEATURE_NAME, values)
        mean_before = model._feature_mean.get(_FEATURE_NAME, 0.0)

        # Push a NaN — the guard inside _update_feature_statistics should
        # skip the entire vector without mutating state.
        features = InteractionFeatureVector()
        setattr(features, _FEATURE_NAME, float("nan"))
        model._update_feature_statistics(features)

        mean_after = model._feature_mean.get(_FEATURE_NAME, 0.0)
        assert mean_before == pytest.approx(mean_after, abs=1e-12)


class TestWelfordReset:
    def test_reset_clears_state(self) -> None:
        model = _make_user_model()
        _feed(model, _FEATURE_NAME, [0.1, 0.2, 0.3, 0.4])
        assert _FEATURE_NAME in model._feature_mean
        model.reset_statistics()
        assert model._feature_mean == {}
        assert model._feature_m2 == {}
        assert model._feature_count == 0
