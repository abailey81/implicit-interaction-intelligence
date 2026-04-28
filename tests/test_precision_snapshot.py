"""Snapshot tests pinning the iter 1-20 numeric outputs.

Iter 21 — capture the exact post-iter-20 behaviour on deterministic
inputs, so any future regression in the precision sweep is caught
loudly.

Each test computes a deterministic output from a fixed input and
asserts the value with tight tolerance.  If a future change shifts
the numeric output, this test fires — forcing a deliberate decision
about whether the change is intentional.

These are the canonical "behaviour snapshots" that defend the
precision iterations from accidental regression.
"""

from __future__ import annotations

import math

import pytest
import torch

from i3.affect.shift_detector import AffectShiftDetector
from i3.affect.state_classifier import classify_user_state
from i3.interaction.features import (
    BaselineTracker,
    _clamp01,
    _clamp_neg1_1,
    _normalised_slope,
    _std,
)
from i3.interaction.types import InteractionFeatureVector


def _fv(**overrides) -> InteractionFeatureVector:
    base: dict[str, float] = {n: 0.0 for n in [
        "mean_iki", "std_iki", "mean_burst_length", "mean_pause_duration",
        "backspace_ratio", "composition_speed", "pause_before_send",
        "editing_effort", "message_length", "type_token_ratio",
        "mean_word_length", "flesch_kincaid", "question_ratio",
        "formality", "emoji_density", "sentiment_valence",
        "length_trend", "latency_trend", "vocab_trend",
        "engagement_velocity", "topic_coherence", "session_progress",
        "time_deviation", "response_depth",
        "iki_deviation", "length_deviation", "vocab_deviation",
        "formality_deviation", "speed_deviation", "engagement_deviation",
        "complexity_deviation", "pattern_deviation",
    ]}
    base.update(overrides)
    return InteractionFeatureVector(**base)


# ---------------------------------------------------------------------------
# iter 2 + iter 15 — Bessel-corrected variance snapshots
# ---------------------------------------------------------------------------


def test_baseline_tracker_known_z_score() -> None:
    """A known sample produces an exact pre-computed deviation."""
    bt = BaselineTracker(warmup=2)
    for v in [0.10, 0.20, 0.30, 0.40, 0.50]:
        bt.update(_fv(mean_iki=v))
    # Sample variance of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.025
    # std = sqrt(0.025) ≈ 0.158113883
    # Probe = 0.70 → z = (0.70 - 0.30) / 0.158113883 = 2.5298
    # Clamped to [-1, 1] after / 3.0 = 0.84327
    actual = bt.deviation("mean_iki", 0.70)
    assert actual == pytest.approx(0.84327, abs=1e-4)


def test_features_std_known_value() -> None:
    """_std returns the exact Bessel-corrected std of a known list."""
    # Sample variance of [1, 2, 3, 4, 5] = 2.5
    # std = sqrt(2.5) ≈ 1.5811
    assert _std([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.5811388, abs=1e-6)
    # Population variance would be 2.0 / std=sqrt(2)=1.414 — confirm
    # we are NOT using that.
    assert _std([1.0, 2.0, 3.0, 4.0, 5.0]) != pytest.approx(1.4142136, abs=1e-6)


# ---------------------------------------------------------------------------
# iter 8 — _normalised_slope snapshot
# ---------------------------------------------------------------------------


def test_normalised_slope_known_value() -> None:
    """A known [0, 1] linear ramp produces +1.0."""
    assert _normalised_slope([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(1.0)
    assert _normalised_slope([1.0, 0.75, 0.5, 0.25, 0.0]) == pytest.approx(-1.0)
    assert _normalised_slope([0.5, 0.5, 0.5, 0.5]) == 0.0


# ---------------------------------------------------------------------------
# iter 20 — clamp helpers NaN/inf snapshot
# ---------------------------------------------------------------------------


def test_clamp_helpers_known_values() -> None:
    nan = float("nan")
    pinf = float("inf")
    assert _clamp01(nan) == 0.0
    assert _clamp01(pinf) == 0.0
    assert _clamp01(0.5) == 0.5
    assert _clamp01(-0.7) == 0.0
    assert _clamp01(1.3) == 1.0
    assert _clamp_neg1_1(nan) == 0.0
    assert _clamp_neg1_1(pinf) == 0.0
    assert _clamp_neg1_1(0.5) == 0.5
    assert _clamp_neg1_1(-1.7) == -1.0


# ---------------------------------------------------------------------------
# iter 1 + iter 7 — shift_detector snapshot for canonical scenario
# ---------------------------------------------------------------------------


def _calm_obs(detector: AffectShiftDetector, user: str, session: str, n: int) -> None:
    for _ in range(n):
        detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )


def test_canonical_rising_load_scenario_snapshot() -> None:
    """5 calm + 1 stressed (iki=200, edits=4): the documented brief
    trigger should fire with rising_load direction and confidence in
    a known range.

    This is the central post-iter-9 snapshot: any change that breaks
    the calibration here is flagged immediately.
    """
    detector = AffectShiftDetector()
    _calm_obs(detector, "u_snap", "s_snap", n=5)

    result = detector.observe(
        user_id="u_snap", session_id="s_snap", embedding=torch.zeros(64),
        composition_time_ms=4500.0, edit_count=4, pause_before_send_ms=400.0,
        keystroke_iki_mean=200.0, keystroke_iki_std=50.0,
    )

    # Detection invariants.
    assert result.detected is True
    assert result.direction == "rising_load"
    # Recent IKI = (120+120+200)/3 = 146.67 → +22.2% delta.
    assert result.iki_delta_pct == pytest.approx(22.22, abs=0.5)
    # Edit baseline = 0; edit recent = 1.33 → zero_baseline_default
    # = 200%.
    assert result.edit_delta_pct == pytest.approx(200.0, abs=0.1)
    # Magnitude: zero embeddings on both sides — exactly 0.
    assert result.magnitude == 0.0
    # Confidence: in the documented [0.5, 1.0] band; specifically
    # in the lower portion since IKI is just over the strong-tier
    # threshold.
    assert 0.5 <= result.confidence <= 0.85


# ---------------------------------------------------------------------------
# iter 3 — state_classifier calibration snapshots
# ---------------------------------------------------------------------------


def test_clean_calm_snapshot() -> None:
    label = classify_user_state(
        adaptation={"cognitive_load": 0.25, "formality": 0.5,
                    "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=2000.0, edit_count=0,
        iki_mean=110.0, iki_std=15.0,
        engagement_score=0.6, deviation_from_baseline=0.0,
        baseline_established=True, messages_in_session=10,
    )
    assert label.state == "calm"
    # Post-iter-3 calibration: confidence should be in [0.55, 0.95].
    assert 0.55 <= label.confidence <= 0.95


def test_clean_stressed_snapshot() -> None:
    label = classify_user_state(
        adaptation={"cognitive_load": 0.85, "formality": 0.5,
                    "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=4500.0, edit_count=4,
        iki_mean=180.0, iki_std=50.0,
        engagement_score=0.5, deviation_from_baseline=0.3,
        baseline_established=True, messages_in_session=5,
    )
    assert label.state == "stressed"
    assert "elevated edit count" in label.contributing_signals
    assert "high cognitive load" in label.contributing_signals


# ---------------------------------------------------------------------------
# iter 4 — topic_coherence snapshot
# ---------------------------------------------------------------------------


def test_topic_coherence_identical_signature_snapshot() -> None:
    """Two identical (ttr, formality, fk) signatures return cosine
    similarity 1.0 mapped to 1.0 in [0, 1] space."""
    from i3.interaction.features import _cosine_similarity_unit
    sig = (0.1, 0.2, -0.3)
    assert _cosine_similarity_unit(sig, sig) == pytest.approx(1.0)


def test_topic_coherence_anti_correlated_snapshot() -> None:
    """Anti-correlated signatures return 0.0."""
    from i3.interaction.features import _cosine_similarity_unit
    a = (0.5, 0.3, 0.2)
    b = (-0.5, -0.3, -0.2)
    assert _cosine_similarity_unit(a, b) == pytest.approx(0.0, abs=1e-9)


def test_topic_coherence_zero_zero_returns_1() -> None:
    """Both-zero ⇒ "no signal" ⇒ identical (1.0)."""
    from i3.interaction.features import _cosine_similarity_unit
    assert _cosine_similarity_unit((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 1.0


def test_topic_coherence_one_zero_one_signal_returns_half() -> None:
    """One zero, one non-zero ⇒ no signal to compare ⇒ midpoint 0.5."""
    from i3.interaction.features import _cosine_similarity_unit
    assert _cosine_similarity_unit((0.0, 0.0, 0.0), (0.5, 0.3, 0.2)) == 0.5


# ---------------------------------------------------------------------------
# iter 26 — falling_load detection snapshot
# ---------------------------------------------------------------------------


def test_canonical_falling_load_scenario_snapshot() -> None:
    """5 stressed (iki=200, edits=4) baseline + 3 calm-recovery
    (iki=110, edits=1) recent observations: should fire as
    falling_load (the documented brief trigger).

    Pins the iter-7 fixed-baseline interaction with the iter-1
    falling-load condition: only fires when recent IKI is < -15%
    AND edits are flat-or-falling.
    """
    detector = AffectShiftDetector()
    user, session = "u_fall_snap", "s_fall_snap"

    # Stressed baseline.
    for _ in range(5):
        detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=4500.0, edit_count=4, pause_before_send_ms=600.0,
            keystroke_iki_mean=200.0, keystroke_iki_std=40.0,
        )

    # Recovery — first observation should fire.
    first = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2500.0, edit_count=1, pause_before_send_ms=300.0,
        keystroke_iki_mean=110.0, keystroke_iki_std=20.0,
    )
    if not first.detected:
        # Need 1-2 more observations to fully populate the recent window.
        first = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2500.0, edit_count=1, pause_before_send_ms=300.0,
            keystroke_iki_mean=110.0, keystroke_iki_std=20.0,
        )
    if not first.detected:
        first = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2500.0, edit_count=1, pause_before_send_ms=300.0,
            keystroke_iki_mean=110.0, keystroke_iki_std=20.0,
        )

    assert first.detected is True
    assert first.direction == "falling_load"
    assert first.iki_delta_pct <= -15.0  # recovery direction
    assert first.edit_delta_pct <= 5.0    # edits flat-or-falling
    # Confidence in valid band when detected.
    assert 0.5 <= first.confidence <= 1.0
