"""Iter 34 — CognitiveLoadAdapter dynamic-range regression tests.

Pre-iter-34 the adapter had a double-normalisation bug (mean_word_length
divided by 10, flesch_kincaid divided by 20) that suppressed two of
its four signals.  cognitive_load saturated around 0.6 even on the
most complex inputs, leaving the SLM cross-attention and the state
classifier with a narrow stuck-in-the-middle signal.

These tests pin the post-fix dynamic range so any future regression
that re-introduces a similar bug fires immediately.
"""

from __future__ import annotations

import pytest

from i3.adaptation.controller import AdaptationController
from i3.config import AdaptationConfig
from i3.interaction.types import InteractionFeatureVector
from i3.user_model.types import DeviationMetrics


def _fv(**overrides) -> InteractionFeatureVector:
    base = {n: 0.0 for n in [
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


def _zero_dev() -> DeviationMetrics:
    return DeviationMetrics(
        current_vs_baseline=0.0, current_vs_session=0.0,
        engagement_score=0.5, magnitude=0.0,
        iki_deviation=0.0, length_deviation=0.0, vocab_deviation=0.0,
        formality_deviation=0.0, speed_deviation=0.0,
        engagement_deviation=0.0, complexity_deviation=0.0,
        pattern_deviation=0.0,
    )


@pytest.fixture
def ctrl() -> AdaptationController:
    return AdaptationController(AdaptationConfig())


def test_cognitive_load_low_on_zero_signals(ctrl) -> None:
    """All-zero features → cognitive_load near 0.1 (the lower
    end of the [0.1, 0.9] span; low complexity in, low complexity out)."""
    v = ctrl.compute(_fv(), _zero_dev())
    assert v.cognitive_load == pytest.approx(0.1, abs=0.05)


def test_cognitive_load_mid_on_balanced_signals(ctrl) -> None:
    """Mid-complexity inputs (all features ≈ 0.5) → cognitive_load ≈ 0.6."""
    v = ctrl.compute(
        _fv(type_token_ratio=0.5, mean_word_length=0.5,
            flesch_kincaid=0.5, message_length=0.5),
        _zero_dev(),
    )
    assert 0.5 <= v.cognitive_load <= 0.7


def test_cognitive_load_high_on_complex_signals(ctrl) -> None:
    """High-complexity inputs (all features ≈ 0.9) → cognitive_load ≈ 0.9.

    Pre-iter-34 this maxed at ~0.58 because two of four signals were
    suppressed.  After iter 34, every signal contributes its full
    [0, 1] range and cognitive_load reaches the 0.9 ceiling.
    """
    v = ctrl.compute(
        _fv(type_token_ratio=0.9, mean_word_length=0.9,
            flesch_kincaid=0.9, message_length=0.9),
        _zero_dev(),
    )
    assert v.cognitive_load >= 0.85


def test_cognitive_load_full_dynamic_range_spread(ctrl) -> None:
    """Spread between zero-signal and max-signal cognitive_load is at
    least 0.7 — the adapter actually responds to content variation."""
    v_zero = ctrl.compute(_fv(), _zero_dev())
    v_max = ctrl.compute(
        _fv(type_token_ratio=1.0, mean_word_length=1.0,
            flesch_kincaid=1.0, message_length=1.0),
        _zero_dev(),
    )
    spread = v_max.cognitive_load - v_zero.cognitive_load
    assert spread >= 0.7, (
        f"cognitive_load spread should be >= 0.7 after iter 34 "
        f"(was ~0.54 with the double-normalisation bug); got {spread}"
    )


def test_cognitive_load_responds_to_each_signal_individually(ctrl) -> None:
    """Each of the four signals (ttr, mean_word_length, flesch_kincaid,
    message_length) should produce a comparable individual contribution
    when raised in isolation — proving none of them is silently
    suppressed."""
    v_base = ctrl.compute(_fv(), _zero_dev())
    contributions = {}
    for name in ("type_token_ratio", "mean_word_length",
                 "flesch_kincaid", "message_length"):
        v_one = ctrl.compute(_fv(**{name: 1.0}), _zero_dev())
        contributions[name] = v_one.cognitive_load - v_base.cognitive_load

    # Every signal must individually move cognitive_load by at least
    # ~0.20 (one of four signals at max → user_complexity += 0.25 →
    # cognitive_load gets ~0.25 + 0.10 = +0.35 from baseline 0.1, so
    # at least 0.20 over the zero baseline.
    for name, delta in contributions.items():
        assert delta >= 0.15, (
            f"signal '{name}' contributes only {delta:.3f}; should be "
            f">= 0.15 (was the suppressed-signal regression)"
        )

    # All four signals should have similar magnitudes (not one tenth
    # the others).  Allow 2x range across signals.
    max_c = max(contributions.values())
    min_c = min(contributions.values())
    assert max_c <= 2.0 * min_c, (
        f"signals have wildly imbalanced contributions: {contributions}"
    )
