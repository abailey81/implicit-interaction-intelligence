"""Iter 4 tests for the ``topic_coherence`` session-feature.

Before iter 4: rounding-Jaccard over (type_token_ratio, formality,
flesch_kincaid) at 0.1 resolution.  Brittle — a 0.05 shift in all
three features could collapse coherence from 1.0 to 0.0.

After iter 4: cosine similarity over the same three features.
Continuous, smooth, scale-invariant.
"""

from __future__ import annotations

import math
import time

import pytest

from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.interaction.types import InteractionFeatureVector


def _build_history_fv(
    ttr: float, formality: float, fk: float, message_length: float = 0.3
) -> InteractionFeatureVector:
    """Build a minimal history-vector with the three coherence-relevant fields."""
    fv_kwargs: dict[str, float] = {
        "mean_iki": 0.1,
        "std_iki": 0.05,
        "mean_burst_length": 0.1,
        "mean_pause_duration": 0.05,
        "backspace_ratio": 0.0,
        "composition_speed": 0.3,
        "pause_before_send": 0.05,
        "editing_effort": 0.0,
        "message_length": message_length,
        "type_token_ratio": ttr,
        "mean_word_length": 0.4,
        "flesch_kincaid": fk,
        "question_ratio": 0.0,
        "formality": formality,
        "emoji_density": 0.0,
        "sentiment_valence": 0.0,
        "length_trend": 0.0,
        "latency_trend": 0.0,
        "vocab_trend": 0.0,
        "engagement_velocity": 0.5,
        "topic_coherence": 0.5,
        "session_progress": 0.1,
        "time_deviation": 0.0,
        "response_depth": 0.5,
        "iki_deviation": 0.0,
        "length_deviation": 0.0,
        "vocab_deviation": 0.0,
        "formality_deviation": 0.0,
        "speed_deviation": 0.0,
        "engagement_deviation": 0.0,
        "complexity_deviation": 0.0,
        "pattern_deviation": 0.0,
    }
    return InteractionFeatureVector(**fv_kwargs)


def _coherence_for(
    *,
    history_fv: InteractionFeatureVector,
    current_text: str,
) -> float:
    """Run the extractor and return the resulting topic_coherence value."""
    extractor = FeatureExtractor()
    baseline = BaselineTracker(warmup=2)
    baseline.update(history_fv)
    baseline.update(history_fv)

    fv = extractor.extract(
        keystroke_metrics={
            "mean_iki_ms": 100.0,
            "std_iki_ms": 20.0,
            "mean_burst_length": 8.0,
            "mean_pause_duration_ms": 200.0,
            "backspace_ratio": 0.0,
            "composition_speed_cps": 4.0,
            "pause_before_send_ms": 200.0,
            "editing_effort": 0.0,
        },
        message_text=current_text,
        history=[history_fv],
        baseline=baseline,
        session_start_ts=0.0,
        current_ts=10.0,
    )
    return fv.topic_coherence


# ---------------------------------------------------------------------------
# Range, continuity, monotonicity
# ---------------------------------------------------------------------------


def test_coherence_in_zero_one_interval() -> None:
    """topic_coherence is always a similarity in [0, 1]."""
    cases = [
        ("hello there", _build_history_fv(0.7, 0.5, 0.4)),
        ("totally different topic now", _build_history_fv(0.3, 0.9, 0.1)),
        ("a", _build_history_fv(0.0, 0.0, 0.0)),
        ("the quick brown fox jumps over the lazy dog", _build_history_fv(0.95, 0.95, 0.95)),
    ]
    for text, prev in cases:
        score = _coherence_for(history_fv=prev, current_text=text)
        assert 0.0 <= score <= 1.0, f"out of range: {score} for {text!r}"


def test_coherence_is_continuous_under_small_perturbation() -> None:
    """A small perturbation must not collapse coherence (no cliffs).

    iter 4: was rounding-Jaccard at 0.1 resolution — a 0.05 shift
    in each of (ttr, formality, fk) would cross every rounding
    boundary and collapse coherence to 0.  With cosine similarity
    a 0.05 perturbation moves coherence by < 0.05.
    """
    base = _build_history_fv(0.5, 0.5, 0.4)
    # Perturb the *current* message's features just slightly — by
    # repeating a canonical text we keep the linguistic features
    # nearly identical.
    score_a = _coherence_for(history_fv=base, current_text="the quick brown fox jumps")
    score_b = _coherence_for(history_fv=base, current_text="the quick brown fox jumps over")
    assert abs(score_a - score_b) < 0.4, (
        f"continuity violated: |{score_a} - {score_b}| >= 0.4"
    )


# ---------------------------------------------------------------------------
# Identical-history sanity (high coherence)
# ---------------------------------------------------------------------------


def test_identical_history_gives_high_coherence() -> None:
    """If the previous and current messages have near-identical linguistic
    signatures, coherence should be near 1.0."""
    prev = _build_history_fv(0.5, 0.5, 0.4)
    score = _coherence_for(
        history_fv=prev,
        current_text="the cat sat on the mat",  # neutral, average-vocab
    )
    # Cosine of two near-identical 3-vectors is near 1.
    assert score >= 0.8, f"expected high coherence, got {score}"


# ---------------------------------------------------------------------------
# Far-apart history gives lower coherence
# ---------------------------------------------------------------------------


def test_far_apart_signatures_give_lower_coherence() -> None:
    """Strong difference between previous and current features should
    produce notably lower coherence."""
    # Previous message is high-FK, formal, varied vocab.
    prev_high = _build_history_fv(0.9, 0.95, 0.95)
    # Current message is low-FK, casual, repetitive: "ok ok ok ok".
    casual_text = "ok ok ok ok"
    low_score = _coherence_for(history_fv=prev_high, current_text=casual_text)

    # Same casual text but the *previous* turn was also casual.
    prev_low = _build_history_fv(0.3, 0.2, 0.1)
    high_score = _coherence_for(history_fv=prev_low, current_text=casual_text)

    # "Same-style" pairing should score higher than "opposite-style" pairing.
    assert high_score > low_score, (
        f"same-style coherence ({high_score}) should exceed cross-style "
        f"({low_score}) for {casual_text!r}"
    )


# ---------------------------------------------------------------------------
# Empty history: coherence == 0.0 (no comparison possible)
# ---------------------------------------------------------------------------


def test_empty_history_gives_zero_coherence() -> None:
    extractor = FeatureExtractor()
    baseline = BaselineTracker(warmup=2)
    fv = extractor.extract(
        keystroke_metrics={"mean_iki_ms": 100.0, "composition_speed_cps": 4.0},
        message_text="hello there",
        history=[],
        baseline=baseline,
        session_start_ts=0.0,
        current_ts=10.0,
    )
    assert fv.topic_coherence == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_coherence_is_deterministic() -> None:
    prev = _build_history_fv(0.6, 0.55, 0.45)
    text = "the quick brown fox jumps over the lazy dog"
    a = _coherence_for(history_fv=prev, current_text=text)
    b = _coherence_for(history_fv=prev, current_text=text)
    c = _coherence_for(history_fv=prev, current_text=text)
    assert a == b == c


# ---------------------------------------------------------------------------
# Iter 22 — _cosine_similarity_unit non-finite safety
# ---------------------------------------------------------------------------


def test_cosine_similarity_unit_handles_nan_inputs() -> None:
    """Iter 22: any NaN / inf in the input vector returns 0.5
    (midpoint) rather than propagating NaN through the score."""
    from i3.interaction.features import _cosine_similarity_unit

    nan = float("nan")
    pinf = float("inf")
    ninf = float("-inf")

    # NaN in one vector
    assert _cosine_similarity_unit((nan, 0.1, 0.2), (0.1, 0.1, 0.2)) == 0.5
    # +inf in one vector
    assert _cosine_similarity_unit((pinf, 0.1, 0.2), (0.1, 0.1, 0.2)) == 0.5
    # -inf in one vector
    assert _cosine_similarity_unit((0.1, 0.1, 0.2), (ninf, 0.1, 0.2)) == 0.5
    # Mixed pathological + normal
    assert _cosine_similarity_unit((nan, pinf, ninf), (0.1, 0.2, 0.3)) == 0.5
