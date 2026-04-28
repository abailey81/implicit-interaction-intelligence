"""Property-based fuzzing for the affect-detection pipeline.

Iter 13 — uses Hypothesis to generate random sequences of synthetic
keystroke / embedding observations and asserts the invariants of
``AffectShiftDetector``, ``BaselineTracker`` and
``classify_user_state`` hold across the full input space (not just
the curated scenarios in the regular tests).

Invariants asserted:

* No exception ever escapes ``observe`` / ``update`` / ``classify``.
* Every numeric output is finite (no NaN / inf).
* Every probability / similarity is in [0, 1].
* Every signed-percent delta is finite.
* Every label belongs to its declared finite vocabulary.
* The detector confidence is in [0.5, 1.0] when ``detected=True``,
  exactly 0.0 when ``detected=False``.
* BaselineTracker.deviation is always in [-1, 1].
* state classifier confidence is in [0, 1].
"""

from __future__ import annotations

import math

import hypothesis
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from i3.affect.shift_detector import AffectShiftDetector
from i3.affect.state_classifier import classify_user_state
from i3.interaction.features import BaselineTracker
from i3.interaction.types import InteractionFeatureVector


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_NON_NEG_FLOAT = st.floats(
    min_value=0.0,
    max_value=1e5,
    allow_nan=False,
    allow_infinity=False,
)
_UNIT_FLOAT = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_NON_NEG_INT = st.integers(min_value=0, max_value=1000)
_BOOL = st.booleans()


def _embedding_strategy(dim: int = 64):
    return st.lists(_FINITE_FLOAT, min_size=dim, max_size=dim).map(
        lambda v: torch.tensor(v, dtype=torch.float32)
    )


# ---------------------------------------------------------------------------
# AffectShiftDetector property tests
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@given(
    sequence=st.lists(
        st.tuples(
            _embedding_strategy(64),  # embedding
            _NON_NEG_FLOAT,           # composition_time_ms
            _NON_NEG_INT,             # edit_count
            _NON_NEG_FLOAT,           # pause_before_send_ms
            _NON_NEG_FLOAT,           # keystroke_iki_mean
            _NON_NEG_FLOAT,           # keystroke_iki_std
        ),
        min_size=1,
        max_size=12,
    ),
)
def test_shift_detector_invariants_hold_across_random_sequences(sequence) -> None:
    detector = AffectShiftDetector()
    user, session = "u_prop", "s_prop"

    for emb, comp, edits, pause, iki_m, iki_s in sequence:
        result = detector.observe(
            user_id=user,
            session_id=session,
            embedding=emb,
            composition_time_ms=comp,
            edit_count=edits,
            pause_before_send_ms=pause,
            keystroke_iki_mean=iki_m,
            keystroke_iki_std=iki_s,
        )
        # Every numeric must be finite.
        assert math.isfinite(result.magnitude), f"non-finite magnitude: {result.magnitude}"
        assert math.isfinite(result.iki_delta_pct), f"non-finite iki_delta_pct"
        assert math.isfinite(result.edit_delta_pct), f"non-finite edit_delta_pct"
        assert math.isfinite(result.confidence), f"non-finite confidence"
        # Confidence convention.
        if result.detected:
            assert 0.5 <= result.confidence <= 1.0, f"confidence={result.confidence} for detected"
        else:
            assert result.confidence == 0.0, f"confidence={result.confidence} for non-detected"
        # Direction belongs to vocabulary.
        assert result.direction in {"rising_load", "falling_load", "neutral"}
        # Suggestion is a string.
        assert isinstance(result.suggestion, str)


@settings(max_examples=80, deadline=None)
@given(
    bad_emb=st.one_of(
        st.none(),
        st.lists(_FINITE_FLOAT, min_size=0, max_size=200).map(
            lambda v: torch.tensor(v, dtype=torch.float32)
        ),
        st.just(torch.zeros(0)),
        st.just(torch.tensor([float("nan")] * 64)),
        st.just(torch.tensor([float("inf")] * 64)),
    ),
)
def test_shift_detector_handles_pathological_embeddings(bad_emb) -> None:
    """Even pathological embedding inputs must not raise or produce
    non-finite outputs."""
    detector = AffectShiftDetector()
    result = detector.observe(
        user_id="u_path",
        session_id="s_path",
        embedding=bad_emb,
        composition_time_ms=2000.0,
        edit_count=0,
        pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    assert math.isfinite(result.magnitude)
    assert math.isfinite(result.confidence)


# ---------------------------------------------------------------------------
# BaselineTracker property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    samples=st.lists(_UNIT_FLOAT, min_size=1, max_size=50),
    probe=_FINITE_FLOAT,
)
def test_baseline_deviation_always_in_minus_one_to_one(samples, probe) -> None:
    """The deviation z-score must always land in [-1, 1] regardless
    of input distribution."""
    bt = BaselineTracker(warmup=2)
    fv_kwargs: dict[str, float] = {n: 0.0 for n in [
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
    for s in samples:
        fv_kwargs["mean_iki"] = s
        fv = InteractionFeatureVector(**fv_kwargs)
        bt.update(fv)

    dev = bt.deviation("mean_iki", probe)
    assert math.isfinite(dev), f"non-finite deviation: {dev}"
    assert -1.0 <= dev <= 1.0, f"deviation out of range: {dev}"


@settings(max_examples=80, deadline=None)
@given(samples=st.lists(_UNIT_FLOAT, min_size=2, max_size=30))
def test_baseline_get_std_is_finite_non_negative(samples) -> None:
    bt = BaselineTracker()
    fv_kwargs: dict[str, float] = {n: 0.0 for n in [
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
    for s in samples:
        fv_kwargs["formality"] = s
        bt.update(InteractionFeatureVector(**fv_kwargs))

    std = bt.get_std("formality")
    assert math.isfinite(std)
    assert std >= 0.0


# ---------------------------------------------------------------------------
# state_classifier property tests
# ---------------------------------------------------------------------------


@settings(max_examples=80, deadline=None)
@given(
    cl=_UNIT_FLOAT,
    formality=_UNIT_FLOAT,
    composition=_NON_NEG_FLOAT,
    edits=_NON_NEG_INT,
    iki_m=_NON_NEG_FLOAT,
    iki_s=_NON_NEG_FLOAT,
    eng=_UNIT_FLOAT,
    dev=_FINITE_FLOAT,
    baseline_est=_BOOL,
    msg_count=_NON_NEG_INT,
)
def test_state_classifier_invariants(
    cl, formality, composition, edits, iki_m, iki_s, eng, dev, baseline_est, msg_count
) -> None:
    """The state classifier never raises, always produces a valid
    state label and a confidence in [0, 1]."""
    label = classify_user_state(
        adaptation={
            "cognitive_load": cl,
            "formality": formality,
            "verbosity": 0.5,
            "accessibility": 0.0,
        },
        composition_time_ms=composition,
        edit_count=edits,
        iki_mean=iki_m,
        iki_std=iki_s,
        engagement_score=eng,
        deviation_from_baseline=dev,
        baseline_established=baseline_est,
        messages_in_session=msg_count,
    )
    assert label.state in {"calm", "focused", "stressed", "tired", "distracted", "warming up"}
    assert math.isfinite(label.confidence)
    assert 0.0 <= label.confidence <= 1.0
    if label.secondary_state is not None:
        assert label.secondary_state in {
            "calm", "focused", "stressed", "tired", "distracted", "warming up"
        }
        assert label.secondary_state != label.state
    # Contributing signals are strings.
    for s in label.contributing_signals:
        assert isinstance(s, str)
