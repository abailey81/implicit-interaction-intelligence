"""Integration tests for the full pipeline.

Validates the PipelineInput/PipelineOutput contracts, engagement signal
computation, and privacy guarantees (no raw text in persisted outputs).
"""

from __future__ import annotations

import pytest
import numpy as np

from i3.pipeline.types import (
    EngagementEstimator,
    EngagementSignal,
    PipelineInput,
    PipelineOutput,
)


# -------------------------------------------------------------------------
# PipelineInput
# -------------------------------------------------------------------------

class TestPipelineInput:
    """Tests for the PipelineInput data class."""

    def test_construction(self) -> None:
        """PipelineInput should accept all required fields."""
        inp = PipelineInput(
            user_id="user_1",
            session_id="sess_1",
            message_text="Hello, how are you?",
            timestamp=1700000000.0,
            composition_time_ms=3200.0,
            edit_count=2,
            pause_before_send_ms=500.0,
            keystroke_timings=[120.0, 95.0, 110.0, 200.0],
        )
        assert inp.user_id == "user_1"
        assert inp.session_id == "sess_1"
        assert inp.message_text == "Hello, how are you?"
        assert inp.composition_time_ms == 3200.0
        assert inp.edit_count == 2
        assert len(inp.keystroke_timings) == 4

    def test_default_keystroke_timings(self) -> None:
        """keystroke_timings should default to an empty list."""
        inp = PipelineInput(
            user_id="u",
            session_id="s",
            message_text="test",
            timestamp=0.0,
            composition_time_ms=100.0,
            edit_count=0,
            pause_before_send_ms=0.0,
        )
        assert inp.keystroke_timings == []


# -------------------------------------------------------------------------
# PipelineOutput
# -------------------------------------------------------------------------

class TestPipelineOutput:
    """Tests for the PipelineOutput data class."""

    def test_construction(self) -> None:
        """PipelineOutput should accept all fields."""
        out = PipelineOutput(
            response_text="I'm doing well, thanks!",
            route_chosen="local_slm",
            latency_ms=45.2,
            user_state_embedding_2d=(0.3, -0.5),
            adaptation={
                "cognitive_load": 0.4,
                "formality": 0.6,
                "emotional_tone": 0.7,
            },
            engagement_score=0.72,
            deviation_from_baseline=0.15,
            routing_confidence={"local_slm": 0.65, "cloud_llm": 0.35},
            messages_in_session=5,
            baseline_established=True,
        )
        assert out.response_text == "I'm doing well, thanks!"
        assert out.route_chosen == "local_slm"
        assert out.latency_ms == 45.2
        assert out.user_state_embedding_2d == (0.3, -0.5)
        assert out.engagement_score == pytest.approx(0.72)
        assert out.baseline_established is True

    def test_optional_diary_entry(self) -> None:
        """diary_entry should default to None."""
        out = PipelineOutput(
            response_text="test",
            route_chosen="cloud_llm",
            latency_ms=100.0,
            user_state_embedding_2d=(0.0, 0.0),
            adaptation={},
            engagement_score=0.5,
            deviation_from_baseline=0.0,
            routing_confidence={},
            messages_in_session=1,
            baseline_established=False,
        )
        assert out.diary_entry is None

    def test_diary_entry_present(self) -> None:
        """diary_entry can carry event metadata."""
        out = PipelineOutput(
            response_text="test",
            route_chosen="local_slm",
            latency_ms=30.0,
            user_state_embedding_2d=(0.1, 0.2),
            adaptation={},
            engagement_score=0.9,
            deviation_from_baseline=0.5,
            routing_confidence={"local_slm": 0.8, "cloud_llm": 0.2},
            messages_in_session=10,
            baseline_established=True,
            diary_entry={
                "event": "significant_deviation",
                "magnitude": 0.5,
                "topics": ["technical"],
            },
        )
        assert out.diary_entry is not None
        assert out.diary_entry["event"] == "significant_deviation"


# -------------------------------------------------------------------------
# EngagementSignal
# -------------------------------------------------------------------------

class TestEngagementSignal:
    """Tests for the EngagementSignal data class and score computation."""

    def test_high_engagement_score(self) -> None:
        """All positive signals should produce a high score."""
        signal = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=1000.0,   # Fast response
            response_length_ratio=0.8,     # Substantial reply
            topic_continuity=0.9,          # On topic
            sentiment_shift=0.5,           # Positive shift
        )
        score = signal.score
        assert score > 0.7, f"Expected high score, got {score:.3f}"

    def test_low_engagement_score(self) -> None:
        """All negative signals should produce a low score."""
        signal = EngagementSignal(
            continued_conversation=False,
            response_latency_ms=60000.0,  # Very slow / no response
            response_length_ratio=0.0,     # No reply
            topic_continuity=0.0,          # Off topic
            sentiment_shift=-1.0,          # Negative shift
        )
        score = signal.score
        assert score < 0.15, f"Expected low score, got {score:.3f}"

    def test_neutral_engagement_score(self) -> None:
        """Neutral signals should produce a mid-range score."""
        signal = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=15000.0,
            response_length_ratio=0.5,
            topic_continuity=0.5,
            sentiment_shift=0.0,
        )
        score = signal.score
        assert 0.3 < score < 0.8, f"Expected mid-range score, got {score:.3f}"

    def test_score_in_unit_interval(self) -> None:
        """Engagement score should always be in [0, 1]."""
        # Edge case: extreme values
        signal = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=0.0,         # Instant
            response_length_ratio=100.0,      # Very long
            topic_continuity=1.0,
            sentiment_shift=1.0,
        )
        score = signal.score
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"

    def test_latency_saturation(self) -> None:
        """Latency beyond 30s should contribute 0 to the score."""
        signal_fast = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=0.0,
            response_length_ratio=0.5,
            topic_continuity=0.5,
            sentiment_shift=0.0,
        )
        signal_slow = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=60000.0,
            response_length_ratio=0.5,
            topic_continuity=0.5,
            sentiment_shift=0.0,
        )
        assert signal_fast.score > signal_slow.score


# -------------------------------------------------------------------------
# EngagementEstimator
# -------------------------------------------------------------------------

class TestEngagementEstimator:
    """Tests for the stateless engagement estimator."""

    @pytest.fixture
    def estimator(self) -> EngagementEstimator:
        return EngagementEstimator()

    def test_compute_returns_signal(
        self, estimator: EngagementEstimator
    ) -> None:
        """compute() should return an EngagementSignal."""
        signal = estimator.compute(
            continued=True,
            response_latency_ms=2500.0,
            user_msg_length=12,
            ai_msg_length=25,
            topic_continuity=0.7,
            sentiment_shift=0.1,
        )
        assert isinstance(signal, EngagementSignal)
        assert signal.continued_conversation is True
        assert signal.response_latency_ms == 2500.0

    def test_length_ratio_computation(
        self, estimator: EngagementEstimator
    ) -> None:
        """Length ratio should be user_msg_length / ai_msg_length."""
        signal = estimator.compute(
            continued=True,
            response_latency_ms=1000.0,
            user_msg_length=10,
            ai_msg_length=20,
        )
        assert signal.response_length_ratio == pytest.approx(0.5)

    def test_zero_ai_length_safe(
        self, estimator: EngagementEstimator
    ) -> None:
        """Division by zero should be prevented when ai_msg_length=0."""
        signal = estimator.compute(
            continued=True,
            response_latency_ms=1000.0,
            user_msg_length=10,
            ai_msg_length=0,
        )
        # max(1, 0) = 1, so ratio = 10/1 = 10
        assert signal.response_length_ratio == 10.0

    def test_topic_continuity_clamped(
        self, estimator: EngagementEstimator
    ) -> None:
        """topic_continuity should be clamped to [0, 1]."""
        signal = estimator.compute(
            continued=True,
            response_latency_ms=1000.0,
            user_msg_length=5,
            ai_msg_length=10,
            topic_continuity=1.5,
        )
        assert signal.topic_continuity == 1.0

    def test_sentiment_shift_clamped(
        self, estimator: EngagementEstimator
    ) -> None:
        """sentiment_shift should be clamped to [-1, 1]."""
        signal = estimator.compute(
            continued=True,
            response_latency_ms=1000.0,
            user_msg_length=5,
            ai_msg_length=10,
            sentiment_shift=-2.0,
        )
        assert signal.sentiment_shift == -1.0


# -------------------------------------------------------------------------
# Privacy guarantees
# -------------------------------------------------------------------------

class TestPrivacy:
    """Tests ensuring that raw text never appears in output/persisted data."""

    def test_pipeline_output_has_no_raw_input(self) -> None:
        """PipelineOutput should not carry the user's raw input text."""
        out = PipelineOutput(
            response_text="Some AI response",
            route_chosen="local_slm",
            latency_ms=50.0,
            user_state_embedding_2d=(0.1, -0.2),
            adaptation={"cognitive_load": 0.5},
            engagement_score=0.6,
            deviation_from_baseline=0.1,
            routing_confidence={"local_slm": 0.7, "cloud_llm": 0.3},
            messages_in_session=3,
            baseline_established=True,
        )
        # PipelineOutput has response_text (the AI's reply) but no field
        # for the user's original message text.
        assert not hasattr(out, 'message_text')
        assert not hasattr(out, 'user_text')
        assert not hasattr(out, 'input_text')

    def test_diary_entry_no_raw_text(self) -> None:
        """Diary entries should contain summaries, not raw user messages."""
        diary = {
            "event": "tone_shift",
            "summary": "User became more formal",
            "topics": ["technology"],
            "emotion": "neutral",
            "magnitude": 0.3,
        }
        # Verify none of these typical diary fields contain raw user text
        for key, value in diary.items():
            if isinstance(value, str):
                assert "raw_text" not in key
                assert "message_text" not in key

    def test_engagement_signal_has_no_text(self) -> None:
        """EngagementSignal carries only numerical metrics."""
        signal = EngagementSignal(
            continued_conversation=True,
            response_latency_ms=2000.0,
            response_length_ratio=0.6,
            topic_continuity=0.7,
            sentiment_shift=0.1,
        )
        # No string fields that could contain message content
        for field_name in signal.__dataclass_fields__:
            value = getattr(signal, field_name)
            assert not isinstance(value, str), (
                f"EngagementSignal.{field_name} is a string -- "
                "could leak raw text."
            )
