"""Core data types for the Pipeline Orchestration Engine.

Defines the input/output contracts for the pipeline and the
:class:`EngagementSignal` used to compute reward signals for the
contextual bandit router.

All types are plain dataclasses to keep the pipeline layer free of heavy
framework dependencies.  The :class:`PipelineOutput` carries everything
the frontend dashboard needs to render user-state visualisations, routing
confidence bars, and adaptation gauges.

Privacy note
~~~~~~~~~~~~
:class:`PipelineInput` *does* carry raw ``message_text`` because the
pipeline needs it for feature extraction, routing, and generation.
However, raw text is **never persisted** -- only abstract
representations (embeddings, scalar metrics, topic keywords) reach the
diary store.  The pipeline enforces this guarantee at every write
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pipeline input
# ---------------------------------------------------------------------------

@dataclass
class PipelineInput:
    """Input to the pipeline from a single user message.

    Attributes:
        user_id: Unique user identifier (opaque string).
        session_id: Identifier for the current conversation session.
        message_text: Raw message text submitted by the user.
        timestamp: Unix epoch seconds when the message was submitted.
        composition_time_ms: Total time the user spent composing the
            message (milliseconds), as reported by the client.
        edit_count: Number of edits (cut/paste, undo, etc.) observed
            during composition.
        pause_before_send_ms: Hesitation time between the last keystroke
            and pressing "send" (milliseconds).
        keystroke_timings: List of inter-key interval durations in
            milliseconds, captured from the client WebSocket stream.
    """

    user_id: str
    session_id: str
    message_text: str
    timestamp: float
    composition_time_ms: float
    edit_count: int
    pause_before_send_ms: float
    keystroke_timings: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """Output from the pipeline after processing a single user message.

    Contains the generated response text, the routing decision, timing
    information, and a snapshot of all user-state metrics that the
    frontend dashboard needs for real-time visualisation.

    Attributes:
        response_text: The AI-generated response to the user's message.
        route_chosen: Which generation backend was used -- ``"local_slm"``
            or ``"cloud_llm"``.
        latency_ms: End-to-end pipeline latency in milliseconds.
        user_state_embedding_2d: 2-D projection of the 64-dim user-state
            embedding, suitable for scatter-plot visualisation.
        adaptation: Serialised :class:`~src.adaptation.types.AdaptationVector`
            as a nested dict.
        engagement_score: Current engagement estimate in [0, 1].
        deviation_from_baseline: Cosine distance between the user's
            current state embedding and their long-term baseline.
        routing_confidence: Per-arm selection probabilities from the
            contextual bandit (keys: ``"local_slm"``, ``"cloud_llm"``).
        messages_in_session: Running message count for the active session.
        baseline_established: Whether the user model has accumulated
            enough observations for a reliable baseline.
        diary_entry: Optional diary entry dict, included only when a
            significant event is detected during the exchange.
    """

    response_text: str
    route_chosen: str
    latency_ms: float

    # State update for frontend dashboards
    user_state_embedding_2d: tuple[float, float]
    adaptation: dict
    engagement_score: float
    deviation_from_baseline: float
    routing_confidence: dict[str, float]
    messages_in_session: int
    baseline_established: bool

    # Diary entry (optional, only on significant events)
    diary_entry: Optional[dict] = None


# ---------------------------------------------------------------------------
# Engagement signal
# ---------------------------------------------------------------------------

@dataclass
class EngagementSignal:
    """Engagement signal derived from user behaviour AFTER receiving a response.

    The five sub-signals capture orthogonal aspects of engagement:

    1. **Continued conversation** -- Did the user send another message?
    2. **Response latency** -- How quickly did the user respond?
    3. **Response length ratio** -- Is the user's next message proportional
       to the AI's response?
    4. **Topic continuity** -- Did the user stay on topic?
    5. **Sentiment shift** -- Did the user's sentiment change after the
       AI's response?

    The composite :attr:`score` is the arithmetic mean of these five
    sub-signals, each normalised to [0, 1].

    Attributes:
        continued_conversation: ``True`` if the user sent a follow-up
            message within the session timeout window.
        response_latency_ms: Milliseconds between the AI response being
            delivered and the user starting their next message.
        response_length_ratio: Ratio of the user's next message length
            (in words) to the AI's response length (in words).
        topic_continuity: Estimated topic overlap between the AI's
            response and the user's follow-up, in [0, 1].
        sentiment_shift: Change in sentiment valence between the user's
            previous and current messages, in [-1, 1].
    """

    continued_conversation: bool
    response_latency_ms: float
    response_length_ratio: float
    topic_continuity: float
    sentiment_shift: float

    @property
    def score(self) -> float:
        """Compute an overall engagement score in [0, 1].

        Returns the arithmetic mean of five normalised sub-signals:

        - **Continuation**: 1.0 if the user continued, else 0.0.
        - **Latency**: Faster responses indicate higher engagement.
          Saturates at 30 s (score drops to 0.0 beyond that).
        - **Length ratio**: Longer replies indicate engagement. Capped
          at 1.0.
        - **Topic continuity**: Passed through directly (already [0, 1]).
        - **Sentiment shift**: Mapped from [-1, 1] to [0, 1].
        """
        signals = [
            1.0 if self.continued_conversation else 0.0,
            max(0.0, 1.0 - self.response_latency_ms / 30_000.0),
            min(1.0, self.response_length_ratio),
            self.topic_continuity,
            (self.sentiment_shift + 1.0) / 2.0,
        ]
        return float(np.mean(signals))


# ---------------------------------------------------------------------------
# Engagement estimator (stateless helper)
# ---------------------------------------------------------------------------

class EngagementEstimator:
    """Stateless utility that computes :class:`EngagementSignal` from raw metrics.

    This class does not maintain per-user state -- it simply packages the
    raw metrics into an :class:`EngagementSignal` and returns the composite
    score.  Per-user tracking (previous response time, previous response
    length, etc.) is handled by the :class:`~src.pipeline.engine.Pipeline`.

    Example::

        estimator = EngagementEstimator()
        signal = estimator.compute(
            continued=True,
            response_latency_ms=2500.0,
            user_msg_length=12,
            ai_msg_length=25,
            topic_continuity=0.7,
            sentiment_shift=0.1,
        )
        print(signal.score)  # ~0.72
    """

    def compute(
        self,
        continued: bool,
        response_latency_ms: float,
        user_msg_length: int,
        ai_msg_length: int,
        topic_continuity: float = 0.5,
        sentiment_shift: float = 0.0,
    ) -> EngagementSignal:
        """Build an :class:`EngagementSignal` from raw interaction metrics.

        Args:
            continued: Whether the user sent a follow-up message.
            response_latency_ms: Time between AI response delivery and
                the user's next keystroke (milliseconds).
            user_msg_length: Word count of the user's follow-up message.
            ai_msg_length: Word count of the AI's most recent response.
            topic_continuity: Estimated topic overlap [0, 1].
            sentiment_shift: Sentiment delta [-1, 1].

        Returns:
            A fully populated :class:`EngagementSignal`.
        """
        length_ratio = (
            user_msg_length / max(1, ai_msg_length)
        )
        return EngagementSignal(
            continued_conversation=continued,
            response_latency_ms=response_latency_ms,
            response_length_ratio=length_ratio,
            topic_continuity=max(0.0, min(1.0, topic_continuity)),
            sentiment_shift=max(-1.0, min(1.0, sentiment_shift)),
        )
