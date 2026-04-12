"""Data types for the Intelligent Router.

Defines the core value objects used throughout the routing subsystem:
routing context vectors, route choices, and routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RouteChoice(Enum):
    """Available routing destinations."""

    LOCAL_SLM = "local_slm"
    CLOUD_LLM = "cloud_llm"


@dataclass
class RoutingContext:
    """12-dimensional context vector for the contextual bandit.

    This context captures everything the bandit needs to make an informed
    routing decision: compressed user state, query characteristics,
    session metadata, and system-level estimates.

    Attributes:
        user_state_compressed: 4-dimensional PCA projection of the full
            64-dimensional user-state embedding from the encoder.
        query_complexity: Estimated complexity of the current query (0-1).
        topic_sensitivity: Estimated privacy sensitivity of the topic (0-1).
        user_patience: Estimated patience from interaction tempo (0-1).
            Lower values mean the user types/clicks rapidly and expects
            fast responses.
        session_progress: Normalised progress through the session (0-1).
        baseline_established: Whether the user model has accumulated
            enough observations for a reliable baseline.
        previous_route: Which route was chosen last turn.
            0 = SLM, 1 = Cloud, -1 = no previous turn.
        previous_engagement: Engagement signal from the previous turn (0-1).
        time_of_day: Normalised hour of day (0-1), where 0.0 = midnight
            and 0.5 = noon.
        message_count: Number of messages exchanged in the current session.
        cloud_latency_est: Estimated cloud round-trip latency, normalised
            to [0, 1] where 1 represents the configured timeout ceiling.
        slm_confidence: The local SLM's self-reported confidence on the
            current input (0-1).  When unavailable, defaults to 0.5.
    """

    user_state_compressed: list[float]
    query_complexity: float
    topic_sensitivity: float
    user_patience: float
    session_progress: float
    baseline_established: bool
    previous_route: int  # 0=SLM, 1=Cloud, -1=none
    previous_engagement: float
    time_of_day: float
    message_count: int
    cloud_latency_est: float
    slm_confidence: float

    def to_vector(self) -> np.ndarray:
        """Flatten the context into a 12-dimensional float64 vector.

        The vector layout is:
            [0:4]  user_state_compressed (4 dims)
            [4]    query_complexity
            [5]    topic_sensitivity
            [6]    user_patience
            [7]    session_progress
            [8]    baseline_established (0.0 or 1.0)
            [9]    previous_route (clamped to 0 if negative)
            [10]   previous_engagement
            [11]   time_of_day

        Returns:
            A 12-element numpy array of dtype float64.
        """
        return np.array(
            [
                *self.user_state_compressed,       # 4 dims
                self.query_complexity,             # dim 4
                self.topic_sensitivity,            # dim 5
                self.user_patience,                # dim 6
                self.session_progress,             # dim 7
                float(self.baseline_established),  # dim 8
                float(max(0, self.previous_route)),  # dim 9
                self.previous_engagement,          # dim 10
                self.time_of_day,                  # dim 11
            ],
            dtype=np.float64,
        )

    def validate(self) -> list[str]:
        """Run basic sanity checks and return a list of warnings (empty = OK).

        Does not raise; the caller decides how to handle violations.
        """
        warnings: list[str] = []
        if len(self.user_state_compressed) != 4:
            warnings.append(
                f"user_state_compressed has {len(self.user_state_compressed)} "
                f"dims, expected 4"
            )
        for name, val in [
            ("query_complexity", self.query_complexity),
            ("topic_sensitivity", self.topic_sensitivity),
            ("user_patience", self.user_patience),
            ("session_progress", self.session_progress),
            ("previous_engagement", self.previous_engagement),
            ("time_of_day", self.time_of_day),
            ("cloud_latency_est", self.cloud_latency_est),
            ("slm_confidence", self.slm_confidence),
        ]:
            if not (0.0 <= val <= 1.0):
                warnings.append(f"{name}={val} is outside [0, 1]")
        if self.previous_route not in (-1, 0, 1):
            warnings.append(f"previous_route={self.previous_route} not in {{-1, 0, 1}}")
        if self.message_count < 0:
            warnings.append(f"message_count={self.message_count} is negative")
        return warnings


@dataclass
class RoutingDecision:
    """The output of the Intelligent Router for a single query.

    Attributes:
        chosen_route: Which destination was selected.
        confidence: Mapping from route name to its estimated selection
            probability (sums to ~1.0).
        context: The routing context that was used for the decision.
        was_privacy_override: True if the decision was forced to LOCAL_SLM
            because the query contained sensitive content.
        reasoning: A short, human-readable explanation of why this route
            was chosen (useful for logging / debugging).
    """

    chosen_route: RouteChoice
    confidence: dict[str, float]
    context: RoutingContext
    was_privacy_override: bool
    reasoning: str
