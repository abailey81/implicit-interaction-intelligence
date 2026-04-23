"""Core data types for the persistent user model.

All types are plain dataclasses.  They carry **only** embeddings, scalar
metrics, and metadata -- never raw user text (privacy by architecture).

Three timescales are represented:
- :class:`UserState` -- instantaneous (single encoder output).
- :class:`SessionState` -- within-session EMA of states.
- :class:`UserProfile` -- long-term profile spanning all sessions.

:class:`DeviationMetrics` captures how the current state differs from the
user's established baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# Instantaneous state
# ---------------------------------------------------------------------------

@dataclass
class UserState:
    """Instantaneous user state produced by the TCN encoder.

    Attributes:
        embedding: 64-dim L2-normalised embedding from the encoder.
        timestamp: Unix epoch seconds when the state was captured.
        message_index: Zero-based position of the message within the
            current session.
    """

    embedding: torch.Tensor          # 64-dim from TCN encoder
    timestamp: float
    message_index: int               # Position in session


# ---------------------------------------------------------------------------
# Session-level state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Within-session profile: EMA of user states.

    Maintained by :class:`~src.user_model.model.UserModel` and updated
    after every new encoder output within a single conversation session.

    Attributes:
        embedding: 64-dim EMA of states within this session.
        message_count: Number of messages processed so far in the session.
        start_time: Unix epoch seconds when the session started.
        mean_engagement: Running mean of per-message engagement scores.
        dominant_emotion: Most frequently detected emotion label.
        topics: Accumulated topic labels observed during the session.
        states_history: Ordered list of all :class:`UserState` objects
            captured during this session.
    """

    embedding: torch.Tensor          # 64-dim EMA of states within session
    message_count: int
    start_time: float
    mean_engagement: float
    dominant_emotion: str
    topics: list[str]
    states_history: list[UserState]  # All states in this session


# ---------------------------------------------------------------------------
# Deviation metrics
# ---------------------------------------------------------------------------

@dataclass
class DeviationMetrics:
    """How the current state deviates from established baselines.

    ``current_vs_baseline`` and ``current_vs_session`` are cosine distances
    (1 - cosine_similarity).  Per-feature deviations are z-scores computed
    against the user's long-term feature means and standard deviations.

    Attributes:
        current_vs_baseline: Cosine distance from long-term baseline embedding.
        current_vs_session: Cosine distance from the session-level EMA embedding.
        engagement_score: Derived engagement score in [0, 1].
        magnitude: Overall deviation magnitude (RMS of per-feature z-scores).
        iki_deviation: Z-score deviation of inter-key interval.
        length_deviation: Z-score deviation of message length.
        vocab_deviation: Z-score deviation of vocabulary richness.
        formality_deviation: Z-score deviation of formality.
        speed_deviation: Z-score deviation of composition speed.
        engagement_deviation: Z-score deviation of engagement velocity.
        complexity_deviation: Z-score deviation of complexity.
        pattern_deviation: Aggregate pattern deviation (RMS of all z-scores).
    """

    current_vs_baseline: float       # Cosine distance from long-term baseline
    current_vs_session: float        # Cosine distance from session mean
    engagement_score: float          # Derived from interaction tempo/depth (0-1)
    magnitude: float                 # Overall deviation magnitude

    # Per-feature deviations (z-scores from InteractionFeatureVector baselines)
    iki_deviation: float
    length_deviation: float
    vocab_deviation: float
    formality_deviation: float
    speed_deviation: float
    engagement_deviation: float
    complexity_deviation: float
    pattern_deviation: float         # Overall pattern deviation


# ---------------------------------------------------------------------------
# Persistent user profile
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Persistent user profile spanning all sessions.

    Stored in SQLite via :class:`~src.user_model.store.UserModelStore`.
    Contains only embeddings, scalar metrics, and metadata -- never raw text.

    Attributes:
        user_id: Unique identifier for the user.
        baseline_embedding: Long-term 64-dim baseline embedding (EMA across
            sessions).  ``None`` until the warm-up period completes.
        baseline_features_mean: Per-feature running mean (keyed by feature
            name from :data:`~src.interaction.types.FEATURE_NAMES`).
        baseline_features_std: Per-feature running standard deviation.
        total_sessions: Cumulative number of completed sessions.
        total_messages: Cumulative number of messages across all sessions.
        relationship_strength: Value in [0, 1] that grows logarithmically
            with total interactions, representing familiarity.
        long_term_style: Long-term style preference scores (e.g.
            ``{"formality": 0.6, "verbosity": 0.4}``).
        created_at: When this profile was first created.
        updated_at: When this profile was last modified.
        baseline_established: ``True`` once the warm-up period has completed
            and the baseline embedding is considered reliable.
    """

    user_id: str
    baseline_embedding: torch.Tensor | None          # 64-dim long-term baseline
    baseline_features_mean: dict[str, float] | None   # Mean of each feature
    baseline_features_std: dict[str, float] | None    # Std of each feature
    total_sessions: int
    total_messages: int
    relationship_strength: float                         # 0-1
    long_term_style: dict[str, float]                    # Long-term style prefs
    created_at: datetime
    updated_at: datetime
    baseline_established: bool
