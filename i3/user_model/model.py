"""Three-timescale persistent user model.

Maintains a representation of each user at three temporal resolutions:

1. **Instant state** -- the most recent 64-dim embedding from the TCN
   encoder (updated every message).
2. **Session profile** -- an Exponential Moving Average (EMA) of instant
   states *within* the current session (``alpha = 0.3``).
3. **Long-term profile** -- an EMA of session profiles *across* sessions
   (``alpha = 0.1``), persisted to SQLite between sessions.

All data stored is limited to embeddings, scalar metrics, and metadata.
Raw user text is never captured or persisted (privacy by architecture).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch

from i3.config import UserModelConfig
from i3.user_model.deviation import DeviationComputer
from i3.user_model.types import (
    DeviationMetrics,
    SessionState,
    UserProfile,
    UserState,
)

if TYPE_CHECKING:
    from i3.interaction.types import InteractionFeatureVector

logger = logging.getLogger(__name__)

# Feature names tracked for per-feature baseline statistics.
# These map to InteractionFeatureVector fields used in deviation computation.
_BASELINE_FEATURE_NAMES: list[str] = [
    "mean_iki",
    "message_length",
    "type_token_ratio",
    "formality",
    "composition_speed",
    "engagement_velocity",
    "flesch_kincaid",
]

# SEC: Saturation point for the relationship strength curve. The previous
# version had `1000` hard-coded inline; lift it to a named constant so it can
# be reasoned about and overridden by tests.
_MAX_MESSAGES_FOR_RELATIONSHIP: int = 1000

# SEC: Hard cap on the per-session UserState history. Without a cap a long
# session would grow this list unboundedly. The newest entries are kept.
_MAX_SESSION_STATES_HISTORY: int = 1000


class UserModel:
    """Three-timescale user representation with deviation tracking.

    Args:
        user_id: Unique identifier for the user.
        config: User model configuration (EMA alphas, warm-up, thresholds).
        profile: Optional pre-loaded profile.  If ``None``, a fresh profile
            is created.

    Example::

        from i3.config import UserModelConfig

        config = UserModelConfig()
        model = UserModel("user_42", config)
        model.start_session()

        # For each message:
        deviation = model.update_state(embedding, features)
        print(deviation.engagement_score)

        # End of conversation:
        summary = model.end_session()
    """

    def __init__(
        self,
        user_id: str,
        config: UserModelConfig,
        profile: UserProfile | None = None,
    ) -> None:
        self.user_id = user_id
        self.config = config
        self._deviation_computer = DeviationComputer()

        # Load or create profile
        if profile is not None:
            self.profile = profile
        else:
            now = datetime.now(timezone.utc)
            self.profile = UserProfile(
                user_id=user_id,
                baseline_embedding=None,
                baseline_features_mean=None,
                baseline_features_std=None,
                total_sessions=0,
                total_messages=0,
                relationship_strength=0.0,
                long_term_style={},
                created_at=now,
                updated_at=now,
                baseline_established=False,
            )

        # Session-level state (initialised by start_session)
        self.current_session: SessionState | None = None

        # Instant state (most recent encoder output)
        self.current_state: UserState | None = None

        # Cached deviation metrics
        self._deviation: DeviationMetrics | None = None

        # Running feature statistics for baseline computation (Welford's)
        self._feature_count: int = 0
        self._feature_mean: dict[str, float] = {}
        self._feature_m2: dict[str, float] = {}  # sum of squared diffs

        # Restore from profile if available
        if self.profile.baseline_features_mean is not None:
            self._feature_mean = dict(self.profile.baseline_features_mean)
        if self.profile.baseline_features_std is not None:
            # SEC: Welford reconstruction. For sample variance,
            # std = sqrt(M2 / (n - 1))  =>  M2 = std^2 * (n - 1).
            # Previous code used (std^2 * n) which over-inflated variance
            # and produced inconsistent statistics across restore cycles.
            n = max(self.profile.total_messages, 1)
            self._feature_count = n
            denom = max(n - 1, 1)
            self._feature_m2 = {
                k: (v * v) * denom
                for k, v in self.profile.baseline_features_std.items()
            }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Initialize a new session.

        Resets the session-level EMA and state history.  Must be called
        before :meth:`update_state`.
        """
        now = time.time()
        self.current_session = SessionState(
            embedding=torch.zeros(64),
            message_count=0,
            start_time=now,
            mean_engagement=0.0,
            dominant_emotion="neutral",
            topics=[],
            states_history=[],
        )
        self.current_state = None
        self._deviation = None
        logger.info(
            "Session started for user_id=%s (session #%d)",
            self.user_id,
            self.profile.total_sessions + 1,
        )

    def end_session(self) -> dict:
        """End the current session and update the long-term profile.

        Updates the long-term baseline embedding via EMA, increments
        session/message counters, and computes relationship strength.

        Returns:
            Session summary dict suitable for the interaction diary.
            Contains only metadata and scalar metrics -- no raw text.

        Raises:
            RuntimeError: If no session is active.
        """
        if self.current_session is None:
            raise RuntimeError("No active session to end.")

        session = self.current_session
        now = datetime.now(timezone.utc)

        # Update long-term embedding EMA
        if session.message_count > 0 and session.embedding is not None:
            self._update_longterm_ema(session.embedding)

        # Update counters
        self.profile.total_sessions += 1
        self.profile.total_messages += session.message_count
        self.profile.updated_at = now

        # Update relationship strength (logarithmic growth, capped at [0, 1]).
        # SEC: clamp on both ends and guard total < 0 in case of restore from
        # a corrupted profile.
        total = max(self.profile.total_messages, 0)
        self.profile.relationship_strength = max(
            0.0,
            min(
                1.0,
                math.log1p(total)
                / math.log1p(_MAX_MESSAGES_FOR_RELATIONSHIP),
            ),
        )

        # Persist feature baselines
        if self._feature_mean:
            self.profile.baseline_features_mean = dict(self._feature_mean)
        if self._feature_m2 and self._feature_count > 1:
            self.profile.baseline_features_std = {
                k: math.sqrt(m2 / max(self._feature_count - 1, 1))
                for k, m2 in self._feature_m2.items()
            }

        # Check baseline establishment
        if (
            not self.profile.baseline_established
            and self.profile.total_messages >= self.config.baseline_warmup
            and self.profile.baseline_embedding is not None
        ):
            self.profile.baseline_established = True
            logger.info(
                "Baseline established for user_id=%s after %d messages",
                self.user_id,
                self.profile.total_messages,
            )

        # Build session summary (metadata only, no raw text)
        duration = time.time() - session.start_time
        summary = {
            "user_id": self.user_id,
            "session_number": self.profile.total_sessions,
            "message_count": session.message_count,
            "duration_seconds": round(duration, 2),
            "mean_engagement": round(session.mean_engagement, 4),
            "dominant_emotion": session.dominant_emotion,
            "topics": session.topics,
            "relationship_strength": round(
                self.profile.relationship_strength, 4
            ),
            "baseline_established": self.profile.baseline_established,
        }

        # Clear session state
        self.current_session = None
        self.current_state = None
        self._deviation = None

        logger.info(
            "Session ended for user_id=%s: %d messages in %.1fs",
            self.user_id,
            summary["message_count"],
            duration,
        )
        return summary

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(
        self,
        embedding: torch.Tensor,
        features: InteractionFeatureVector,
    ) -> DeviationMetrics:
        """Update the user model with a new encoder output.

        This is the main entry point called after each user message is
        processed by the TCN encoder.

        Steps:
            1. Store as current instant state.
            2. Update session EMA.
            3. Update per-feature running statistics (Welford's algorithm).
            4. Compute deviation from baseline (if established).
            5. Update baseline embedding after warm-up messages.

        Args:
            embedding: 64-dim embedding from the TCN encoder.
            features: Current interaction feature vector.

        Returns:
            :class:`DeviationMetrics` describing how the current state
            deviates from established baselines.

        Raises:
            RuntimeError: If no session is active.
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active session. Call start_session() first."
            )

        now = time.time()
        session = self.current_session

        # 1. Store as current instant state
        self.current_state = UserState(
            embedding=embedding.detach().clone(),
            timestamp=now,
            message_index=session.message_count,
        )
        session.states_history.append(self.current_state)
        # SEC: Cap states_history so a single long-running session cannot
        # grow memory without bound. Drop oldest entries; the model only
        # uses the most recent ones for EMA derivation anyway.
        if len(session.states_history) > _MAX_SESSION_STATES_HISTORY:
            overflow = len(session.states_history) - _MAX_SESSION_STATES_HISTORY
            del session.states_history[:overflow]
        session.message_count += 1

        # 2. Update session EMA
        self._update_session_ema(embedding)

        # 3. Update per-feature running statistics (Welford's online algorithm)
        self._update_feature_statistics(features)

        # 4. Compute deviation metrics
        self._deviation = self._deviation_computer.compute(
            current_embedding=embedding,
            session_embedding=session.embedding,
            baseline_embedding=self.profile.baseline_embedding,
            features=features,
            baseline_mean=self.profile.baseline_features_mean,
            baseline_std=self.profile.baseline_features_std,
        )

        # 5. Update running engagement mean
        alpha_eng = 0.3
        session.mean_engagement = (
            alpha_eng * self._deviation.engagement_score
            + (1 - alpha_eng) * session.mean_engagement
        )

        # 6. Bootstrap baseline embedding once warm-up is reached.
        # SEC: The long-term baseline is normally updated at session END via
        # `_update_longterm_ema(session.embedding)`. Per-message updates to the
        # baseline cause double-counting against the session-level EMA. We
        # therefore only seed the baseline here when (a) the warm-up count
        # has been reached AND (b) no baseline exists yet, so the very first
        # session can flip `baseline_established` mid-session and produce
        # non-zero deviation metrics on subsequent messages.
        total_msgs = self.profile.total_messages + session.message_count
        if (
            total_msgs >= self.config.baseline_warmup
            and self.profile.baseline_embedding is None
        ):
            self._update_baseline_embedding(session.embedding)
        if (
            total_msgs >= self.config.baseline_warmup
            and self.profile.baseline_embedding is not None
            and not self.profile.baseline_established
        ):
            self.profile.baseline_established = True
            logger.info(
                "Baseline established for user_id=%s at message %d",
                self.user_id,
                total_msgs,
            )

        return self._deviation

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def deviation_from_baseline(self) -> DeviationMetrics:
        """Current deviation metrics.

        Returns:
            The most recently computed :class:`DeviationMetrics`, or a
            zero-initialised instance if no state has been processed yet.
        """
        if self._deviation is not None:
            return self._deviation
        return DeviationMetrics(
            current_vs_baseline=0.0,
            current_vs_session=0.0,
            engagement_score=0.0,
            magnitude=0.0,
            iki_deviation=0.0,
            length_deviation=0.0,
            vocab_deviation=0.0,
            formality_deviation=0.0,
            speed_deviation=0.0,
            engagement_deviation=0.0,
            complexity_deviation=0.0,
            pattern_deviation=0.0,
        )

    @property
    def engagement_score(self) -> float:
        """Current engagement score derived from interaction patterns.

        Returns:
            Float in [0, 1], or 0.0 if no state has been processed.
        """
        if self._deviation is not None:
            return self._deviation.engagement_score
        return 0.0

    @property
    def baseline_established(self) -> bool:
        """Whether the baseline has been established after warm-up.

        The baseline is considered established once the total number of
        messages processed (across all sessions) reaches
        ``config.baseline_warmup``.
        """
        return self.profile.baseline_established

    # ------------------------------------------------------------------
    # Internal EMA helpers
    # ------------------------------------------------------------------

    def _update_session_ema(self, new_embedding: torch.Tensor) -> None:
        """Update the session-level EMA with a new embedding.

        Uses ``config.session_ema_alpha`` as the smoothing factor.

        Args:
            new_embedding: 64-dim embedding from the encoder.
        """
        session = self.current_session
        assert session is not None

        # SEC: Always operate on a detached copy of the input embedding so
        # gradients never propagate into the long-lived state and so the
        # caller cannot mutate state via aliasing.
        new_emb = new_embedding.detach()

        if session.message_count <= 1:
            # First message in session: initialise directly (no EMA on empty).
            session.embedding = new_emb.clone()
        else:
            alpha = self.config.session_ema_alpha
            session.embedding = (
                alpha * new_emb + (1 - alpha) * session.embedding.detach()
            ).clone()

    def _update_longterm_ema(self, session_embedding: torch.Tensor) -> None:
        """Update the long-term baseline via EMA of session embeddings.

        Uses ``config.longterm_ema_alpha`` as the smoothing factor.

        Args:
            session_embedding: Final session-level EMA embedding.
        """
        # SEC: detach + clone to avoid grad propagation and aliasing.
        sess = session_embedding.detach()
        if self.profile.baseline_embedding is None:
            self.profile.baseline_embedding = sess.clone()
        else:
            alpha = self.config.longterm_ema_alpha
            self.profile.baseline_embedding = (
                alpha * sess
                + (1 - alpha) * self.profile.baseline_embedding.detach()
            ).clone()

    def _update_baseline_embedding(self, new_embedding: torch.Tensor) -> None:
        """Incrementally update the baseline embedding with a new observation.

        Uses a slow EMA (``longterm_ema_alpha``) so the baseline drifts
        gradually and captures long-term tendencies rather than transient
        fluctuations.

        Args:
            new_embedding: 64-dim embedding from the encoder.
        """
        # SEC: detach + clone to avoid grad propagation and aliasing.
        new_emb = new_embedding.detach()
        if self.profile.baseline_embedding is None:
            self.profile.baseline_embedding = new_emb.clone()
        else:
            alpha = self.config.longterm_ema_alpha
            self.profile.baseline_embedding = (
                alpha * new_emb
                + (1 - alpha) * self.profile.baseline_embedding.detach()
            ).clone()

    # ------------------------------------------------------------------
    # Test / debug helpers
    # ------------------------------------------------------------------

    def reset_statistics(self) -> None:
        """Reset the running Welford statistics.

        Intended for unit tests and controlled re-baselining. Clears the
        in-memory feature mean/M2 buffers and the cached count, but does
        not delete the persisted profile.
        """
        # SEC: Resetable state for tests (audit checklist requirement).
        self._feature_count = 0
        self._feature_mean = {}
        self._feature_m2 = {}

    # ------------------------------------------------------------------
    # Feature statistics (Welford's online algorithm)
    # ------------------------------------------------------------------

    def _update_feature_statistics(
        self, features: InteractionFeatureVector
    ) -> None:
        """Update running mean and variance for baseline features.

        Uses `Welford's online algorithm
        <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_
        to compute mean and variance in a single pass without storing all
        observations.

        Args:
            features: Current interaction feature vector.
        """
        # SEC: NaN guard - skip the entire feature vector if any tracked field
        # is NaN/Inf to avoid corrupting the running statistics with garbage.
        raw_values: dict[str, float] = {}
        for name in _BASELINE_FEATURE_NAMES:
            value = float(getattr(features, name, 0.0))
            if not math.isfinite(value):
                logger.warning(
                    "Skipping feature update for user_id=%s: non-finite "
                    "value for '%s' (%r)",
                    self.user_id,
                    name,
                    value,
                )
                return
            raw_values[name] = value

        self._feature_count += 1
        n = self._feature_count

        for name, value in raw_values.items():
            # SEC: Use the standard Welford recurrence:
            #   delta      = x - mean_old
            #   mean_new   = mean_old + delta / n
            #   M2        += delta * (x - mean_new)
            # First-ever observation (n == 1) seeds mean = value, M2 = 0.
            old_mean = self._feature_mean.get(name, 0.0)
            if n == 1 or name not in self._feature_mean:
                self._feature_mean[name] = value
                self._feature_m2[name] = 0.0
                continue
            delta = value - old_mean
            new_mean = old_mean + delta / n
            m2_old = self._feature_m2.get(name, 0.0)
            # SEC: Guard against negative M2 from float drift.
            self._feature_m2[name] = max(
                0.0, m2_old + delta * (value - new_mean)
            )
            self._feature_mean[name] = new_mean

        # Update profile with latest statistics
        self.profile.baseline_features_mean = dict(self._feature_mean)
        if n > 1:
            # SEC: max(variance, 0) before sqrt protects against any residual
            # negative M2 from float drift.
            self.profile.baseline_features_std = {
                k: math.sqrt(max(m2 / (n - 1), 0.0))
                for k, m2 in self._feature_m2.items()
            }
