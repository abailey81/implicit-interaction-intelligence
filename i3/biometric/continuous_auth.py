"""Continuous authentication by cosine-drift monitoring.

During a live session, this module tracks the cosine drift of incoming
session embeddings from the user's registered centroid.  When the drift
exceeds three standard deviations of the observed running distribution,
a structured :class:`AuthenticationEvent` is raised -- the signal that
an impostor may have taken over the session, or that the original user
is experiencing a significant behavioural shift (fatigue, injury).

This follows the continuous-authentication protocol of Killourhy &
Maxwell (2009), with a running-mean / running-variance drift statistic
adapted from Welford (1962) for numerical stability.

References
----------
- Killourhy, K. S. & Maxwell, R. A. (2009).  *Comparing anomaly-
  detection algorithms for keystroke dynamics*.  IEEE/IFIP DSN 2009.
- Welford, B. P. (1962).  *Note on a method for calculating corrected
  sums of squares and products*.  Technometrics 4(3), 419-420.
- Monrose, F. & Rubin, A. (1997).  *Authentication via keystroke
  dynamics*.  ACM CCS '97.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from i3.biometric.keystroke_id import KeystrokeBiometricID


# 3-sigma threshold -- the classical "out-of-control" rule from
# statistical process control (Shewhart, 1931).
DEFAULT_SIGMA_THRESHOLD: float = 3.0

# Minimum number of observations before the running std is considered
# reliable enough to use for drift detection.  Below this, events are
# suppressed to avoid spurious alarms on an under-sampled session.
MIN_OBSERVATIONS: int = 8


@dataclass
class AuthenticationEvent:
    """Structured event raised when drift exceeds the sigma threshold.

    Attributes:
        user_id: The user whose session triggered the event.
        session_drift: Current cosine drift (1 - similarity).
        baseline_mean: Running mean of historical drifts for this user.
        baseline_std: Running std of historical drifts.
        sigma_multiplier: How many sigmas out the current drift is.
        severity: Qualitative bucket -- ``"warn"`` (>= 3σ), ``"alert"``
            (>= 4σ), or ``"critical"`` (>= 5σ).
    """

    user_id: str
    session_drift: float
    baseline_mean: float
    baseline_std: float
    sigma_multiplier: float

    @property
    def severity(self) -> str:
        """Return a qualitative severity bucket."""
        if self.sigma_multiplier >= 5.0:
            return "critical"
        if self.sigma_multiplier >= 4.0:
            return "alert"
        return "warn"


@dataclass
class _Welford:
    """Online mean / variance via Welford's (1962) algorithm.

    Numerically stable for long-running sessions; the naive
    ``sum_of_squares - mean^2`` formulation accumulates catastrophic
    cancellation error on streaming data.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        """Incorporate a new observation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Unbiased sample variance.  Zero until at least 2 observations."""
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(max(0.0, self.variance))


@dataclass
class _SessionStats:
    """Per-session Welford accumulator and observation count.

    Kept separate from the identifier class so ``ContinuousAuthentication``
    can be reset per-session without touching the global centroid store.
    """

    welford: _Welford = field(default_factory=_Welford)
    last_drift: float = 0.0


class ContinuousAuthentication:
    """Session-level drift monitor over :class:`KeystrokeBiometricID`.

    Parameters
    ----------
    identifier : KeystrokeBiometricID
        Source of truth for per-user centroids.  The continuous auth
        module only reads from it -- it never mutates centroids.
    sigma_threshold : float, default 3.0
        Number of standard deviations beyond the running drift mean
        that triggers an :class:`AuthenticationEvent`.
    min_observations : int, default 8
        Minimum observations in the Welford accumulator before events
        can fire.  Protects against false positives at session start.

    Notes
    -----
    Instances of this class hold per-session state and should not be
    shared between concurrent sessions.  The typical lifecycle is::

        auth = ContinuousAuthentication(identifier)
        auth.begin_session(user_id)
        for embedding in session_embeddings:
            evt = await auth.observe(embedding)
            if evt is not None:
                handle(evt)
        auth.end_session()
    """

    def __init__(
        self,
        identifier: KeystrokeBiometricID,
        sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if sigma_threshold <= 0:
            raise ValueError(
                f"sigma_threshold must be > 0, got {sigma_threshold}"
            )
        if min_observations < 2:
            raise ValueError(
                f"min_observations must be >= 2, got {min_observations}"
            )
        self._identifier = identifier
        self._sigma_threshold: float = float(sigma_threshold)
        self._min_observations: int = int(min_observations)
        self._active_user: Optional[str] = None
        self._stats: _SessionStats = _SessionStats()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def begin_session(self, user_id: str) -> None:
        """Start a monitored session for ``user_id``.

        Clears any running statistics from a previous session.

        Args:
            user_id: The user under observation.  This must already be
                registered via :meth:`KeystrokeBiometricID.register`;
                otherwise the monitor can never fire (no centroid to
                compare against).
        """
        if not user_id:
            raise ValueError("user_id must be a non-empty string")
        self._active_user = user_id
        self._stats = _SessionStats()

    def end_session(self) -> None:
        """End the currently-monitored session and clear state."""
        self._active_user = None
        self._stats = _SessionStats()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    async def observe(
        self,
        embedding: torch.Tensor,
    ) -> Optional[AuthenticationEvent]:
        """Feed a new session embedding and maybe return an event.

        Args:
            embedding: 1-D 64-dim tensor produced by the TCN encoder.

        Returns:
            An :class:`AuthenticationEvent` when the current drift
            exceeds ``sigma_threshold`` sigmas above the running mean,
            and at least :attr:`_min_observations` samples have been
            accumulated.  ``None`` otherwise.
        """
        if self._active_user is None:
            # Silent no-op rather than raise -- the typical integration
            # sprinkles ``observe`` calls into hot paths, and
            # ``end_session`` races are unavoidable.
            return None

        centroid = await self._identifier.get_centroid(self._active_user)
        if centroid is None:
            # User not yet enrolled -- nothing to compare against.
            return None

        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"embedding must be torch.Tensor, got {type(embedding).__name__}"
            )
        if embedding.dim() != 1 or embedding.numel() != centroid.numel():
            raise ValueError(
                "embedding dimensionality must match the centroid "
                f"({centroid.numel()}), got {tuple(embedding.shape)}"
            )

        q = F.normalize(embedding.detach().cpu().float(), p=2, dim=0)
        similarity = float(torch.dot(centroid, q).item())
        drift = max(0.0, 1.0 - similarity)  # clamp to [0, 2]

        # Update Welford stats BEFORE the threshold check so the mean
        # includes the current observation and the system self-calibrates
        # as the session progresses.
        self._stats.welford.update(drift)
        self._stats.last_drift = drift

        if self._stats.welford.n < self._min_observations:
            return None

        mean = self._stats.welford.mean
        std = self._stats.welford.std
        if std <= 1e-9:
            # Degenerate case -- no variance yet, so the 3-sigma test
            # is meaningless.  Skip rather than fire on every tiny
            # deviation.
            return None

        sigma_mult = (drift - mean) / std
        if sigma_mult >= self._sigma_threshold:
            return AuthenticationEvent(
                user_id=self._active_user,
                session_drift=drift,
                baseline_mean=mean,
                baseline_std=std,
                sigma_multiplier=float(sigma_mult),
            )
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_user(self) -> Optional[str]:
        """The user currently under observation, or ``None``."""
        return self._active_user

    @property
    def observation_count(self) -> int:
        """Number of observations accumulated in the current session."""
        return self._stats.welford.n

    @property
    def drift_mean(self) -> float:
        """Current running-mean drift for the active session."""
        return self._stats.welford.mean

    @property
    def drift_std(self) -> float:
        """Current running-std drift for the active session."""
        return self._stats.welford.std


__all__ = [
    "AuthenticationEvent",
    "ContinuousAuthentication",
    "DEFAULT_SIGMA_THRESHOLD",
    "MIN_OBSERVATIONS",
]
