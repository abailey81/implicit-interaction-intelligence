"""Real-time interaction monitor for I3.

:class:`InteractionMonitor` sits between the WebSocket transport layer and
the feature extraction pipeline.  It maintains per-user keystroke buffers,
detects typing bursts, computes per-message keystroke metrics, and produces
:class:`InteractionFeatureVector` objects on every submitted message.

All mutable state is protected by asyncio locks so the monitor is safe to
use from concurrent coroutines within a single event loop.

Typical usage::

    monitor = InteractionMonitor(feature_window=10, baseline_warmup=5)

    # On every keystroke WebSocket event:
    await monitor.process_keystroke(user_id, keystroke_event)

    # When the user presses "Send":
    fv = await monitor.process_message(
        user_id, text, composition_time_ms, edit_count, pause_before_send_ms,
    )

    # Retrieve the recent feature window for downstream models:
    window = await monitor.get_feature_window(user_id)
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.interaction.types import (
    InteractionFeatureVector,
    KeystrokeEvent,
)


# ====================================================================
# Per-user session state (internal)
# ====================================================================

@dataclass
class _UserSession:
    """Mutable per-user state managed by :class:`InteractionMonitor`."""

    # Keystroke buffer for the current composition
    keystroke_buffer: list[KeystrokeEvent] = field(default_factory=list)

    # Sliding window of recent feature vectors.
    # PERF (L-13, 2026-04-23 audit): deque instead of list so
    # overflow trim is O(1) via ``maxlen`` rather than the previous
    # ``pop(0)`` which is O(n).  ``maxlen`` is set by the monitor on
    # session construction to mirror ``self._feature_window_size``.
    feature_window: "deque[InteractionFeatureVector]" = field(
        default_factory=lambda: deque(maxlen=10)
    )

    # Baseline tracker
    baseline: BaselineTracker = field(default_factory=lambda: BaselineTracker(warmup=5))

    # Session timing
    session_start_ts: float = field(default_factory=time.time)
    last_message_ts: float = 0.0
    message_count: int = 0

    # Per-user asyncio lock
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ====================================================================
# InteractionMonitor
# ====================================================================

# Pause threshold that separates typing bursts (milliseconds).
_BURST_PAUSE_MS: float = 500.0


class InteractionMonitor:
    """Per-user keystroke buffer and session-level feature tracker.

    Args:
        feature_window: Maximum number of recent feature vectors to retain
            per user.  Older vectors are discarded in FIFO order.
        baseline_warmup: Number of messages required before the user
            baseline is considered established for deviation features.
        expected_session_length: Expected session duration in seconds,
            used for the ``session_progress`` feature.
    """

    def __init__(
        self,
        feature_window: int = 10,
        baseline_warmup: int = 5,
        expected_session_length: float = 600.0,
    ) -> None:
        self._feature_window_size = feature_window
        self._baseline_warmup = baseline_warmup
        self._expected_session_length = expected_session_length
        self._sessions: dict[str, _UserSession] = {}
        self._extractor = FeatureExtractor()
        # SEC: class-level lock around the dict mutation below — prevents
        # two concurrent first-time keystroke calls for the same user_id
        # from each constructing a _UserSession and clobbering each other.
        # Mirrors the asyncio.Lock pattern in Pipeline._aget_or_create_user_model.
        self._sessions_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Session management                                                   #
    # ------------------------------------------------------------------ #

    def _get_or_create_session(self, user_id: str) -> _UserSession:
        """Return the session for *user_id*, creating one if necessary.

        Thread-safe via double-checked locking: the fast path avoids the
        lock when the session already exists; the slow path acquires the
        lock, re-checks, and creates if still missing.
        """
        # Fast path — no lock if already present
        session = self._sessions.get(user_id)
        if session is not None:
            return session
        # SEC: slow path under lock prevents lost-update race
        with self._sessions_lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = _UserSession(
                    baseline=BaselineTracker(warmup=self._baseline_warmup),
                    feature_window=deque(maxlen=self._feature_window_size),
                )
                self._sessions[user_id] = session
            return session

    # ------------------------------------------------------------------ #
    # Keystroke processing                                                 #
    # ------------------------------------------------------------------ #

    async def process_keystroke(
        self, user_id: str, event: KeystrokeEvent
    ) -> None:
        """Buffer a keystroke event for the current composition.

        This method is called on every keystroke WebSocket message.  The
        buffer is consumed (and cleared) when :meth:`process_message` is
        called.

        Args:
            user_id: Unique user identifier.
            event: The keystroke timing record.
        """
        session = self._get_or_create_session(user_id)
        async with session.lock:
            session.keystroke_buffer.append(event)

    # ------------------------------------------------------------------ #
    # Message processing                                                   #
    # ------------------------------------------------------------------ #

    async def process_message(
        self,
        user_id: str,
        text: str,
        composition_time_ms: float = 0.0,
        edit_count: int = 0,
        pause_before_send_ms: float = 0.0,
    ) -> InteractionFeatureVector:
        """Process a submitted message and return the feature vector.

        This method:

        1. Computes keystroke-level metrics from the buffered keystrokes.
        2. Clears the keystroke buffer.
        3. Delegates to :class:`FeatureExtractor` for the full 32-dim
           feature vector.
        4. Updates the baseline tracker and the sliding feature window.

        Args:
            user_id: Unique user identifier.
            text: The submitted message text.
            composition_time_ms: Total time spent composing (ms).
            edit_count: Number of edits (backspaces, cut/paste, etc.).
            pause_before_send_ms: Hesitation time before pressing send (ms).

        Returns:
            The 32-dim :class:`InteractionFeatureVector` for this message.
        """
        session = self._get_or_create_session(user_id)
        async with session.lock:
            # 1. Compute keystroke metrics from buffer
            ks_metrics = self._compute_keystroke_metrics(
                session.keystroke_buffer,
                composition_time_ms,
                edit_count,
                pause_before_send_ms,
            )

            # 2. Clear buffer
            session.keystroke_buffer.clear()

            # 3. Build feature vector
            now = time.time()
            fv = self._extractor.extract(
                keystroke_metrics=ks_metrics,
                message_text=text,
                history=list(session.feature_window),
                baseline=session.baseline,
                session_start_ts=session.session_start_ts,
                current_ts=now,
                expected_session_length=self._expected_session_length,
            )

            # 4. Update baseline and window
            session.baseline.update(fv)
            # PERF (L-13): deque(maxlen=N) trims automatically in O(1).
            session.feature_window.append(fv)

            session.last_message_ts = now
            session.message_count += 1

            return fv

    # ------------------------------------------------------------------ #
    # Feature window access                                                #
    # ------------------------------------------------------------------ #

    async def get_feature_window(
        self, user_id: str
    ) -> list[InteractionFeatureVector]:
        """Return the sliding window of recent feature vectors for a user.

        Returns an empty list if the user has no session.
        """
        session = self._sessions.get(user_id)
        if session is None:
            return []
        async with session.lock:
            return list(session.feature_window)

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    async def reset_user(self, user_id: str) -> None:
        """Clear all state for *user_id*.

        The user's keystroke buffer, feature window, baseline tracker, and
        session counters are all reset.  A new session will be created on
        the next interaction.
        """
        session = self._sessions.get(user_id)
        if session is None:
            return
        async with session.lock:
            self._sessions.pop(user_id, None)

    # ------------------------------------------------------------------ #
    # Internal: keystroke metric computation                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_keystroke_metrics(
        buffer: list[KeystrokeEvent],
        composition_time_ms: float,
        edit_count: int,
        pause_before_send_ms: float,
    ) -> dict[str, float]:
        """Derive aggregate keystroke metrics from the raw buffer.

        Returns a dict with keys expected by
        :meth:`FeatureExtractor.extract`.
        """
        if not buffer:
            # No keystrokes recorded (e.g. paste, or no WebSocket data).
            # Return zeroed metrics; the feature extractor handles this.
            return {
                "mean_iki_ms": 0.0,
                "std_iki_ms": 0.0,
                "mean_burst_length": 0.0,
                "mean_pause_duration_ms": 0.0,
                "backspace_ratio": 0.0,
                "composition_speed_cps": 0.0,
                "pause_before_send_ms": pause_before_send_ms,
                "editing_effort": 0.0,
            }

        # -- Inter-key intervals (exclude the first event which has IKI=0)
        ikis = [
            e.inter_key_interval_ms
            for e in buffer
            if e.inter_key_interval_ms > 0
        ]

        mean_iki = _safe_mean(ikis)
        std_iki = _safe_std(ikis)

        # -- Typing bursts (sequences separated by pauses > threshold) ---
        bursts: list[int] = []  # lengths in characters
        pauses: list[float] = []  # inter-burst pause durations (ms)
        current_burst = 1  # first keystroke starts a burst
        for i in range(1, len(buffer)):
            iki = buffer[i].inter_key_interval_ms
            if iki > _BURST_PAUSE_MS:
                bursts.append(current_burst)
                pauses.append(iki)
                current_burst = 1
            else:
                current_burst += 1
        bursts.append(current_burst)  # final burst

        mean_burst_length = _safe_mean([float(b) for b in bursts])
        mean_pause_duration = _safe_mean(pauses)

        # -- Backspace ratio ----------------------------------------------
        total_keys = len(buffer)
        backspace_count = sum(1 for e in buffer if e.key_type == "backspace")
        backspace_ratio = backspace_count / total_keys if total_keys > 0 else 0.0

        # -- Composition speed (characters per second) --------------------
        char_count = sum(1 for e in buffer if e.key_type == "char")
        if composition_time_ms > 0:
            composition_speed = char_count / (composition_time_ms / 1000.0)
        elif ikis:
            total_time_ms = sum(e.inter_key_interval_ms for e in buffer)
            composition_speed = char_count / max(0.001, total_time_ms / 1000.0)
        else:
            composition_speed = 0.0

        # -- Editing effort (ratio of edits to total keystrokes) ----------
        editing_effort = 0.0
        if total_keys > 0:
            editing_effort = min(1.0, (backspace_count + edit_count) / total_keys)

        return {
            "mean_iki_ms": mean_iki,
            "std_iki_ms": std_iki,
            "mean_burst_length": mean_burst_length,
            "mean_pause_duration_ms": mean_pause_duration,
            "backspace_ratio": backspace_ratio,
            "composition_speed_cps": composition_speed,
            "pause_before_send_ms": pause_before_send_ms,
            "editing_effort": editing_effort,
        }


# ====================================================================
# Utility helpers
# ====================================================================

def _safe_mean(values: list[float]) -> float:
    """Return the mean of *values*, or 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    """Return the population standard deviation of *values*, or 0.0."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
