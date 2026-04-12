"""Persistent user model for Implicit Interaction Intelligence (I3).

This package maintains a three-timescale representation of each user and
persists it to SQLite.  It stores **only** embeddings, scalar metrics, and
metadata -- never raw text (privacy by architecture).

Three timescales
----------------
1. **Instant state** (:class:`UserState`) -- the most recent 64-dim
   embedding from the TCN encoder.
2. **Session profile** (:class:`SessionState`) -- an Exponential Moving
   Average (EMA) of instant states within the current session.
3. **Long-term profile** (:class:`UserProfile`) -- an EMA of session
   profiles across all sessions, persisted to SQLite.

:class:`UserModel` orchestrates the three timescales and computes
:class:`DeviationMetrics` that describe how the current state differs
from the user's established baseline.

:class:`UserModelStore` provides async SQLite persistence via ``aiosqlite``.

Exported classes
----------------
UserModel          Three-timescale user model with EMA tracking.
UserProfile        Persistent long-term profile (stored in SQLite).
SessionState       Within-session EMA state.
UserState          Instantaneous encoder output.
DeviationMetrics   How the current state deviates from baselines.
UserModelStore     Async SQLite store for user profiles.
"""

from i3.user_model.types import (
    DeviationMetrics,
    SessionState,
    UserProfile,
    UserState,
)
from i3.user_model.model import UserModel
from i3.user_model.store import UserModelStore

__all__ = [
    "UserModel",
    "UserProfile",
    "SessionState",
    "UserState",
    "DeviationMetrics",
    "UserModelStore",
]
