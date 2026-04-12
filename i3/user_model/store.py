"""Async SQLite persistence for user profiles.

Stores and retrieves :class:`~src.user_model.types.UserProfile` objects
using ``aiosqlite``.  All tensor data is serialised as raw bytes via NumPy;
dict fields are stored as JSON strings.

**Privacy guarantee**: this module never stores raw user text.  Only
embeddings, scalar metrics, and metadata are persisted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
import torch

from i3.user_model.types import UserProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class UserModelStoreError(Exception):
    """Base exception for user model store operations."""


class ProfileNotFoundError(UserModelStoreError):
    """Raised when a requested user profile does not exist."""


class ProfileSerializationError(UserModelStoreError):
    """Raised when serialization or deserialization of profile data fails."""


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             TEXT PRIMARY KEY,
    baseline_embedding  BLOB,
    baseline_embedding_dim INTEGER,
    baseline_features_mean TEXT,
    baseline_features_std  TEXT,
    total_sessions      INTEGER DEFAULT 0,
    total_messages      INTEGER DEFAULT 0,
    relationship_strength REAL DEFAULT 0.0,
    long_term_style     TEXT,
    created_at          TIMESTAMP,
    updated_at          TIMESTAMP,
    baseline_established INTEGER DEFAULT 0
);
"""

# SEC: Idempotent migration for legacy databases that pre-date the
# baseline_embedding_dim column. SQLite has no `ADD COLUMN IF NOT EXISTS`,
# so we attempt the ALTER and tolerate the duplicate-column error.
_MIGRATIONS: list[str] = [
    "ALTER TABLE user_profiles ADD COLUMN baseline_embedding_dim INTEGER",
]

# Embedding dimension expected by the encoder.
_EMBEDDING_DIM = 64


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a 1-D torch.Tensor to raw bytes via NumPy.

    Args:
        tensor: A 1-D tensor (typically 64-dim float32).

    Returns:
        Raw ``float32`` byte representation.
    """
    return tensor.detach().cpu().numpy().astype(np.float32).tobytes()


def _bytes_to_tensor(data: bytes, dim: int = _EMBEDDING_DIM) -> torch.Tensor:
    """Reconstruct a 1-D torch.Tensor from raw bytes.

    Args:
        data: Raw ``float32`` bytes.
        dim: Expected number of elements.

    Returns:
        A 1-D float32 tensor with *dim* elements.

    Raises:
        ProfileSerializationError: If the byte length does not match *dim*.
    """
    expected_len = dim * 4  # float32 = 4 bytes
    if len(data) != expected_len:
        raise ProfileSerializationError(
            f"Expected {expected_len} bytes for {dim}-dim float32 tensor, "
            f"got {len(data)} bytes."
        )
    arr = np.frombuffer(data, dtype=np.float32).copy()
    return torch.from_numpy(arr.reshape(dim))


def _dict_to_json(d: dict | None) -> str | None:
    """Serialize a dict to a JSON string, or ``None``."""
    if d is None:
        return None
    # SEC: sort_keys=True for deterministic on-disk representation; this is
    # required for content hashing, golden-file tests, and reproducible
    # diffs across runs.
    return json.dumps(d, sort_keys=True)


def _json_to_dict(s: str | None) -> dict | None:
    """Deserialize a JSON string to a dict, or ``None``."""
    if s is None:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProfileSerializationError(
            f"Failed to deserialize JSON: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Async store
# ---------------------------------------------------------------------------


class UserModelStore:
    """Async SQLite store for :class:`UserProfile` objects.

    Usage::

        async with UserModelStore("data/user_model.db") as store:
            profile = await store.load_profile("user_123")

    The database and table are created automatically on first use.

    Args:
        db_path: Path to the SQLite database file.  Parent directories
            are created if they do not exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    # -- Context manager ---------------------------------------------------

    async def __aenter__(self) -> UserModelStore:
        """Open the database connection and ensure the schema exists."""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        """Close the database connection."""
        await self.close()

    # -- Lifecycle ---------------------------------------------------------

    async def open(self) -> None:
        """Open the database and create the schema if needed.

        Raises:
            UserModelStoreError: If the database cannot be opened.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(self._db_path))
            self._db.row_factory = aiosqlite.Row
            await self._db.execute(_CREATE_TABLE_SQL)
            # SEC: Run idempotent migrations. SQLite raises OperationalError
            # ("duplicate column name") if a migration was already applied;
            # we swallow that specific error and continue.
            for migration in _MIGRATIONS:
                try:
                    await self._db.execute(migration)
                except Exception as mig_exc:  # noqa: BLE001
                    if "duplicate column name" not in str(mig_exc).lower():
                        raise
            await self._db.commit()
            logger.info("UserModelStore opened: %s", self._db_path)
        except Exception as exc:
            raise UserModelStoreError(
                f"Failed to open database at {self._db_path}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the database connection gracefully."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("UserModelStore closed: %s", self._db_path)

    def _ensure_open(self) -> aiosqlite.Connection:
        """Return the active connection or raise.

        Raises:
            UserModelStoreError: If the store has not been opened.
        """
        if self._db is None:
            raise UserModelStoreError(
                "Store is not open. Use 'async with UserModelStore(...)' or "
                "call 'await store.open()' first."
            )
        return self._db

    # -- CRUD operations ---------------------------------------------------

    async def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile from the database.

        Args:
            user_id: The unique user identifier.

        Returns:
            The reconstructed :class:`UserProfile`, or ``None`` if no
            profile exists for *user_id*.

        Raises:
            UserModelStoreError: On database or deserialization errors.
        """
        db = self._ensure_open()
        try:
            # SEC: Use cursor as a context manager so it is closed even on
            # exception. aiosqlite cursors hold a SQLite statement handle
            # that should not be left dangling.
            async with db.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                logger.debug("No profile found for user_id=%s", user_id)
                return None

            # Deserialize embedding
            baseline_embedding: Optional[torch.Tensor] = None
            if row["baseline_embedding"] is not None:
                # SEC: Honour the persisted dimension when present so that the
                # store can survive an encoder dim change without corrupting
                # data on read. Fall back to the default for legacy rows.
                stored_dim = (
                    row["baseline_embedding_dim"]
                    if "baseline_embedding_dim" in row.keys()
                    else None
                )
                dim = int(stored_dim) if stored_dim else _EMBEDDING_DIM
                baseline_embedding = _bytes_to_tensor(
                    row["baseline_embedding"], dim=dim
                )

            # Deserialize dicts
            baseline_features_mean = _json_to_dict(row["baseline_features_mean"])
            baseline_features_std = _json_to_dict(row["baseline_features_std"])
            long_term_style = _json_to_dict(row["long_term_style"]) or {}

            # Parse timestamps
            created_at = _parse_timestamp(row["created_at"])
            updated_at = _parse_timestamp(row["updated_at"])

            profile = UserProfile(
                user_id=row["user_id"],
                baseline_embedding=baseline_embedding,
                baseline_features_mean=baseline_features_mean,
                baseline_features_std=baseline_features_std,
                total_sessions=row["total_sessions"],
                total_messages=row["total_messages"],
                relationship_strength=row["relationship_strength"],
                long_term_style=long_term_style,
                created_at=created_at,
                updated_at=updated_at,
                baseline_established=bool(row["baseline_established"]),
            )
            logger.debug("Loaded profile for user_id=%s", user_id)
            return profile

        except ProfileSerializationError:
            raise
        except Exception as exc:
            raise UserModelStoreError(
                f"Failed to load profile for user_id={user_id}: {exc}"
            ) from exc

    async def save_profile(self, profile: UserProfile) -> None:
        """Save (upsert) a user profile to the database.

        If a profile with the same ``user_id`` already exists it is
        replaced entirely.

        Args:
            profile: The profile to persist.

        Raises:
            UserModelStoreError: On database or serialization errors.
        """
        db = self._ensure_open()
        try:
            baseline_blob: bytes | None = None
            baseline_dim: int | None = None
            if profile.baseline_embedding is not None:
                baseline_blob = _tensor_to_bytes(profile.baseline_embedding)
                # SEC: Persist the embedding dimension alongside the bytes so
                # the store survives encoder-dim changes without corruption.
                baseline_dim = int(profile.baseline_embedding.numel())

            await db.execute(
                """
                INSERT INTO user_profiles (
                    user_id, baseline_embedding, baseline_embedding_dim,
                    baseline_features_mean, baseline_features_std,
                    total_sessions, total_messages,
                    relationship_strength, long_term_style,
                    created_at, updated_at, baseline_established
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    baseline_embedding     = excluded.baseline_embedding,
                    baseline_embedding_dim = excluded.baseline_embedding_dim,
                    baseline_features_mean = excluded.baseline_features_mean,
                    baseline_features_std  = excluded.baseline_features_std,
                    total_sessions         = excluded.total_sessions,
                    total_messages         = excluded.total_messages,
                    relationship_strength  = excluded.relationship_strength,
                    long_term_style        = excluded.long_term_style,
                    updated_at             = excluded.updated_at,
                    baseline_established   = excluded.baseline_established
                """,
                (
                    profile.user_id,
                    baseline_blob,
                    baseline_dim,
                    _dict_to_json(profile.baseline_features_mean),
                    _dict_to_json(profile.baseline_features_std),
                    profile.total_sessions,
                    profile.total_messages,
                    profile.relationship_strength,
                    _dict_to_json(profile.long_term_style),
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                    int(profile.baseline_established),
                ),
            )
            await db.commit()
            logger.debug("Saved profile for user_id=%s", profile.user_id)

        except Exception as exc:
            raise UserModelStoreError(
                f"Failed to save profile for user_id={profile.user_id}: {exc}"
            ) from exc

    async def delete_profile(self, user_id: str) -> None:
        """Delete a user profile from the database.

        No error is raised if the profile does not exist.

        Args:
            user_id: The unique user identifier.

        Raises:
            UserModelStoreError: On database errors.
        """
        db = self._ensure_open()
        try:
            await db.execute(
                "DELETE FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()
            logger.info("Deleted profile for user_id=%s", user_id)
        except Exception as exc:
            raise UserModelStoreError(
                f"Failed to delete profile for user_id={user_id}: {exc}"
            ) from exc

    async def list_users(self) -> list[str]:
        """Return a list of all user IDs with stored profiles.

        Returns:
            Sorted list of user ID strings.

        Raises:
            UserModelStoreError: On database errors.
        """
        db = self._ensure_open()
        try:
            # SEC: cursor as context manager to guarantee statement cleanup.
            async with db.execute(
                "SELECT user_id FROM user_profiles ORDER BY user_id"
            ) as cursor:
                rows = await cursor.fetchall()
            return [row["user_id"] for row in rows]
        except Exception as exc:
            raise UserModelStoreError(
                f"Failed to list users: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Timestamp parsing helper
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str | None) -> datetime:
    """Parse an ISO-format timestamp string, falling back to ``utcnow``.

    Args:
        value: ISO-format string or ``None``.

    Returns:
        A timezone-aware :class:`datetime` (UTC).
    """
    if value is None:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse timestamp '%s', using utcnow", value)
        return datetime.now(timezone.utc)
