"""Async SQLite persistence for the Interaction Diary.

The store manages two tables:

- **sessions** -- One row per interaction session, storing aggregated
  metrics (mean engagement, cognitive load, accessibility), the dominant
  emotion, topic keywords, and an optional natural-language summary.
- **exchanges** -- One row per user-AI message exchange within a session,
  storing the user-state embedding (as a BLOB), the adaptation vector
  (JSON), the router decision, latency, engagement, and topic keywords.

PRIVACY GUARANTEE
~~~~~~~~~~~~~~~~~
No raw user text or AI response text is ever written to the database.
Only embeddings, scalar metrics, topic keywords, and adaptation parameters
are persisted.  This is enforced at the API boundary -- callers pass
pre-processed data, never raw strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    import torch

    from i3.privacy.encryption import ModelEncryptor

logger = logging.getLogger(__name__)

# SEC: Envelope format for encrypted user_state_embedding blobs.
#   byte 0       — version/flag:  0x00 = plaintext, 0x01 = Fernet-encrypted
#   bytes 1..N   — payload
# Mirrors the format used by :mod:`i3.user_model.store` so tooling can
# detect encryption by inspecting the first byte of any embedding column.
_ENC_VERSION_PLAINTEXT = 0x00
_ENC_VERSION_FERNET_V1 = 0x01


def encrypt_embedding_envelope(
    tensor: torch.Tensor,
    encryptor: ModelEncryptor | None,
) -> bytes:
    """Encode a 1-D torch tensor as a versioned envelope for persistence.

    When *encryptor* is provided, the payload is Fernet-encrypted. Otherwise
    a plaintext envelope (still versioned) is produced so the read path is
    uniform.  This helper exists at module scope so :class:`Pipeline` can
    invoke it without pulling the store's full async machinery into the
    hot path.
    """
    import numpy as np  # local to avoid numpy as a hard store-import dep

    if encryptor is None:
        raw = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        return bytes([_ENC_VERSION_PLAINTEXT]) + raw
    payload = encryptor.encrypt_embedding(tensor)
    return bytes([_ENC_VERSION_FERNET_V1]) + payload


class DiaryStore:
    """Async SQLite store for the interaction diary.

    Schema
    ------
    - ``sessions``: session-level summaries and aggregated metrics.
    - ``exchanges``: per-message exchange records with embeddings, adaptation
      vectors, and engagement signals.

    PRIVACY: No raw user text is ever stored.  Only embeddings, scalar
    metrics, topic keywords, and adaptation parameters.

    Parameters
    ----------
    db_path:
        Filesystem path where the SQLite database will be created or opened.
        Parent directories must already exist.
    """

    def __init__(
        self,
        db_path: str = "data/diary.db",
        encryptor: ModelEncryptor | None = None,
    ) -> None:
        self.db_path = db_path
        self._initialized = False
        # SEC: When a ModelEncryptor is supplied, the caller is responsible
        # for calling :meth:`encrypt_embedding_envelope` before passing bytes
        # to :meth:`log_exchange`.  The store itself does not touch tensors —
        # but it documents the envelope format so auditors can verify the
        # privacy contract end-to-end.
        self._encryptor = encryptor
        # PERF/SEC (H-2, 2026-04-23 audit): hold a single ``aiosqlite``
        # connection for the lifetime of the store.  The previous
        # implementation opened a fresh connection on every call, which
        # cost 5-30 ms per message *and* defeated ``PRAGMA foreign_keys``
        # (a per-connection pragma) on every subsequent op.
        self._db: aiosqlite.Connection | None = None
        # Serialises writes on the shared connection.  aiosqlite's worker
        # thread serialises at the driver level, but an explicit async
        # lock guards transactional multi-statement sequences.
        self._write_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and indices if they do not already exist.

        This method is idempotent and safe to call multiple times.  It must
        be awaited before any read/write operations.  Also opens the
        persistent connection used by every subsequent op.
        """
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            # SEC: FK enforcement + WAL journal for concurrent reads.
            # Both pragmas are per-connection, so holding a persistent
            # connection means they survive across every query.
            await self._db.execute("PRAGMA foreign_keys = ON")
            await self._db.execute("PRAGMA journal_mode = WAL")
            await self._db.execute("PRAGMA synchronous = NORMAL")
            self._write_lock = asyncio.Lock()
        db = self._db
        # Use the persistent connection to run the schema migration.
        await db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id          TEXT PRIMARY KEY,
                    user_id             TEXT NOT NULL,
                    start_time          TIMESTAMP NOT NULL,
                    end_time            TIMESTAMP,
                    message_count       INTEGER DEFAULT 0,
                    -- SEC: 'summary' is the ONLY natural-language column in
                    -- the schema.  It is populated exclusively by
                    -- SessionSummarizer from aggregated metadata
                    -- (topics, scalar metrics, emotion labels) and never
                    -- contains raw user or assistant text.
                    summary             TEXT,
                    dominant_emotion    TEXT,
                    topics              TEXT,
                    mean_engagement     REAL,
                    mean_cognitive_load REAL,
                    mean_accessibility  REAL,
                    relationship_strength REAL,
                    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- SEC: exchanges table intentionally has NO column for
                -- text/message/content/body/raw_text/assistant_response.
                -- Only embeddings (BLOB), scalar metrics, route, and
                -- topic keywords are persisted.
                CREATE TABLE IF NOT EXISTS exchanges (
                    exchange_id          TEXT PRIMARY KEY,
                    session_id           TEXT NOT NULL,
                    timestamp            TIMESTAMP NOT NULL,
                    user_state_embedding BLOB,
                    adaptation_vector    TEXT,
                    route_chosen         TEXT,
                    response_latency_ms  INTEGER,
                    engagement_signal    REAL,
                    topics               TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_exchanges_session
                    ON exchanges(session_id);

                -- SEC: user_facts persists declared personal-context
                -- across sessions (name, favourite colour, occupation,
                -- location, hobby, age, pet).  Values are stored
                -- Fernet-encrypted at rest when an encryptor is
                -- configured (matching the embedding-encryption
                -- model).  The user controls it: a future "forget my
                -- facts" tool wipes the row.  Iter 50 (2026-04-26).
                CREATE TABLE IF NOT EXISTS user_facts (
                    user_id     TEXT NOT NULL,
                    slot        TEXT NOT NULL,
                    value_blob  BLOB NOT NULL,
                    updated_at  TIMESTAMP NOT NULL,
                    PRIMARY KEY (user_id, slot)
                );
                CREATE INDEX IF NOT EXISTS idx_user_facts_user
                    ON user_facts(user_id);
            """)
        await db.commit()
        self._initialized = True
        logger.info("DiaryStore initialised (db=%s)", self.db_path)

    async def close(self) -> None:
        """Close the persistent connection.  Idempotent."""
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("DiaryStore.close() raised")
            self._db = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Raise if :meth:`initialize` has not been called."""
        if not self._initialized or self._db is None:
            raise RuntimeError(
                "DiaryStore has not been initialised. "
                "Call `await store.initialize()` first."
            )

    def _db_required(self) -> aiosqlite.Connection:
        """Return the persistent connection or raise if absent."""
        self._ensure_initialized()
        assert self._db is not None  # narrowed by _ensure_initialized
        return self._db

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield the persistent connection; drop-in for the old
        ``async with self._conn() as db:`` block.

        We do NOT close the underlying connection on block exit — the
        store owns the connection lifecycle.  This makes the change an
        indentation-preserving textual swap across every call site.
        """
        db = self._db_required()
        yield db

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str, user_id: str) -> None:
        """Create a new session record with the current UTC timestamp.

        Parameters
        ----------
        session_id:
            Unique session identifier (typically a UUID4).
        user_id:
            Identifier of the user who owns this session.
        """
        self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO sessions (session_id, user_id, start_time)
                VALUES (?, ?, ?)
                """,
                (session_id, user_id, now),
            )
            await db.commit()
        logger.debug("Session created: %s (user=%s)", session_id, user_id)

    async def end_session(
        self,
        session_id: str,
        summary: str,
        dominant_emotion: str,
        topics: list[str],
        mean_engagement: float,
        mean_cognitive_load: float,
        mean_accessibility: float,
        relationship_strength: float,
    ) -> None:
        """Finalise a session with aggregated metrics and a summary.

        Parameters
        ----------
        session_id:
            The session to close.
        summary:
            A privacy-safe natural-language summary of the session
            (generated from metadata, never from raw text).
        dominant_emotion:
            The most frequently observed emotion label during the session.
        topics:
            Aggregated list of topic keywords extracted via TF-IDF.
        mean_engagement:
            Average engagement signal across all exchanges (0-1).
        mean_cognitive_load:
            Average cognitive load estimate (0-1).
        mean_accessibility:
            Average accessibility adaptation level (0-1).
        relationship_strength:
            Estimated user-AI relationship strength (0-1).
        """
        self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()

        # Count exchanges in this session
        async with self._conn() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM exchanges WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            message_count = row[0] if row else 0

            await db.execute(
                """
                UPDATE sessions
                SET end_time             = ?,
                    message_count        = ?,
                    summary              = ?,
                    dominant_emotion     = ?,
                    topics               = ?,
                    mean_engagement      = ?,
                    mean_cognitive_load  = ?,
                    mean_accessibility   = ?,
                    relationship_strength = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    message_count,
                    summary,
                    dominant_emotion,
                    # SEC: deterministic JSON serialisation (sort_keys) so
                    # identical topic sets always hash to the same blob,
                    # preventing leakage via ordering side-channels.
                    json.dumps(topics, sort_keys=True),
                    mean_engagement,
                    mean_cognitive_load,
                    mean_accessibility,
                    relationship_strength,
                    session_id,
                ),
            )
            await db.commit()
        logger.info(
            "Session ended: %s (%d exchanges)", session_id, message_count
        )

    # ------------------------------------------------------------------
    # Exchange logging
    # ------------------------------------------------------------------

    async def log_exchange(
        self,
        session_id: str,
        user_state_embedding: bytes,
        adaptation_vector: dict,
        route_chosen: str,
        response_latency_ms: int,
        engagement_signal: float,
        topics: list[str],
    ) -> str:
        """Record a single user-AI exchange within a session.

        Parameters
        ----------
        session_id:
            Parent session identifier.
        user_state_embedding:
            Binary representation of the user-state embedding vector.
            Typically ``tensor.numpy().tobytes()``.
        adaptation_vector:
            Serialised adaptation vector as a dict (will be JSON-encoded).
        route_chosen:
            Which generation route was selected (e.g. ``"local_slm"``,
            ``"cloud_llm"``).
        response_latency_ms:
            Round-trip response latency in milliseconds.
        engagement_signal:
            Estimated user engagement for this exchange (0-1).
        topics:
            Topic keywords extracted from the message via TF-IDF.

        Returns
        -------
        str
            The generated ``exchange_id`` (UUID4).
        """
        self._ensure_initialized()
        exchange_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO exchanges (
                    exchange_id, session_id, timestamp,
                    user_state_embedding, adaptation_vector, route_chosen,
                    response_latency_ms, engagement_signal, topics
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange_id,
                    session_id,
                    now,
                    user_state_embedding,
                    # SEC: sort_keys ensures deterministic adaptation_vector
                    # serialisation -- required for reproducible diary
                    # records and for stable equality testing.
                    json.dumps(adaptation_vector, sort_keys=True),
                    route_chosen,
                    response_latency_ms,
                    engagement_signal,
                    json.dumps(topics, sort_keys=True),
                ),
            )
            await db.commit()
        logger.debug(
            "Exchange logged: %s (session=%s, route=%s)",
            exchange_id,
            session_id,
            route_chosen,
        )
        return exchange_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> dict | None:
        """Retrieve a single session by its identifier.

        Returns
        -------
        dict or None
            Session record as a dictionary, or ``None`` if not found.
            The ``topics`` field is deserialised from JSON to a list.
        """
        self._ensure_initialized()
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            result = dict(row)
            if result.get("topics"):
                result["topics"] = json.loads(result["topics"])
            return result

    async def get_user_sessions(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """Retrieve the most recent sessions for a given user.

        Parameters
        ----------
        user_id:
            The user whose sessions to retrieve.
        limit:
            Maximum number of sessions to return (newest first).

        Returns
        -------
        list[dict]
            Session records ordered by ``start_time`` descending.
        """
        self._ensure_initialized()
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sessions
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                record = dict(row)
                if record.get("topics"):
                    record["topics"] = json.loads(record["topics"])
                results.append(record)
            return results

    async def get_session_exchanges(self, session_id: str) -> list[dict]:
        """Retrieve all exchanges for a given session, ordered by time.

        Parameters
        ----------
        session_id:
            The session whose exchanges to retrieve.

        Returns
        -------
        list[dict]
            Exchange records with ``adaptation_vector`` and ``topics``
            deserialised from JSON.
        """
        self._ensure_initialized()
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM exchanges
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                record = dict(row)
                if record.get("adaptation_vector"):
                    record["adaptation_vector"] = json.loads(
                        record["adaptation_vector"]
                    )
                if record.get("topics"):
                    record["topics"] = json.loads(record["topics"])
                results.append(record)
            return results

    async def get_user_stats(self, user_id: str) -> dict:
        """Compute aggregate statistics for a user across all sessions.

        Returns
        -------
        dict
            Keys: ``total_sessions``, ``total_messages``,
            ``avg_engagement``, ``avg_cognitive_load``,
            ``avg_accessibility``, ``topic_frequency`` (dict of
            topic -> count, sorted descending).
        """
        self._ensure_initialized()
        async with self._conn() as db:
            # Aggregated session-level stats
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*)               AS total_sessions,
                    COALESCE(SUM(message_count), 0) AS total_messages,
                    AVG(mean_engagement)   AS avg_engagement,
                    AVG(mean_cognitive_load) AS avg_cognitive_load,
                    AVG(mean_accessibility) AS avg_accessibility
                FROM sessions
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            # SEC: AVG() over zero rows returns NULL in SQLite, not 0.
            # Coerce None -> 0.0 so downstream consumers can rely on
            # numeric types (prevents `TypeError: NoneType < float`).
            stats: dict = {
                "total_sessions": (row[0] if row else 0) or 0,
                "total_messages": (row[1] if row else 0) or 0,
                "avg_engagement": (row[2] if row else 0.0) or 0.0,
                "avg_cognitive_load": (row[3] if row else 0.0) or 0.0,
                "avg_accessibility": (row[4] if row else 0.0) or 0.0,
                "topic_frequency": {},
            }

            # Topic frequency aggregation
            cursor = await db.execute(
                "SELECT topics FROM sessions WHERE user_id = ? AND topics IS NOT NULL",
                (user_id,),
            )
            topic_counts: dict[str, int] = {}
            async for topic_row in cursor:
                try:
                    topics = json.loads(topic_row[0])
                except (json.JSONDecodeError, TypeError):
                    continue
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Sort by frequency descending
            stats["topic_frequency"] = dict(
                sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)
            )
            return stats

    # ------------------------------------------------------------------
    # Retention / pruning
    # ------------------------------------------------------------------

    async def prune_old_entries(
        self, user_id: str, max_entries: int = 1000
    ) -> int:
        """Delete the oldest sessions for a user beyond ``max_entries``.

        Enforces the ``diary.max_entries`` retention configuration.  Also
        cascades into ``exchanges`` so that orphaned exchange rows are not
        left behind.

        Parameters
        ----------
        user_id:
            The user whose old sessions to prune.
        max_entries:
            Maximum number of sessions to keep (newest first).  Defaults
            to the spec value of 1000.

        Returns
        -------
        int
            The number of sessions deleted (0 if under the cap).
        """
        # SEC: parameterised query.  user_id and max_entries flow through
        # placeholders -- never string-interpolated.  This method is the
        # ONLY mutation path that deletes diary data and is gated on a
        # numeric retention threshold; it cannot be exploited to delete
        # arbitrary rows by ID.
        self._ensure_initialized()
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")

        async with self._conn() as db:
            # SEC: enable FK enforcement for cascade-correctness checks.
            await db.execute("PRAGMA foreign_keys = ON")

            # Identify session_ids to delete: oldest beyond the cap.
            cursor = await db.execute(
                """
                SELECT session_id FROM sessions
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT -1 OFFSET ?
                """,
                (user_id, max_entries),
            )
            old_session_ids = [row[0] for row in await cursor.fetchall()]
            if not old_session_ids:
                return 0

            # SEC: Delete dependent exchanges first, then sessions.  The
            # f-string below ONLY interpolates a string of '?' characters
            # (placeholders), NEVER user data.  All actual values flow
            # through the second argument to db.execute() as bound
            # parameters -- this is the standard parameterised IN-clause
            # idiom and is SQL-injection safe.
            placeholders = ",".join("?" for _ in old_session_ids)
            await db.execute(
                f"DELETE FROM exchanges WHERE session_id IN ({placeholders})",
                old_session_ids,
            )
            await db.execute(
                f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                old_session_ids,
            )
            await db.commit()

        logger.info(
            "Pruned %d old diary sessions for user=%s (cap=%d)",
            len(old_session_ids),
            user_id,
            max_entries,
        )
        return len(old_session_ids)

    async def get_recent_diary_entries(
        self, user_id: str, n: int = 5
    ) -> list[dict]:
        """Get recent session summaries formatted for the diary display.

        Returns lightweight records suitable for rendering in a diary or
        timeline UI: session id, start time, message count, summary, topics,
        and dominant emotion.

        Parameters
        ----------
        user_id:
            User whose diary entries to retrieve.
        n:
            Maximum number of entries (newest first).

        Returns
        -------
        list[dict]
            Diary entries ordered by ``start_time`` descending.
        """
        self._ensure_initialized()
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    session_id,
                    start_time,
                    end_time,
                    message_count,
                    summary,
                    dominant_emotion,
                    topics,
                    mean_engagement,
                    relationship_strength
                FROM sessions
                WHERE user_id = ? AND end_time IS NOT NULL
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (user_id, n),
            )
            rows = await cursor.fetchall()
            entries = []
            for row in rows:
                entry = dict(row)
                if entry.get("topics"):
                    entry["topics"] = json.loads(entry["topics"])
                entries.append(entry)
            return entries

    # ------------------------------------------------------------------
    # User-facts (cross-session personal-context memory) — Iter 50
    # ------------------------------------------------------------------
    # SEC: stored values are Fernet-encrypted when an encryptor is
    # configured, matching the embedding-encryption model.  When no
    # encryptor is set, values are stored plaintext but still in a
    # versioned envelope (first byte = 0x00 plaintext, 0x01 Fernet)
    # so the read path is uniform.

    @staticmethod
    def _encode_fact_value(
        value: str, encryptor: "ModelEncryptor | None",
    ) -> bytes:
        """Encode a fact value string as a versioned envelope BLOB."""
        raw = value.encode("utf-8")
        if encryptor is None:
            return bytes([_ENC_VERSION_PLAINTEXT]) + raw
        return bytes([_ENC_VERSION_FERNET_V1]) + encryptor.encrypt(raw)

    @staticmethod
    def _decode_fact_value(
        blob: bytes, encryptor: "ModelEncryptor | None",
    ) -> str | None:
        """Decode a fact value BLOB.  Returns None on any decode error."""
        if not blob:
            return None
        version = blob[0]
        payload = blob[1:]
        try:
            if version == _ENC_VERSION_PLAINTEXT:
                return payload.decode("utf-8")
            if version == _ENC_VERSION_FERNET_V1:
                if encryptor is None:
                    return None
                return encryptor.decrypt(payload).decode("utf-8")
        except Exception:  # pragma: no cover — defensive decode
            return None
        return None

    async def set_user_fact(
        self,
        user_id: str,
        slot: str,
        value: str,
    ) -> None:
        """Persist a single (user_id, slot) → value row.  Upsert."""
        if not user_id or not slot or not value:
            return
        await self.initialize()
        encoded = self._encode_fact_value(value, self._encryptor)
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO user_facts (user_id, slot, value_blob, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, slot) DO UPDATE
                SET value_blob = excluded.value_blob,
                    updated_at = excluded.updated_at
                """,
                (user_id, slot, encoded),
            )
            await db.commit()

    async def get_user_facts(self, user_id: str) -> dict[str, str]:
        """Return all stored facts for a user as ``{slot: value}``."""
        if not user_id:
            return {}
        await self.initialize()
        async with self._conn() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT slot, value_blob FROM user_facts WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
        out: dict[str, str] = {}
        for row in rows:
            decoded = self._decode_fact_value(
                bytes(row["value_blob"]), self._encryptor,
            )
            if decoded:
                out[row["slot"]] = decoded
        return out

    async def forget_user_facts(
        self, user_id: str, slot: str | None = None,
    ) -> int:
        """Delete user facts.  Single slot when provided, else all.

        Returns the number of rows deleted.  Used by the "forget my
        facts" tool — the user controls retention.
        """
        if not user_id:
            return 0
        await self.initialize()
        async with self._conn() as db:
            if slot:
                cursor = await db.execute(
                    "DELETE FROM user_facts WHERE user_id = ? AND slot = ?",
                    (user_id, slot),
                )
            else:
                cursor = await db.execute(
                    "DELETE FROM user_facts WHERE user_id = ?",
                    (user_id,),
                )
            await db.commit()
            return cursor.rowcount or 0
