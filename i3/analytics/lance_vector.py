"""LanceDB vector store for user/session embedding similarity search.

LanceDB is an embedded vector database built on top of the Apache Arrow
columnar format.  It supports both exact and IVF-PQ approximate-nearest-
neighbour search and stores indices on disk in the open ``.lance``
format.

Schema
------
Each row represents a single session-level (or exchange-level) embedding:

================ =============================================
``user_id``      ``str``        owning user identifier
``session_id``   ``str``        session identifier
``ts``           ``datetime``   UTC timestamp of the embedding
``embedding``    ``float32[64]`` 64-dim user/session embedding
``adaptation``   ``str``        JSON-serialised adaptation dict
================ =============================================

Privacy
-------
Embeddings are already lossy projections of the original signals — they
cannot be inverted to raw text.  In deployments, the on-disk LanceDB
directory must reside under the same encrypted-at-rest volume as the
SQLite diary (see :mod:`i3.privacy.encryption`) so that at-rest
confidentiality is preserved.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import helpers
# ---------------------------------------------------------------------------


def _require_lancedb() -> Any:
    try:
        import lancedb as _lancedb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LanceDB is not installed. Install the analytics extras:\n"
            "    poetry install --with analytics\n"
            "or: pip install 'lancedb>=0.11'"
        ) from exc
    return _lancedb


def _require_pyarrow() -> Any:
    try:
        import pyarrow as _pa
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyArrow is not installed. Install the analytics extras:\n"
            "    poetry install --with analytics\n"
            "or: pip install 'pyarrow>=15.0'"
        ) from exc
    return _pa


def _require_numpy() -> Any:
    try:
        import numpy as _np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "NumPy is required for embedding operations."
        ) from exc
    return _np


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarUserResult:
    """A single hit from :meth:`LanceUserEmbeddingStore.search_similar`.

    Attributes:
        user_id: Identifier of the matched user.
        session_id: Identifier of the matched session/embedding row.
        distance: Cosine (or L2) distance from the query vector.
        score: Similarity score (1 - distance for cosine).
        adaptation: Deserialised adaptation dict for the match.
        ts: Timestamp of the matched embedding.
    """

    user_id: str
    session_id: str
    distance: float
    score: float
    adaptation: dict[str, Any]
    ts: datetime


# ---------------------------------------------------------------------------
# Main store
# ---------------------------------------------------------------------------


class LanceUserEmbeddingStore:
    """On-disk LanceDB store for user/session embeddings.

    Args:
        uri: Filesystem path of the LanceDB directory.  Created if it
            does not exist.
        table_name: Name of the Lance table (default ``"user_embeddings"``).
        embedding_dim: Dimension of the stored embeddings.  Must match
            the encoder output — the default of ``64`` matches the I3
            encoder.
        metric: Distance metric for search; one of ``"cosine"`` or
            ``"l2"``.

    Raises:
        ImportError: If ``lancedb`` or ``pyarrow`` are not installed.
    """

    def __init__(
        self,
        uri: str | Path,
        table_name: str = "user_embeddings",
        embedding_dim: int = 64,
        metric: str = "cosine",
    ) -> None:
        self.uri = Path(uri)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric
        self._lancedb = _require_lancedb()
        self._pa = _require_pyarrow()
        self._np = _require_numpy()
        self.uri.mkdir(parents=True, exist_ok=True)
        self._db = self._lancedb.connect(str(self.uri))
        self._table: Any | None = None

    # ------------------------------------------------------------------
    # Schema / table management
    # ------------------------------------------------------------------

    def _arrow_schema(self) -> Any:
        """Return the Arrow schema for the user_embeddings table."""
        pa = self._pa
        return pa.schema(
            [
                pa.field("user_id", pa.string(), nullable=False),
                pa.field("session_id", pa.string(), nullable=False),
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field(
                    "embedding",
                    pa.list_(pa.float32(), self.embedding_dim),
                    nullable=False,
                ),
                pa.field("adaptation", pa.string(), nullable=True),
            ]
        )

    def _get_or_create_table(self) -> Any:
        """Return the Lance table, creating it with the correct schema if needed."""
        if self._table is not None:
            return self._table
        existing = set(self._db.table_names())
        if self.table_name in existing:
            self._table = self._db.open_table(self.table_name)
        else:
            empty = self._pa.Table.from_pylist([], schema=self._arrow_schema())
            self._table = self._db.create_table(
                self.table_name, data=empty, mode="create"
            )
        return self._table

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        user_id: str,
        session_id: str,
        ts: datetime,
        embedding: np.ndarray | Sequence[float],
        adaptation: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a single embedding row.

        Rows are identified by ``(user_id, session_id)``.  If a row
        already exists it is deleted before the new one is appended,
        producing upsert semantics.

        Args:
            user_id: Owning user identifier.
            session_id: Session identifier — unique per user.
            ts: UTC timestamp of the embedding.  Naive timestamps are
                coerced to UTC.
            embedding: Embedding vector of dimension
                :attr:`embedding_dim`.
            adaptation: Optional adaptation dict; serialised to JSON.

        Raises:
            ValueError: If the embedding shape does not match
                :attr:`embedding_dim`.
        """
        np = self._np
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding must be 1-D with shape ({self.embedding_dim},), "
                f"got shape {vec.shape}"
            )
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        table = self._get_or_create_table()
        # Upsert via delete + add; LanceDB supports a merge_insert API on
        # newer versions but delete/add is always available and atomic
        # enough for a single-writer analytics store.
        try:
            table.delete(
                f"user_id = '{self._escape(user_id)}' "
                f"AND session_id = '{self._escape(session_id)}'"
            )
        except Exception as exc:  # pragma: no cover - table empty
            logger.debug("Lance delete no-op: %s", exc)
        row = {
            "user_id": user_id,
            "session_id": session_id,
            "ts": ts,
            "embedding": vec.tolist(),
            "adaptation": json.dumps(adaptation or {}, sort_keys=True),
        }
        table.add([row])

    @staticmethod
    def _escape(value: str) -> str:
        """Escape single quotes for Lance's filter DSL."""
        return value.replace("'", "''")

    # ------------------------------------------------------------------
    # Reads / search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        embedding: np.ndarray | Sequence[float],
        k: int = 10,
        exclude_user_id: str | None = None,
    ) -> list[SimilarUserResult]:
        """Return the ``k`` nearest embeddings to the query vector.

        Args:
            embedding: Query embedding of dimension :attr:`embedding_dim`.
            k: Number of hits to return.
            exclude_user_id: If provided, filter out all rows owned by
                this user (useful when searching for users *other than*
                the query user).

        Returns:
            A list of :class:`SimilarUserResult` ordered by ascending
            distance.
        """
        np = self._np
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self.embedding_dim,):
            raise ValueError(
                f"query embedding must be 1-D with shape ({self.embedding_dim},)"
            )
        table = self._get_or_create_table()
        q = table.search(vec.tolist()).metric(self.metric).limit(k)
        if exclude_user_id is not None:
            q = q.where(f"user_id != '{self._escape(exclude_user_id)}'")
        results = q.to_list()
        hits: list[SimilarUserResult] = []
        for r in results:
            distance = float(r.get("_distance", r.get("distance", float("nan"))))
            score = 1.0 - distance if self.metric == "cosine" else -distance
            adaptation_raw = r.get("adaptation") or "{}"
            try:
                adaptation = json.loads(adaptation_raw)
            except (TypeError, json.JSONDecodeError):
                adaptation = {}
            ts = r.get("ts")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts is None:
                ts = datetime.now(timezone.utc)
            hits.append(
                SimilarUserResult(
                    user_id=r["user_id"],
                    session_id=r["session_id"],
                    distance=distance,
                    score=score,
                    adaptation=adaptation,
                    ts=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                )
            )
        return hits

    def search_by_adaptation_cluster(
        self, adaptation_archetype: dict[str, float], k: int = 20
    ) -> list[SimilarUserResult]:
        """Fetch rows whose adaptation closely matches an archetype.

        Strategy: compute per-user mean adaptation over the table and
        return the users whose Euclidean distance to
        ``adaptation_archetype`` is smallest.  This is a deliberately
        simple baseline; production deployments may upgrade to a
        dedicated adaptation-vector index.

        Args:
            adaptation_archetype: Target adaptation dict (dim -> value).
            k: Number of hits to return.

        Returns:
            A list of :class:`SimilarUserResult` sorted by ascending
            distance.  ``ts`` is set to the most recent row for each
            user.
        """
        np = self._np
        table = self._get_or_create_table()
        rows = table.to_pandas()
        if rows.empty:
            return []
        archetype_keys = sorted(adaptation_archetype.keys())
        archetype_vec = np.asarray(
            [adaptation_archetype[k] for k in archetype_keys], dtype=np.float64
        )

        def _flatten(js: str) -> np.ndarray:
            try:
                d = json.loads(js)
            except (TypeError, json.JSONDecodeError):
                d = {}
            return np.asarray(
                [float(d.get(k, 0.0)) for k in archetype_keys], dtype=np.float64
            )

        rows["adaptation_vec"] = rows["adaptation"].map(_flatten)
        rows["dist"] = rows["adaptation_vec"].map(
            lambda v: float(np.linalg.norm(v - archetype_vec))
        )
        top = rows.nsmallest(k, "dist")
        hits: list[SimilarUserResult] = []
        for _, r in top.iterrows():
            ts_val = r["ts"]
            if hasattr(ts_val, "to_pydatetime"):
                ts = ts_val.to_pydatetime()
            else:
                ts = ts_val
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            try:
                adaptation = json.loads(r["adaptation"] or "{}")
            except (TypeError, json.JSONDecodeError):
                adaptation = {}
            hits.append(
                SimilarUserResult(
                    user_id=r["user_id"],
                    session_id=r["session_id"],
                    distance=float(r["dist"]),
                    score=-float(r["dist"]),
                    adaptation=adaptation,
                    ts=ts,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(
        self, num_partitions: int = 256, num_sub_vectors: int = 16
    ) -> None:
        """Build an IVF-PQ index for approximate-nearest-neighbour search.

        Args:
            num_partitions: Number of Voronoi cells (IVF).  Rule of
                thumb: ``sqrt(num_rows)``.
            num_sub_vectors: Number of PQ sub-vectors.  Must divide
                ``embedding_dim`` evenly.
        """
        if self.embedding_dim % num_sub_vectors != 0:
            raise ValueError(
                f"num_sub_vectors ({num_sub_vectors}) must divide "
                f"embedding_dim ({self.embedding_dim})"
            )
        table = self._get_or_create_table()
        # LanceDB API: create_index(metric, num_partitions, num_sub_vectors, ...)
        table.create_index(
            metric=self.metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="embedding",
            replace=True,
        )
        logger.info(
            "LanceDB IVF-PQ index built: partitions=%d sub_vectors=%d metric=%s",
            num_partitions,
            num_sub_vectors,
            self.metric,
        )

    def compact(self) -> None:
        """Compact Lance fragment files and clean up old versions.

        Should be invoked periodically (e.g. nightly) to amortise the
        cost of small upserts.  Safe to call when the table is empty.
        """
        table = self._get_or_create_table()
        # LanceDB's compact_files collapses small write fragments; the
        # API surface differs slightly between versions, so we try each.
        try:
            table.compact_files()
        except AttributeError:  # pragma: no cover - very old lancedb
            try:
                table.optimize()
            except AttributeError:
                logger.warning(
                    "LanceDB installation lacks compact_files/optimize; "
                    "skipping compaction."
                )
                return
        try:
            table.cleanup_old_versions()
        except AttributeError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of rows currently in the table."""
        table = self._get_or_create_table()
        # count_rows() on newer LanceDB, len() on older.
        try:
            return int(table.count_rows())
        except AttributeError:  # pragma: no cover
            return len(table.to_pandas())
