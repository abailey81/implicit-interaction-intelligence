"""Apache Arrow + Parquet interop helpers.

Arrow is the columnar in-memory format that underpins DuckDB, Polars,
LanceDB and pandas.  A single Arrow buffer can be passed between those
libraries **without copying**, which is essential for the analytics
throughput targets of I3.

This module exposes a tiny surface:

* :func:`embedding_batch_to_arrow` — pack a list of NumPy embeddings
  into a ``FixedSizeList`` Arrow column.
* :func:`arrow_table_from_diary` — pull the diary ``sessions`` and/or
  ``exchanges`` tables into Arrow via DuckDB's SQLite attach.
* :func:`write_parquet` / :func:`read_parquet` — thin Parquet wrappers
  with sensible compression defaults for analytics snapshots.

Privacy
-------
No helper in this module reads anything outside the diary schema, so
the "no raw text" invariant holds automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pyarrow as pa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import helpers
# ---------------------------------------------------------------------------


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
        raise ImportError("NumPy is required for embedding interop.") from exc
    return _np


def _require_duckdb() -> Any:
    try:
        import duckdb as _duckdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DuckDB is not installed. Install the analytics extras:\n"
            "    pip install 'duckdb>=1.0'"
        ) from exc
    return _duckdb


# ---------------------------------------------------------------------------
# Embedding batch packing
# ---------------------------------------------------------------------------


def embedding_batch_to_arrow(
    embeddings: Sequence[np.ndarray],
    embedding_dim: int | None = None,
    ids: Sequence[str] | None = None,
) -> pa.Table:
    """Pack a list of 1-D embeddings into an Arrow ``FixedSizeList`` table.

    All embeddings must share the same dimensionality.  The result is
    a single Arrow :class:`pyarrow.Table` with columns ``id``
    (optional) and ``embedding``.

    Args:
        embeddings: Iterable of NumPy 1-D arrays.
        embedding_dim: If provided, validated against every embedding.
            If ``None``, inferred from the first embedding.
        ids: Optional per-embedding identifier strings; must match the
            length of ``embeddings``.

    Returns:
        An Arrow table with a ``FixedSizeList<float32, D>`` column.
    """
    pa = _require_pyarrow()
    np = _require_numpy()
    if not embeddings:
        raise ValueError("embeddings must be non-empty")
    if embedding_dim is None:
        embedding_dim = int(embeddings[0].shape[0])
    flat: list[float] = []
    for i, e in enumerate(embeddings):
        arr = np.asarray(e, dtype=np.float32).ravel()
        if arr.shape[0] != embedding_dim:
            raise ValueError(
                f"embeddings[{i}] has dim {arr.shape[0]}, expected {embedding_dim}"
            )
        flat.extend(arr.tolist())
    values = pa.array(flat, type=pa.float32())
    embedding_col = pa.FixedSizeListArray.from_arrays(values, embedding_dim)
    columns: list[Any] = [embedding_col]
    names: list[str] = ["embedding"]
    if ids is not None:
        if len(ids) != len(embeddings):
            raise ValueError("ids must match length of embeddings")
        columns.insert(0, pa.array(list(ids), type=pa.string()))
        names.insert(0, "id")
    return pa.Table.from_arrays(columns, names=names)


# ---------------------------------------------------------------------------
# Diary -> Arrow
# ---------------------------------------------------------------------------


def arrow_table_from_diary(
    path: str | Path = "data/diary.db",
    *,
    table: str = "sessions",
    filters: dict[str, Any] | None = None,
) -> pa.Table:
    """Read the diary (or exchanges) table into an Arrow table via DuckDB.

    Args:
        path: Path to the SQLite diary.
        table: Which diary table to read (``"sessions"`` or
            ``"exchanges"``).
        filters: Optional equality filter dict, e.g.
            ``{"user_id": "alice"}``.  Values are bound as parameters.

    Returns:
        An Arrow :class:`pyarrow.Table`.

    Raises:
        ValueError: If ``table`` is not one of the known diary tables.
    """
    if table not in {"sessions", "exchanges"}:
        raise ValueError("table must be 'sessions' or 'exchanges'")
    duckdb = _require_duckdb()
    pa = _require_pyarrow()
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL sqlite_scanner;")
    except Exception:  # pragma: no cover
        pass
    con.execute("LOAD sqlite_scanner;")
    con.execute(
        f"ATTACH '{Path(path).as_posix()}' AS diary (TYPE SQLITE, READ_ONLY);"
    )
    where = ""
    params: list[Any] = []
    if filters:
        clauses: list[str] = []
        for k, v in filters.items():
            # Column allow-list to stop SQL injection via key.
            if not k.isidentifier():
                raise ValueError(f"invalid filter column: {k!r}")
            clauses.append(f"{k} = ?")
            params.append(v)
        where = "WHERE " + " AND ".join(clauses)
    sql = f"SELECT * FROM diary.{table} {where}"
    try:
        arrow_table: Any = con.execute(sql, params).arrow()
    finally:
        con.close()
    assert isinstance(arrow_table, pa.Table)
    return arrow_table


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def write_parquet(
    table: pa.Table,
    path: str | Path,
    *,
    compression: str = "zstd",
    compression_level: int = 3,
) -> Path:
    """Write an Arrow table to Parquet with analytics-friendly defaults.

    Args:
        table: Arrow table to persist.
        path: Output Parquet file path.  Parent directories are created.
        compression: Parquet compression codec.  ``"zstd"`` is the
            default — roughly 2x smaller than ``"snappy"`` at similar
            read speed.
        compression_level: Codec-specific level.

    Returns:
        The resolved output path.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        str(out),
        compression=compression,
        compression_level=compression_level,
    )
    logger.info("Parquet written: %s (%d rows)", out, table.num_rows)
    _ = pa  # keep import alive
    return out


def read_parquet(path: str | Path) -> pa.Table:
    """Read a Parquet file into an Arrow table.

    Args:
        path: Parquet file path.

    Returns:
        The Arrow :class:`pyarrow.Table`.
    """
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(str(path))
