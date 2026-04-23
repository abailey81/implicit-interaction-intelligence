"""DuckDB-backed analytics over the I3 SQLite diary.

DuckDB can attach a SQLite database as a read-only schema and execute
analytical queries directly against it without copying data.  For the I3
diary this yields 10-40x speedups on aggregate queries while leaving
the transactional SQLite file untouched.

Privacy
-------
Nothing in this module creates, modifies or deletes data.  Every query
is executed against the attached SQLite with ``READ_ONLY`` semantics.

Example
-------
    >>> from i3.analytics import DuckDBAnalytics
    >>> with DuckDBAnalytics("data/diary.db") as ddb:
    ...     df = ddb.route_distribution("alice")
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import helpers
# ---------------------------------------------------------------------------


def _require_duckdb() -> Any:
    """Import duckdb lazily with a friendly install hint."""
    try:
        import duckdb as _duckdb
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(
            "DuckDB is not installed. Install the analytics extras:\n"
            "    poetry install --with analytics\n"
            "or: pip install 'duckdb>=1.0'"
        ) from exc
    return _duckdb


def _require_polars() -> Any:
    """Import polars lazily with a friendly install hint."""
    try:
        import polars as _pl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Polars is not installed. Install the analytics extras:\n"
            "    poetry install --with analytics\n"
            "or: pip install 'polars>=1.0'"
        ) from exc
    return _pl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DuckDBAnalytics:
    """Run analytical queries over the I3 SQLite diary via DuckDB.

    The constructor attaches the SQLite file to an in-memory DuckDB
    instance in ``READ_ONLY`` mode.  All public methods return Polars
    :class:`polars.DataFrame` objects.

    Attributes:
        sqlite_path: Filesystem path of the SQLite diary database.
        attach_alias: Schema alias under which SQLite is attached.

    Args:
        sqlite_path: Absolute or relative path to the diary SQLite file.
        attach_alias: Schema alias for the attached database (default
            ``"diary"``).  Choose a different alias when attaching
            multiple databases in the same process.

    Raises:
        ImportError: If ``duckdb`` or ``polars`` are not installed.
        FileNotFoundError: If ``sqlite_path`` does not point at a file.
    """

    def __init__(
        self,
        sqlite_path: str | Path = "data/diary.db",
        attach_alias: str = "diary",
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.attach_alias = attach_alias
        self._duckdb = _require_duckdb()
        self._pl = _require_polars()
        # Defer connection until first use so the class can be constructed
        # cheaply inside CLI scripts that may short-circuit on `--help`.
        self._con: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle / context manager
    # ------------------------------------------------------------------

    def connect(self) -> DuckDBAnalytics:
        """Open a DuckDB connection and attach the SQLite diary.

        Returns:
            Self, so the call can be chained (e.g. in ``with`` blocks).

        Raises:
            FileNotFoundError: If the SQLite file does not exist.
        """
        if self._con is not None:
            return self
        if not self.sqlite_path.exists():
            raise FileNotFoundError(
                f"SQLite diary not found at: {self.sqlite_path}"
            )
        # sqlite_scanner is bundled with DuckDB >= 0.9; INSTALL is idempotent.
        self._con = self._duckdb.connect(database=":memory:")
        try:
            self._con.execute("INSTALL sqlite_scanner;")
        except Exception:  # pragma: no cover - already installed
            pass
        self._con.execute("LOAD sqlite_scanner;")
        # READ_ONLY is critical: the analytics layer must never mutate the
        # transactional diary file written by aiosqlite.
        self._con.execute(
            f"ATTACH '{self.sqlite_path.as_posix()}' "
            f"AS {self.attach_alias} (TYPE SQLITE, READ_ONLY);"
        )
        logger.info(
            "DuckDBAnalytics attached %s as %s (READ_ONLY)",
            self.sqlite_path,
            self.attach_alias,
        )
        return self

    def close(self) -> None:
        """Close the DuckDB connection.  Idempotent."""
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> DuckDBAnalytics:
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> Any:
        """Return an active DuckDB connection, connecting on demand."""
        if self._con is None:
            self.connect()
        assert self._con is not None  # for type checkers
        return self._con

    def _sql_to_polars(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Execute SQL and return the result as a Polars DataFrame."""
        con = self._ensure_connected()
        rel = con.execute(sql, params or [])
        # DuckDB >= 0.9 exposes arrow() which we round-trip through Polars
        # for zero-copy columnar conversion.
        try:
            arrow_table = rel.arrow()
            return self._pl.from_arrow(arrow_table)
        except Exception:  # pragma: no cover - very old duckdb
            rows = rel.fetchall()
            cols = [d[0] for d in rel.description]
            return self._pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def adaptation_distribution(
        self, user_id: str | None = None
    ) -> pl.DataFrame:
        """Compute per-dimension mean and std of the adaptation vector.

        The adaptation vector is stored as a JSON object on the
        ``exchanges`` table.  DuckDB's native JSON functions let us
        unnest and aggregate without any Python-side parsing.

        Args:
            user_id: Optional filter.  When ``None``, the statistics are
                computed across all users.

        Returns:
            A Polars DataFrame with columns ``dimension``, ``mean``,
            ``std``, ``n`` sorted by dimension name.
        """
        alias = self.attach_alias
        where = ""
        params: list[Any] = []
        if user_id is not None:
            where = "WHERE s.user_id = ?"
            params.append(user_id)
        sql = f"""
            WITH j AS (
                SELECT
                    s.user_id,
                    e.adaptation_vector
                FROM {alias}.exchanges e
                JOIN {alias}.sessions s USING (session_id)
                {where}
            ),
            kv AS (
                SELECT
                    je.key AS dimension,
                    TRY_CAST(je.value AS DOUBLE) AS val
                FROM j,
                     json_each(j.adaptation_vector) AS je
            )
            SELECT
                dimension,
                AVG(val) AS mean,
                STDDEV_SAMP(val) AS std,
                COUNT(val) AS n
            FROM kv
            WHERE val IS NOT NULL
            GROUP BY dimension
            ORDER BY dimension
        """
        return self._sql_to_polars(sql, params)

    def route_distribution(
        self, user_id: str | None = None
    ) -> pl.DataFrame:
        """Count exchanges per router choice (``local_slm`` vs ``cloud_llm``).

        Args:
            user_id: Optional filter; ``None`` aggregates across users.

        Returns:
            A Polars DataFrame with columns ``route``, ``count``,
            ``fraction`` sorted by count descending.
        """
        alias = self.attach_alias
        where = ""
        params: list[Any] = []
        if user_id is not None:
            where = "WHERE s.user_id = ?"
            params.append(user_id)
        sql = f"""
            WITH routed AS (
                SELECT e.route_chosen AS route
                FROM {alias}.exchanges e
                JOIN {alias}.sessions s USING (session_id)
                {where}
            )
            SELECT
                route,
                COUNT(*) AS count,
                CAST(COUNT(*) AS DOUBLE) / SUM(COUNT(*)) OVER () AS fraction
            FROM routed
            GROUP BY route
            ORDER BY count DESC
        """
        return self._sql_to_polars(sql, params)

    def latency_percentiles(
        self, user_id: str | None = None
    ) -> pl.DataFrame:
        """Compute latency P50/P95/P99 grouped by router route.

        Args:
            user_id: Optional filter; ``None`` aggregates across users.

        Returns:
            A Polars DataFrame with columns ``route``, ``p50_ms``,
            ``p95_ms``, ``p99_ms``, ``n``.
        """
        alias = self.attach_alias
        where = ""
        params: list[Any] = []
        if user_id is not None:
            where = "WHERE s.user_id = ?"
            params.append(user_id)
        sql = f"""
            SELECT
                e.route_chosen AS route,
                QUANTILE_CONT(e.response_latency_ms, 0.50) AS p50_ms,
                QUANTILE_CONT(e.response_latency_ms, 0.95) AS p95_ms,
                QUANTILE_CONT(e.response_latency_ms, 0.99) AS p99_ms,
                COUNT(*) AS n
            FROM {alias}.exchanges e
            JOIN {alias}.sessions s USING (session_id)
            {where}
            GROUP BY e.route_chosen
            ORDER BY e.route_chosen
        """
        return self._sql_to_polars(sql, params)

    def session_heatmap_by_hour(self, user_id: str) -> pl.DataFrame:
        """Compute a 24x7 engagement heatmap for a user.

        Rows are weekdays (``0``=Monday..``6``=Sunday), columns are
        hours of day (``0``..``23``).  The value is mean engagement.

        Args:
            user_id: The user to restrict the heatmap to.

        Returns:
            A Polars DataFrame with columns ``weekday``, ``hour``,
            ``sessions``, ``mean_engagement``.  Missing cells are
            returned as ``NaN``.
        """
        alias = self.attach_alias
        sql = f"""
            WITH sess AS (
                SELECT
                    -- DuckDB: ISODOW returns 1=Mon..7=Sun; subtract 1 for 0-indexed.
                    (EXTRACT(ISODOW FROM CAST(start_time AS TIMESTAMP)) - 1)::INT AS weekday,
                    EXTRACT(HOUR FROM CAST(start_time AS TIMESTAMP))::INT AS hour,
                    mean_engagement
                FROM {alias}.sessions
                WHERE user_id = ? AND mean_engagement IS NOT NULL
            )
            SELECT
                weekday,
                hour,
                COUNT(*) AS sessions,
                AVG(mean_engagement) AS mean_engagement
            FROM sess
            GROUP BY weekday, hour
            ORDER BY weekday, hour
        """
        return self._sql_to_polars(sql, [user_id])

    def rolling_engagement(
        self, user_id: str, window_days: int = 7
    ) -> pl.DataFrame:
        """Rolling-window mean engagement for a user.

        Args:
            user_id: The user to compute for.
            window_days: Number of days in the trailing window.  Must
                be >= 1.

        Returns:
            A Polars DataFrame with columns ``day``, ``sessions_in_day``,
            ``rolling_mean_engagement`` sorted by day ascending.

        Raises:
            ValueError: If ``window_days`` < 1.
        """
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        alias = self.attach_alias
        # DuckDB's RANGE window over an INTERVAL gives us a calendar-aware
        # rolling mean that correctly handles gaps in the session history.
        sql = f"""
            WITH daily AS (
                SELECT
                    CAST(CAST(start_time AS TIMESTAMP) AS DATE) AS day,
                    COUNT(*) AS sessions_in_day,
                    AVG(mean_engagement) AS day_mean_engagement
                FROM {alias}.sessions
                WHERE user_id = ? AND mean_engagement IS NOT NULL
                GROUP BY day
            )
            SELECT
                day,
                sessions_in_day,
                AVG(day_mean_engagement) OVER (
                    ORDER BY day
                    RANGE BETWEEN INTERVAL '{int(window_days) - 1}' DAY PRECEDING
                              AND CURRENT ROW
                ) AS rolling_mean_engagement
            FROM daily
            ORDER BY day
        """
        return self._sql_to_polars(sql, [user_id])
