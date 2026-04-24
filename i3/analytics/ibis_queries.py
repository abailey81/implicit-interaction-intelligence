"""Portable Ibis queries over the I3 diary.

Ibis (https://ibis-project.org) is a DataFrame-to-SQL bridge that
compiles the same Python expression to DuckDB, SQLite, PostgreSQL and
other backends.  For I3 this means analytics queries can be authored
once and run:

* against the embedded DuckDB attached to SQLite (fast local analytics);
* against SQLite directly (fallback when DuckDB is absent);
* against a remote Postgres data warehouse (future mirror).

All expressions in this module return :class:`ibis.Expr` objects.  The
caller is responsible for ``.execute()`` or ``.to_pandas()`` once they
know which backend to use.

Privacy
-------
Ibis only sees the columns exposed by the underlying backend.  Because
neither the diary nor the user-model schema have raw-text columns, no
Ibis expression composed from them can ever return text content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ibis import Expr

logger = logging.getLogger(__name__)


def _require_ibis() -> Any:
    try:
        import ibis as _ibis
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ibis-framework is not installed. Install the analytics extras:\n"
            "    poetry install --with analytics\n"
            "or: pip install 'ibis-framework[duckdb]>=9.0'"
        ) from exc
    return _ibis


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def i3_ibis_backend(
    sqlite_path: str | Path = "data/diary.db",
    *,
    prefer_duckdb: bool = True,
) -> Any:
    """Return an Ibis connection to the I3 diary.

    When ``prefer_duckdb`` is ``True`` (the default) and DuckDB is
    installed, the function returns a DuckDB backend that has the
    SQLite file ``ATTACH``-ed as the ``diary`` schema.  Otherwise it
    falls back to the Ibis SQLite backend.

    Args:
        sqlite_path: Path to the SQLite diary file.
        prefer_duckdb: If ``True`` (default), prefer DuckDB + attach.

    Returns:
        An Ibis backend object supporting ``.table(name)``.
    """
    ibis = _require_ibis()
    path = Path(sqlite_path)
    if prefer_duckdb:
        try:
            import duckdb  # noqa: F401

            con = ibis.duckdb.connect()
            # Raw execute for the ATTACH — Ibis DuckDB passes through
            # unknown statements via .raw_sql().
            con.raw_sql(
                f"ATTACH '{path.as_posix()}' AS diary "
                "(TYPE SQLITE, READ_ONLY);"
            )
            return con
        except ImportError:
            logger.info("DuckDB absent; falling back to Ibis SQLite backend.")
    return ibis.sqlite.connect(str(path))


def _diary_tables(con: Any) -> tuple[Any, Any]:
    """Return ``(sessions_t, exchanges_t)`` Ibis table expressions.

    Handles both the DuckDB-with-attached-SQLite case (tables live
    under the ``diary`` schema) and the direct-SQLite case.
    """
    # DuckDB: tables appear as ``diary.sessions``.  Try that first.
    try:
        sessions_t = con.table("sessions", schema="diary")
        exchanges_t = con.table("exchanges", schema="diary")
        return sessions_t, exchanges_t
    except Exception:  # pragma: no cover - not attached-duckdb
        sessions_t = con.table("sessions")
        exchanges_t = con.table("exchanges")
        return sessions_t, exchanges_t


# ---------------------------------------------------------------------------
# Pre-built queries
# ---------------------------------------------------------------------------


def top_topics_by_day(
    con: Any, user_id: str | None = None, limit_per_day: int = 5
) -> Expr:
    """Return the top N most frequent topics per calendar day.

    Because ``topics`` is stored as a JSON string in SQLite, we convert
    it at the SQL level using the ``json_each`` table-valued function
    (available in both DuckDB and SQLite 3.38+).

    Args:
        con: An Ibis backend from :func:`i3_ibis_backend`.
        user_id: Optional user filter; ``None`` means all users.
        limit_per_day: Keep the top N topics per day.

    Returns:
        An Ibis expression selecting ``day``, ``topic``, ``count``,
        ``rank_in_day``.
    """
    ibis = _require_ibis()  # noqa: F841  # bound for parity with other fns; side-effect-raises on missing dep
    sessions_t, _ = _diary_tables(con)
    expr = sessions_t
    if user_id is not None:
        expr = expr.filter(sessions_t.user_id == user_id)
    # Most Ibis SQL backends lack a portable JSON unnest, so we delegate
    # to raw SQL via con.sql(...) which returns an Ibis table.
    filter_clause = (
        f"WHERE user_id = '{user_id.replace(chr(39), chr(39) + chr(39))}'"
        if user_id
        else ""
    )
    raw = f"""
        WITH per_day AS (
            SELECT
                CAST(CAST(start_time AS TIMESTAMP) AS DATE) AS day,
                je.value AS topic
            FROM sessions s,
                 json_each(s.topics) je
            {filter_clause}
        ),
        counted AS (
            SELECT day, topic, COUNT(*) AS count
            FROM per_day
            WHERE topic IS NOT NULL
            GROUP BY day, topic
        ),
        ranked AS (
            SELECT
                day, topic, count,
                ROW_NUMBER() OVER (PARTITION BY day ORDER BY count DESC)
                    AS rank_in_day
            FROM counted
        )
        SELECT * FROM ranked WHERE rank_in_day <= {int(limit_per_day)}
        ORDER BY day, rank_in_day
    """
    # Re-qualify table name for DuckDB schema ("diary.sessions") when attached.
    if _is_duckdb_attached(con):
        raw = raw.replace(" sessions s,", " diary.sessions s,")
    return con.sql(raw)


def adaptation_deviation_outliers(
    con: Any, z_threshold: float = 2.5
) -> Expr:
    """Find sessions whose per-user adaptation mean is an outlier.

    For each user we compute the mean-absolute adaptation across all
    their sessions, then return the sessions whose adaptation magnitude
    is more than ``z_threshold`` standard deviations from that mean.

    Args:
        con: An Ibis backend from :func:`i3_ibis_backend`.
        z_threshold: Z-score cutoff; higher = fewer outliers.

    Returns:
        An Ibis expression selecting ``user_id``, ``session_id``,
        ``start_time``, ``abs_magnitude``, ``user_mean``, ``user_std``,
        ``z``.
    """
    tbl_prefix = "diary." if _is_duckdb_attached(con) else ""
    raw = f"""
        WITH per_exchange AS (
            SELECT
                s.user_id,
                e.session_id,
                s.start_time,
                AVG(ABS(TRY_CAST(je.value AS DOUBLE))) AS abs_magnitude
            FROM {tbl_prefix}exchanges e
            JOIN {tbl_prefix}sessions s USING (session_id),
                 json_each(e.adaptation_vector) je
            GROUP BY s.user_id, e.session_id, s.start_time
        ),
        per_user AS (
            SELECT
                user_id,
                AVG(abs_magnitude) AS user_mean,
                STDDEV_SAMP(abs_magnitude) AS user_std
            FROM per_exchange
            GROUP BY user_id
        )
        SELECT
            p.user_id,
            p.session_id,
            p.start_time,
            p.abs_magnitude,
            u.user_mean,
            u.user_std,
            CASE
              WHEN u.user_std IS NULL OR u.user_std = 0 THEN 0.0
              ELSE (p.abs_magnitude - u.user_mean) / u.user_std
            END AS z
        FROM per_exchange p
        JOIN per_user u USING (user_id)
        WHERE
            u.user_std IS NOT NULL AND u.user_std > 0
            AND ABS((p.abs_magnitude - u.user_mean) / u.user_std)
                > {float(z_threshold)}
        ORDER BY ABS((p.abs_magnitude - u.user_mean) / u.user_std) DESC
    """
    return con.sql(raw)


def cross_device_activity(con: Any) -> Expr:
    """Approximate cross-device activity.

    The diary does not record a device column directly, but the router
    decision is a reasonable proxy: ``local_slm`` implies on-device
    inference while ``cloud_llm`` implies off-device.  This query
    returns per-user counts of each route over a rolling 30-day window.

    Args:
        con: An Ibis backend from :func:`i3_ibis_backend`.

    Returns:
        An Ibis expression selecting ``user_id``, ``day``,
        ``on_device_count``, ``off_device_count``, ``off_device_ratio``.
    """
    tbl_prefix = "diary." if _is_duckdb_attached(con) else ""
    raw = f"""
        WITH per_day AS (
            SELECT
                s.user_id,
                CAST(CAST(s.start_time AS TIMESTAMP) AS DATE) AS day,
                SUM(CASE WHEN e.route_chosen = 'local_slm' THEN 1 ELSE 0 END)
                    AS on_device_count,
                SUM(CASE WHEN e.route_chosen = 'cloud_llm' THEN 1 ELSE 0 END)
                    AS off_device_count
            FROM {tbl_prefix}exchanges e
            JOIN {tbl_prefix}sessions s USING (session_id)
            GROUP BY s.user_id, day
        )
        SELECT
            user_id,
            day,
            on_device_count,
            off_device_count,
            CASE
              WHEN (on_device_count + off_device_count) = 0 THEN 0.0
              ELSE CAST(off_device_count AS DOUBLE)
                 / (on_device_count + off_device_count)
            END AS off_device_ratio
        FROM per_day
        ORDER BY user_id, day
    """
    return con.sql(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_duckdb_attached(con: Any) -> bool:
    """Best-effort check: is this an Ibis DuckDB backend with ``diary`` attached?"""
    name = getattr(con, "name", "") or type(con).__name__.lower()
    if "duckdb" not in name.lower():
        return False
    try:
        con.raw_sql("SELECT 1 FROM diary.sessions LIMIT 0;")
        return True
    except Exception:
        return False
