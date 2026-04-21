"""Integration tests for the DuckDB, Polars, Ibis and Arrow paths.

Each test builds a tiny SQLite diary with 100 synthetic rows, then
exercises one of the :mod:`i3.analytics` adapters against it.  The
tests are skipped gracefully when any of the heavy optional
dependencies are missing, so CI can run on either the minimal or the
full install.
"""

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# --- soft-dep gating ------------------------------------------------------

duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")
pl = pytest.importorskip("polars", reason="polars not installed")
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")


# --------------------------------------------------------------------------
# Fixture: synthetic diary
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_diary(tmp_path: Path) -> Path:
    """Create a 100-session diary with 3 exchanges per session."""
    db_path = tmp_path / "diary.db"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            summary TEXT,
            dominant_emotion TEXT,
            topics TEXT,
            mean_engagement REAL,
            mean_cognitive_load REAL,
            mean_accessibility REAL,
            relationship_strength REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE exchanges (
            exchange_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            user_state_embedding BLOB,
            adaptation_vector TEXT,
            route_chosen TEXT,
            response_latency_ms INTEGER,
            engagement_signal REAL,
            topics TEXT
        );
        """
    )
    rng = random.Random(42)
    users = [f"u{i}" for i in range(5)]
    base_ts = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    sessions_rows = []
    exchanges_rows = []
    for s_idx in range(100):
        sid = str(uuid.uuid4())
        uid = rng.choice(users)
        ts = base_ts + timedelta(hours=s_idx * 3)
        mean_eng = rng.uniform(0.3, 0.9)
        topics_list = rng.sample(
            ["math", "code", "history", "art", "sports", "cooking"], k=3
        )
        sessions_rows.append(
            (
                sid,
                uid,
                ts.isoformat(),
                (ts + timedelta(minutes=15)).isoformat(),
                3,
                "summary text is auto-generated not raw",
                rng.choice(["joy", "neutral", "curious"]),
                json.dumps(topics_list, sort_keys=True),
                mean_eng,
                rng.uniform(0.1, 0.7),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.0, 1.0),
            )
        )
        for ex_idx in range(3):
            adaptation = {
                "verbosity": rng.uniform(-1, 1),
                "warmth": rng.uniform(-1, 1),
                "technicality": rng.uniform(-1, 1),
            }
            exchanges_rows.append(
                (
                    str(uuid.uuid4()),
                    sid,
                    (ts + timedelta(minutes=ex_idx * 5)).isoformat(),
                    b"\x00",
                    json.dumps(adaptation, sort_keys=True),
                    rng.choice(["local_slm", "cloud_llm"]),
                    rng.randint(30, 800),
                    rng.uniform(0.2, 0.95),
                    json.dumps(topics_list[:1], sort_keys=True),
                )
            )
    con.executemany(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        sessions_rows,
    )
    con.executemany(
        "INSERT INTO exchanges VALUES (?,?,?,?,?,?,?,?,?)", exchanges_rows
    )
    con.commit()
    con.close()
    return db_path


# --------------------------------------------------------------------------
# DuckDB engine
# --------------------------------------------------------------------------


def test_duckdb_route_distribution(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        df = ddb.route_distribution()
    assert set(df.columns) >= {"route", "count", "fraction"}
    assert df["count"].sum() == 300
    # Fractions should sum to 1.0 (up to float tolerance).
    assert abs(df["fraction"].sum() - 1.0) < 1e-6
    assert set(df["route"].to_list()).issubset({"local_slm", "cloud_llm"})


def test_duckdb_latency_percentiles(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        df = ddb.latency_percentiles()
    assert set(df.columns) >= {"route", "p50_ms", "p95_ms", "p99_ms", "n"}
    for p50, p95, p99 in zip(
        df["p50_ms"].to_list(),
        df["p95_ms"].to_list(),
        df["p99_ms"].to_list(),
    ):
        assert p50 <= p95 <= p99


def test_duckdb_adaptation_distribution_keys(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        df = ddb.adaptation_distribution()
    dims = set(df["dimension"].to_list())
    assert dims == {"verbosity", "warmth", "technicality"}
    for mean, std in zip(df["mean"].to_list(), df["std"].to_list()):
        assert -1.0 <= mean <= 1.0
        assert std >= 0.0


def test_duckdb_per_user_filter(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        a = ddb.route_distribution()
        b = ddb.route_distribution(user_id="u0")
    assert a["count"].sum() == 300
    assert b["count"].sum() < a["count"].sum()


def test_duckdb_session_heatmap(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        df = ddb.session_heatmap_by_hour("u0")
    assert set(df.columns) >= {"weekday", "hour", "sessions", "mean_engagement"}
    assert df["sessions"].sum() > 0
    for w in df["weekday"].to_list():
        assert 0 <= int(w) <= 6
    for h in df["hour"].to_list():
        assert 0 <= int(h) <= 23


def test_duckdb_rolling_engagement(synthetic_diary: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        df = ddb.rolling_engagement("u0", window_days=3)
    assert set(df.columns) >= {"day", "sessions_in_day", "rolling_mean_engagement"}
    assert df.height >= 1
    # Rolling means must be within [0, 1] given engagement is in [0.3, 0.9].
    for v in df["rolling_mean_engagement"].to_list():
        if v is not None:
            assert 0.0 <= v <= 1.0


def test_duckdb_read_only(synthetic_diary: Path) -> None:
    """Attempting to write through DuckDB must raise."""
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with DuckDBAnalytics(synthetic_diary) as ddb:
        con = ddb._ensure_connected()  # noqa: SLF001
        with pytest.raises(Exception):
            con.execute("INSERT INTO diary.sessions VALUES ('x','u','2026-01-01','',0,'','','','0','0','0','0',CURRENT_TIMESTAMP)")


def test_duckdb_missing_file(tmp_path: Path) -> None:
    from i3.analytics.duckdb_engine import DuckDBAnalytics

    with pytest.raises(FileNotFoundError):
        DuckDBAnalytics(tmp_path / "nope.db").connect()


# --------------------------------------------------------------------------
# Polars feature extractor
# --------------------------------------------------------------------------


def test_polars_feature_extractor_shape(synthetic_diary: Path) -> None:
    from i3.analytics.polars_features import (
        FEATURE_NAMES,
        PolarsFeatureExtractor,
    )

    ext = PolarsFeatureExtractor(synthetic_diary)
    df = ext.extract()
    assert "user_id" in df.columns
    # 32 feature names + user_id
    assert len(df.columns) == len(FEATURE_NAMES) + 1
    for name in FEATURE_NAMES:
        assert name in df.columns
    assert df.height == 5  # 5 synthetic users


def test_polars_feature_extractor_single_user(synthetic_diary: Path) -> None:
    from i3.analytics.polars_features import PolarsFeatureExtractor

    ext = PolarsFeatureExtractor(synthetic_diary)
    df = ext.extract(user_id="u0")
    assert df.height == 1
    assert df["user_id"].to_list() == ["u0"]


# --------------------------------------------------------------------------
# Arrow interop + Parquet round-trip
# --------------------------------------------------------------------------


def test_arrow_table_from_diary(synthetic_diary: Path) -> None:
    from i3.analytics.arrow_interop import arrow_table_from_diary

    tbl = arrow_table_from_diary(synthetic_diary, table="sessions")
    assert tbl.num_rows == 100
    assert "user_id" in tbl.column_names


def test_arrow_filter_injection_rejected(synthetic_diary: Path) -> None:
    from i3.analytics.arrow_interop import arrow_table_from_diary

    with pytest.raises(ValueError):
        arrow_table_from_diary(
            synthetic_diary, table="sessions", filters={"user_id; DROP TABLE sessions": "x"}
        )


def test_embedding_batch_to_arrow() -> None:
    import numpy as np

    from i3.analytics.arrow_interop import embedding_batch_to_arrow

    embs = [np.ones(16, dtype=np.float32) * i for i in range(3)]
    tbl = embedding_batch_to_arrow(embs, embedding_dim=16, ids=["a", "b", "c"])
    assert tbl.num_rows == 3
    assert tbl.column_names == ["id", "embedding"]


def test_parquet_roundtrip(tmp_path: Path, synthetic_diary: Path) -> None:
    from i3.analytics.arrow_interop import (
        arrow_table_from_diary,
        read_parquet,
        write_parquet,
    )

    tbl = arrow_table_from_diary(synthetic_diary, table="sessions")
    out = write_parquet(tbl, tmp_path / "snap.parquet")
    loaded = read_parquet(out)
    assert loaded.num_rows == tbl.num_rows
    assert set(loaded.column_names) == set(tbl.column_names)


# --------------------------------------------------------------------------
# Ibis (only if installed)
# --------------------------------------------------------------------------


def test_ibis_cross_device_activity(synthetic_diary: Path) -> None:
    ibis_mod = pytest.importorskip("ibis", reason="ibis-framework not installed")
    _ = ibis_mod  # just ensuring availability
    from i3.analytics.ibis_queries import (
        cross_device_activity,
        i3_ibis_backend,
    )

    con = i3_ibis_backend(synthetic_diary)
    expr = cross_device_activity(con)
    df = expr.execute()
    assert set(df.columns) >= {
        "user_id",
        "day",
        "on_device_count",
        "off_device_count",
        "off_device_ratio",
    }
    assert len(df) > 0
