"""Analytics layer for Implicit Interaction Intelligence (I3).

This package adds a **read-only analytics stack** that sits on top of the
existing ``aiosqlite`` diary and user-model stores without modifying them.
It uses the 2026 modern data stack:

* **DuckDB** (embedded OLAP) attaches the SQLite diary file directly and
  executes analytical queries an order of magnitude faster than SQLite.
* **LanceDB** (embedded vector DB) indexes user/session embeddings for
  fast approximate-nearest-neighbour similarity search via IVF-PQ.
* **Polars** provides Rust-backed columnar DataFrames and streaming
  feature pipelines.
* **Ibis** supplies a portable DataFrame-to-SQL front-end (DuckDB,
  SQLite, PostgreSQL all share the same expression API).
* **Apache Arrow + Parquet** handle zero-copy columnar interop and
  on-disk snapshots.

Privacy
-------
The analytics layer inherits the I3 **"no raw text"** guarantee because
it can only read columns that already exist in the diary/user-model
schemas, and those schemas have no text fields (only embeddings, scalar
metrics, topic keywords and adaptation parameters).

All heavy dependencies are **soft-imported** at call-time — importing
:mod:`i3.analytics` has no side effects on a stock install.
"""

from __future__ import annotations

__all__ = [
    "DuckDBAnalytics",
    "LanceUserEmbeddingStore",
    "PolarsFeatureExtractor",
    "adaptation_deviation_outliers",
    "arrow_table_from_diary",
    "cross_device_activity",
    "embedding_batch_to_arrow",
    "i3_ibis_backend",
    "read_parquet",
    "top_topics_by_day",
    "write_parquet",
]


def __getattr__(name: str):
    """Lazily resolve public symbols to avoid importing heavy deps at import time."""
    if name == "DuckDBAnalytics":
        from i3.analytics.duckdb_engine import DuckDBAnalytics

        return DuckDBAnalytics
    if name == "LanceUserEmbeddingStore":
        from i3.analytics.lance_vector import LanceUserEmbeddingStore

        return LanceUserEmbeddingStore
    if name == "PolarsFeatureExtractor":
        from i3.analytics.polars_features import PolarsFeatureExtractor

        return PolarsFeatureExtractor
    if name in {
        "i3_ibis_backend",
        "top_topics_by_day",
        "adaptation_deviation_outliers",
        "cross_device_activity",
    }:
        from i3.analytics import ibis_queries as _iq

        return getattr(_iq, name)
    if name in {
        "embedding_batch_to_arrow",
        "arrow_table_from_diary",
        "write_parquet",
        "read_parquet",
    }:
        from i3.analytics import arrow_interop as _ai

        return getattr(_ai, name)
    raise AttributeError(f"module 'i3.analytics' has no attribute {name!r}")
