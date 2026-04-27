"""Iter 76 — DiaryStore schema PII-free contract tests.

The diary schema is the *durable* persistence layer.  It is
intentionally PII-free: the exchanges table stores embeddings (BLOB),
scalar metrics, routes, and topic keywords — NEVER the raw user or
assistant text.  These tests pin that contract by inspecting the
SQLite schema after an open() call.

If a future migration accidentally adds a column matching one of the
forbidden names, the test fails fast and the privacy guarantee is
preserved.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


# Names that MUST NEVER appear as column names in the exchanges or
# sessions table.  Any of these would mean the schema started
# persisting raw natural-language text — a hard privacy violation.
_FORBIDDEN_COLS = {
    "text", "message", "content", "body", "raw_text",
    "user_text", "user_message", "input_text", "input",
    "assistant_text", "assistant_response", "ai_response",
    "response_text", "completion", "completion_text",
    "prompt", "prompt_text",
}


def _open_diary():
    """Open a fresh DiaryStore against a temp SQLite file."""
    from i3.diary.store import DiaryStore
    db_path = Path(tempfile.mkdtemp()) / "test_diary.db"
    store = DiaryStore(db_path=str(db_path))
    return store, db_path


@pytest.fixture
def diary():
    store, db_path = _open_diary()
    asyncio.get_event_loop().run_until_complete(store.initialize())
    yield store, db_path
    asyncio.get_event_loop().run_until_complete(store.close())


def _columns(db_path: Path, table: str) -> list[str]:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]
    finally:
        conn.close()


def test_exchanges_table_has_no_natural_language_columns(diary):
    _store, db_path = diary
    cols = _columns(db_path, "exchanges")
    forbidden_present = set(c.lower() for c in cols) & _FORBIDDEN_COLS
    assert not forbidden_present, \
        f"PII-leak risk: exchanges has columns {forbidden_present}"


def test_sessions_summary_is_only_natural_language_column(diary):
    """Sessions.summary is the *only* permitted text column — and it
    is populated by SessionSummariser from aggregated metadata, never
    raw text.  Verify no other text-leaking column slipped in."""
    _store, db_path = diary
    cols = set(c.lower() for c in _columns(db_path, "sessions"))
    forbidden = (cols & _FORBIDDEN_COLS) - {"summary"}
    assert not forbidden, \
        f"PII-leak risk: sessions has columns {forbidden}"


def test_exchanges_has_required_metric_columns(diary):
    """Iter 76 contract: exchanges retains the per-turn analytics
    columns the dashboard depends on."""
    _store, db_path = diary
    cols = set(c.lower() for c in _columns(db_path, "exchanges"))
    required = {"exchange_id", "session_id", "timestamp",
                "user_state_embedding", "adaptation_vector",
                "route_chosen", "response_latency_ms",
                "engagement_signal", "topics"}
    missing = required - cols
    assert not missing, f"exchanges missing required columns: {missing}"


def test_user_facts_table_uses_blob_for_value(diary):
    """The iter-50 user_facts table must use BLOB (so encrypted bytes
    are storeable) — never TEXT (which would persist plaintext if
    encryption was disabled)."""
    import sqlite3
    _store, db_path = diary
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("PRAGMA table_info(user_facts)")
        cols = {row[1].lower(): row[2].upper() for row in cur.fetchall()}
    finally:
        conn.close()
    assert cols.get("value_blob") == "BLOB", \
        f"user_facts.value_blob is {cols.get('value_blob')!r}, must be BLOB"
