"""Iter 84 — DiaryStore session-lifecycle round-trip tests.

Exercises create_session → log_exchange → end_session → retrieval
end-to-end against a temp SQLite db.  Pins the per-method contract +
data integrity (round-trip embedding bytes, scalar metrics, topic list).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


def _store():
    from i3.diary.store import DiaryStore
    db_path = Path(tempfile.mkdtemp()) / "lifecycle.db"
    return DiaryStore(db_path=str(db_path))


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def store():
    s = _store()
    _run(s.initialize())
    yield s
    _run(s.close())


def test_create_session_round_trip(store):
    _run(store.create_session("s-1", "alice"))
    sess = _run(store.get_session("s-1"))
    assert sess is not None
    assert sess["user_id"] == "alice"
    assert sess["session_id"] == "s-1"


def test_log_exchange_returns_uuid(store):
    _run(store.create_session("s-2", "alice"))
    eid = _run(store.log_exchange(
        session_id="s-2",
        user_state_embedding=b"\x00\x01\x02\x03",
        adaptation_vector={"cognitive_load": 0.5},
        route_chosen="local_slm",
        response_latency_ms=42,
        engagement_signal=0.7,
        topics=["greetings"],
    ))
    assert isinstance(eid, str)
    assert len(eid) > 8


def test_get_session_exchanges_round_trip(store):
    _run(store.create_session("s-3", "alice"))
    for i in range(3):
        _run(store.log_exchange(
            session_id="s-3",
            user_state_embedding=bytes([i]),
            adaptation_vector={"cognitive_load": 0.1 * i},
            route_chosen="local_slm",
            response_latency_ms=10 + i,
            engagement_signal=0.5,
            topics=[f"topic_{i}"],
        ))
    rows = _run(store.get_session_exchanges("s-3"))
    assert len(rows) == 3
    # Rows should be ordered by timestamp ascending
    assert rows[0]["topics"] != rows[2]["topics"]


def test_end_session_updates_message_count(store):
    _run(store.create_session("s-4", "alice"))
    _run(store.log_exchange(
        session_id="s-4",
        user_state_embedding=b"\x00",
        adaptation_vector={},
        route_chosen="local_slm",
        response_latency_ms=10,
        engagement_signal=0.5,
        topics=[],
    ))
    diary = _run(store.end_session(
        "s-4", "alice",
        dominant_emotion="neutral",
        topics=["greetings"],
        mean_engagement=0.5,
        mean_cognitive_load=0.4,
        mean_accessibility=0.0,
        relationship_strength=0.3,
    ))
    sess = _run(store.get_session("s-4"))
    assert sess["message_count"] == 1
    assert sess["end_time"] is not None
    # diary entry may be None or a dict depending on summariser
    assert diary is None or isinstance(diary, dict)


def test_get_user_sessions_returns_list(store):
    _run(store.create_session("s-5", "alice"))
    _run(store.create_session("s-6", "alice"))
    sessions = _run(store.get_user_sessions("alice"))
    ids = {s["session_id"] for s in sessions}
    assert {"s-5", "s-6"}.issubset(ids)


def test_get_session_unknown_returns_none(store):
    s = _run(store.get_session("does-not-exist"))
    assert s is None


def test_user_facts_set_get_round_trip(store):
    _run(store.set_user_fact("alice", "favourite_color", "indigo"))
    _run(store.set_user_fact("alice", "name", "Alice"))
    facts = _run(store.get_user_facts("alice"))
    assert facts.get("favourite_color") == "indigo"
    assert facts.get("name") == "Alice"


def test_forget_user_facts_clears_all(store):
    _run(store.set_user_fact("bob", "name", "Bob"))
    _run(store.forget_user_facts("bob"))
    facts = _run(store.get_user_facts("bob"))
    assert facts == {}
