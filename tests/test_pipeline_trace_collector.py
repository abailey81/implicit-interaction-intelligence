"""Iter 105 — PipelineTraceCollector smoke + invariants.

Pins the trace-collection helper that powers the Flow dashboard
(web/js/flow_dashboard.js).  Each chat turn opens stage(...) context
managers, and the collector emits a JSON-safe trace dict.
"""
from __future__ import annotations

import pytest

from i3.observability.pipeline_trace import (
    PipelineTrace,
    PipelineTraceCollector,
    StageRecord,
    TurnHandle,
)


@pytest.fixture
def coll():
    return PipelineTraceCollector(max_traces_in_memory=10)


def test_start_turn_returns_handle(coll):
    h = coll.start_turn("alice", "s1")
    assert isinstance(h, TurnHandle)
    assert h.user_id == "alice"
    assert h.session_id == "s1"
    assert isinstance(h.turn_id, str) and len(h.turn_id) > 8


def test_stage_records_to_handle(coll):
    h = coll.start_turn("u", "s")
    with coll.stage(h, "encoder", "Encoder forward") as rec:
        assert isinstance(rec, StageRecord)
    assert len(h.stages) == 1
    assert h.stages[0].stage_id == "encoder"
    assert h.stages[0].fired is True


def test_stage_records_timing(coll):
    h = coll.start_turn("u", "s")
    with coll.stage(h, "x", "X stage"):
        pass
    rec = h.stages[0]
    assert rec.started_at_ms is not None
    assert rec.ended_at_ms is not None
    assert rec.ended_at_ms >= rec.started_at_ms


def test_finalise_returns_dict(coll):
    h = coll.start_turn("u", "s")
    with coll.stage(h, "a", "A"):
        pass
    with coll.stage(h, "b", "B"):
        pass
    out = coll.finalise(h)
    assert isinstance(out, dict)
    assert "stages" in out
    assert len(out["stages"]) == 2


def test_finalise_round_trips_json(coll):
    import json
    h = coll.start_turn("u", "s")
    with coll.stage(h, "a", "A"):
        pass
    out = coll.finalise(h)
    s = json.dumps(out)
    parsed = json.loads(s)
    assert parsed["stages"][0]["stage_id"] == "a"


def test_recent_returns_list(coll):
    for i in range(3):
        h = coll.start_turn("u", "s")
        with coll.stage(h, f"x{i}", "X"):
            pass
        coll.finalise(h)
    rec = coll.recent(n=10)
    assert isinstance(rec, list)
    assert len(rec) >= 3


def test_arrow_can_be_recorded(coll):
    h = coll.start_turn("u", "s")
    with coll.stage(h, "a", "A"):
        pass
    with coll.stage(h, "b", "B"):
        pass
    coll.arrow(h, "a", "b")
    out = coll.finalise(h)
    assert "arrow_flows" in out


def test_max_stages_per_turn_caps(coll):
    h = coll.start_turn("u", "s")
    # Push way more than the per-turn cap
    for i in range(500):
        with coll.stage(h, f"s{i}", f"S{i}"):
            pass
    # Should not have unbounded growth — cap enforced
    assert len(h.stages) < 500
