"""Iter 118 — StageRecord field-clipping + dataclass invariants."""
from __future__ import annotations

import pytest

from i3.observability.pipeline_trace import StageRecord


def test_stage_record_minimal_construction():
    sr = StageRecord(stage_id="encoder", label="Encoder")
    assert sr.stage_id == "encoder"
    assert sr.label == "Encoder"
    assert sr.fired is False
    assert sr.latency_ms == 0.0
    assert sr.inputs == {}
    assert sr.outputs == {}
    assert sr.notes == ""
    assert sr.is_tool is False


def test_stage_record_full_construction():
    sr = StageRecord(
        stage_id="generation",
        label="SLM generate",
        fired=True,
        started_at_ms=10.0,
        ended_at_ms=42.0,
        latency_ms=32.0,
        inputs={"tokens": 12},
        outputs={"text_len": 80},
        notes="generated 80 chars",
        is_tool=False,
    )
    assert sr.fired is True
    assert sr.latency_ms == 32.0
    assert sr.inputs["tokens"] == 12


def test_stage_record_default_factory_isolation():
    a = StageRecord(stage_id="a", label="A")
    b = StageRecord(stage_id="b", label="B")
    a.inputs["k"] = 1
    a.outputs["o"] = 2
    assert b.inputs == {}
    assert b.outputs == {}


def test_stage_record_is_tool_flag():
    sr = StageRecord(stage_id="tool:math", label="Math tool", is_tool=True)
    assert sr.is_tool is True


def test_stage_record_supports_named_args_only():
    """Dataclass should accept stage_id + label as positional or named."""
    sr = StageRecord("retrieval", "Retrieval")
    assert sr.stage_id == "retrieval"
    assert sr.label == "Retrieval"


def test_stage_record_can_be_serialised_via_asdict():
    from dataclasses import asdict
    sr = StageRecord(stage_id="x", label="X", fired=True,
                     latency_ms=5.0)
    d = asdict(sr)
    assert d["stage_id"] == "x"
    assert d["fired"] is True
    assert d["latency_ms"] == 5.0
