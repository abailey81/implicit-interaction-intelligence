"""Iter 61 — PipelineOutput dataclass contract tests.

The WebSocket frame schema (server/websocket.py) reads ~30 fields from
PipelineOutput.  A field rename or accidental drop silently breaks
the dashboard.  These tests pin the contract: every field the WS
layer reads must exist on the dataclass with the right default.
"""
from __future__ import annotations

import pytest

from i3.pipeline.types import PipelineOutput


# Fields the WebSocket layer reads (extracted from server/websocket.py).
# Each entry: (field name, expected default for None-able fields).
_WS_FIELDS_REQUIRED = [
    "response_text",
    "route_chosen",
    "latency_ms",
    "user_state_embedding_2d",
    "adaptation",
    "engagement_score",
    "deviation_from_baseline",
    "routing_confidence",
    "messages_in_session",
    "baseline_established",
]

_WS_FIELDS_OPTIONAL = [
    "diary_entry",
    "response_path",
    "retrieval_score",
    "adaptation_changes",
    "affect_shift",
    "safety_caveat",
    "user_state_label",
    "accessibility",
    "biometric",
    "coreference_resolution",
    "personalisation",
    "critique",
    "multimodal",
    "gaze",
    "pipeline_trace",
    "routing_decision",
    "privacy_budget",
    "safety",
    "session_memory",
    "explain_plan",
    "personal_facts",
    "intent_result",
]


def _minimal() -> PipelineOutput:
    """Construct PipelineOutput with the smallest viable arg set."""
    return PipelineOutput(
        response_text="hi",
        route_chosen="local_slm",
        latency_ms=10.0,
        user_state_embedding_2d=(0.0, 0.0),
        adaptation={},
        engagement_score=0.5,
        deviation_from_baseline=0.0,
        routing_confidence={"local_slm": 1.0},
        messages_in_session=1,
        baseline_established=True,
    )


def test_required_fields_set_via_init():
    out = _minimal()
    for f in _WS_FIELDS_REQUIRED:
        assert hasattr(out, f), f"required field {f!r} missing"


@pytest.mark.parametrize("f", _WS_FIELDS_OPTIONAL)
def test_optional_field_present_with_default(f):
    out = _minimal()
    assert hasattr(out, f), f"optional field {f!r} missing"


def test_optional_fields_default_to_none_or_empty():
    """Optional fields should default to None, empty container, or a
    documented sentinel string so the WS layer's getattr(...) calls
    don't surface garbage."""
    out = _minimal()
    sentinels = {
        "response_path": "unknown",  # explicit string sentinel by design
        "retrieval_score": 0.0,      # explicit numeric default
    }
    for f in _WS_FIELDS_OPTIONAL:
        v = getattr(out, f)
        if f in sentinels:
            assert v == sentinels[f], (
                f"field {f!r} default {v!r} != sentinel {sentinels[f]!r}"
            )
            continue
        assert v is None or v == [] or v == {}, (
            f"field {f!r} default unexpected: {v!r}"
        )


def test_iter51_fields_present():
    """Iter 51 added safety_caveat / personal_facts / intent_result —
    pin them explicitly so a future refactor that drops one breaks
    fast."""
    out = _minimal()
    assert hasattr(out, "safety_caveat")
    assert hasattr(out, "personal_facts")
    assert hasattr(out, "intent_result")
    assert out.safety_caveat is None
    assert out.personal_facts is None
    assert out.intent_result is None


def test_can_construct_with_all_iter51_fields():
    out = PipelineOutput(
        response_text="ok",
        route_chosen="local_slm",
        latency_ms=42.0,
        user_state_embedding_2d=(0.1, 0.2),
        adaptation={"cognitive_load": 0.5},
        engagement_score=0.7,
        deviation_from_baseline=0.1,
        routing_confidence={"local_slm": 0.8, "cloud_llm": 0.2},
        messages_in_session=3,
        baseline_established=True,
        safety_caveat="ⓘ note",
        personal_facts={"name": "Alice"},
        intent_result={"action": "set_timer", "params": {"duration_seconds": 60}},
    )
    assert out.safety_caveat == "ⓘ note"
    assert out.personal_facts == {"name": "Alice"}
    assert out.intent_result["action"] == "set_timer"
