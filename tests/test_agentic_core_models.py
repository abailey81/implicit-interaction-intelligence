"""Iter 124 — HMAFIntent / HMAFResponse Pydantic model tests."""
from __future__ import annotations

import pytest

from i3.huawei.agentic_core_runtime import HMAFIntent, HMAFResponse


def test_hmaf_intent_minimal():
    i = HMAFIntent(name="get_user_adaptation", source_device="phone")
    assert i.name == "get_user_adaptation"
    assert i.source_device == "phone"
    assert isinstance(i.correlation_id, str)
    assert len(i.correlation_id) > 8
    assert isinstance(i.timestamp, str)
    assert i.parameters == {}


def test_hmaf_intent_with_parameters():
    i = HMAFIntent(
        name="set_timer", source_device="phone",
        parameters={"duration_seconds": 300},
    )
    assert i.parameters["duration_seconds"] == 300


def test_hmaf_intent_rejects_extra_fields():
    """extra='forbid' on the BaseModel."""
    with pytest.raises(Exception):
        HMAFIntent(
            name="x", source_device="phone",
            unknown_field="should fail",  # type: ignore[call-arg]
        )


def test_hmaf_intent_rejects_empty_name():
    with pytest.raises(Exception):
        HMAFIntent(name="", source_device="phone")


def test_hmaf_intent_rejects_empty_source_device():
    with pytest.raises(Exception):
        HMAFIntent(name="x", source_device="")


def test_hmaf_intent_correlation_id_can_be_supplied():
    cid = "test-correlation-id-12345"
    i = HMAFIntent(name="x", source_device="phone", correlation_id=cid)
    assert i.correlation_id == cid


def test_hmaf_response_minimal_construction():
    r = HMAFResponse(
        intent_name="x",
        correlation_id="test-cid",
        ok=True,
        terminal_action="route_local",
        payload={},
        steps_executed=1,
        latency_ms=10.0,
    )
    assert r.ok is True
    assert r.intent_name == "x"


def test_hmaf_response_serialises_to_dict():
    r = HMAFResponse(
        intent_name="x",
        correlation_id="cid",
        ok=False,
        terminal_action="error",
        payload={"reason": "no_handler"},
        steps_executed=0,
        latency_ms=5.0,
    )
    d = r.model_dump()
    assert d["ok"] is False
    assert d["payload"]["reason"] == "no_handler"


def test_hmaf_intent_round_trips_through_json():
    import json
    i = HMAFIntent(
        name="get_user_adaptation",
        source_device="watch",
        parameters={"user_id": "alice"},
    )
    s = i.model_dump_json()
    parsed = json.loads(s)
    assert parsed["name"] == "get_user_adaptation"
    assert parsed["source_device"] == "watch"
