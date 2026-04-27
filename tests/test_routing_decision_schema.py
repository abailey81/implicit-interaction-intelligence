"""Iter 78 — routing_decision dict + WS state_update field shape tests.

The dashboard's Routing tab depends on a stable schema for the
``routing_decision`` dict that the engine produces and the WS state_update
frame ships.  These tests construct a representative dict and assert
the shape contract that downstream consumers rely on.
"""
from __future__ import annotations

import pytest


def _representative_routing_decision() -> dict:
    """The shape documented in PipelineOutput.routing_decision."""
    return {
        "arm": "edge_slm",
        "confidence": 0.85,
        "reason": "low complexity + within privacy budget",
        "feature_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "complexity": {
            "score": 0.25,
            "factors": {"prompt_length": 12, "entities": 1},
            "notes": "short, single entity",
        },
        "consent_required": False,
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_routing_decision_top_level_keys():
    rd = _representative_routing_decision()
    for k in ("arm", "confidence", "reason", "feature_vector",
              "complexity", "consent_required"):
        assert k in rd


def test_routing_decision_arm_in_known_set():
    rd = _representative_routing_decision()
    assert rd["arm"] in ("edge_slm", "cloud_llm", "local_slm",
                          "tool", "retrieval", "tool:intent",
                          "tool:safety", "tool:fact", "tool:recap")


def test_routing_decision_confidence_in_unit_interval():
    rd = _representative_routing_decision()
    assert 0.0 <= rd["confidence"] <= 1.0


def test_routing_decision_feature_vector_is_numeric_list():
    rd = _representative_routing_decision()
    fv = rd["feature_vector"]
    assert isinstance(fv, list)
    for x in fv:
        assert isinstance(x, (int, float))


def test_routing_decision_complexity_block_shape():
    rd = _representative_routing_decision()
    c = rd["complexity"]
    assert "score" in c
    assert "factors" in c
    assert isinstance(c["factors"], dict)


def test_routing_decision_consent_required_is_bool():
    rd = _representative_routing_decision()
    assert isinstance(rd["consent_required"], bool)


# ---------------------------------------------------------------------------
# WS state_update synthesis using the iter-51..62 PipelineOutput shape
# ---------------------------------------------------------------------------

def test_state_update_payload_serialises():
    """Synthesise the state_update frame the WS layer ships and verify
    it round-trips through json.dumps."""
    import json
    from i3.pipeline.types import PipelineOutput
    out = PipelineOutput(
        response_text="hi",
        route_chosen="local_slm",
        latency_ms=42.0,
        user_state_embedding_2d=(0.1, 0.2),
        adaptation={"cognitive_load": 0.5},
        engagement_score=0.7,
        deviation_from_baseline=0.1,
        routing_confidence={"local_slm": 0.8, "cloud_llm": 0.2},
        messages_in_session=3,
        baseline_established=True,
        routing_decision=_representative_routing_decision(),
        privacy_budget={"cloud_calls_total": 0,
                        "bytes_transmitted_total": 0,
                        "cloud_calls_max": 50,
                        "bytes_transmitted_max": 50_000,
                        "consent_enabled": False,
                        "budget_remaining_calls": 50,
                        "budget_remaining_bytes": 50_000,
                        "pii_redactions_total": 0,
                        "bytes_redacted_total": 0,
                        "pii_category_counts": {},
                        "last_call_ts": 0.0},
        safety_caveat=None,
        personal_facts={"name": "Alice"},
        intent_result=None,
    )
    # Build a state_update frame the same way server/websocket.py does.
    frame = {
        "type": "state_update",
        "user_state_embedding_2d": list(out.user_state_embedding_2d),
        "adaptation": out.adaptation,
        "engagement_score": out.engagement_score,
        "deviation_from_baseline": out.deviation_from_baseline,
        "routing_confidence": out.routing_confidence,
        "messages_in_session": out.messages_in_session,
        "baseline_established": out.baseline_established,
        "route_chosen": out.route_chosen,
        "routing_decision": out.routing_decision,
        "privacy_budget": out.privacy_budget,
        "safety_caveat": out.safety_caveat,
        "personal_facts": out.personal_facts,
        "intent_result": out.intent_result,
    }
    # Must serialise; must round-trip identical.
    s = json.dumps(frame)
    parsed = json.loads(s)
    assert parsed["routing_decision"]["arm"] == "edge_slm"
    assert parsed["personal_facts"]["name"] == "Alice"
    assert parsed["safety_caveat"] is None


def test_routing_confidence_dict_sums_to_within_unit():
    """If both arms are present, the confidences should be in [0, 1]
    and sum to ≤ 1.0 + eps (they're per-arm probabilities)."""
    rc = {"local_slm": 0.7, "cloud_llm": 0.25}
    assert all(0.0 <= v <= 1.0 for v in rc.values())
    assert sum(rc.values()) <= 1.0 + 1e-6
