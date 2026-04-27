"""Iter 77 — IntentResult / SUPPORTED_ACTIONS / ACTION_SLOTS contract tests.

Pins the canonical action vocabulary + per-action slot whitelist + the
IntentResult.to_dict() shape that the WS layer + dashboard depend on.
"""
from __future__ import annotations

import pytest

from i3.intent.types import (
    ACTION_SLOTS,
    SUPPORTED_ACTIONS,
    IntentResult,
)


# ---------------------------------------------------------------------------
# Vocabulary / slot whitelist invariants
# ---------------------------------------------------------------------------

def test_supported_actions_is_tuple_of_strings():
    assert isinstance(SUPPORTED_ACTIONS, tuple)
    assert len(SUPPORTED_ACTIONS) >= 10
    for a in SUPPORTED_ACTIONS:
        assert isinstance(a, str) and a.islower()


def test_action_slots_keys_match_supported_actions():
    """Every action listed in SUPPORTED_ACTIONS must have a slot whitelist
    (even if empty for cancel / unsupported)."""
    assert set(ACTION_SLOTS.keys()) == set(SUPPORTED_ACTIONS)


def test_cancel_and_unsupported_have_no_slots():
    assert ACTION_SLOTS["cancel"] == set()
    assert ACTION_SLOTS["unsupported"] == set()


def test_set_timer_requires_duration_seconds():
    assert ACTION_SLOTS["set_timer"] == {"duration_seconds"}


def test_send_message_has_recipient_and_message():
    assert ACTION_SLOTS["send_message"] == {"recipient", "message"}


# ---------------------------------------------------------------------------
# IntentResult dataclass + confidence + to_dict
# ---------------------------------------------------------------------------

def _ok():
    return IntentResult(
        raw_input="set timer for 5 min",
        raw_output='{"action":"set_timer","params":{"duration_seconds":300}}',
        parsed={"action": "set_timer", "params": {"duration_seconds": 300}},
        valid_json=True,
        valid_action=True,
        valid_slots=True,
        action="set_timer",
        params={"duration_seconds": 300},
        latency_ms=42.123,
        backend="qwen-lora",
    )


def test_confidence_full_when_all_valid():
    r = _ok()
    assert r.confidence == pytest.approx(1.0, abs=0.001)


def test_confidence_partial_when_only_json_valid():
    r = IntentResult(
        raw_input="?", raw_output="{}", parsed={"action": "wat"},
        valid_json=True, valid_action=False, valid_slots=False,
    )
    assert 0.3 <= r.confidence <= 0.5


def test_confidence_zero_when_nothing_valid():
    r = IntentResult(
        raw_input="?", raw_output="prose", parsed=None,
        valid_json=False, valid_action=False, valid_slots=False,
    )
    assert r.confidence == 0.0


def test_to_dict_shape():
    r = _ok()
    d = r.to_dict()
    for k in ("raw_input", "action", "params", "valid_json",
              "valid_action", "valid_slots", "confidence",
              "latency_ms", "backend", "error"):
        assert k in d, f"missing {k!r} in IntentResult.to_dict()"
    # Latency rounded to 2 dp
    assert d["latency_ms"] == 42.12


def test_to_dict_serializes_cleanly_to_json():
    """Ensure the dict can survive json.dumps without exotic types."""
    import json
    r = _ok()
    s = json.dumps(r.to_dict())
    parsed = json.loads(s)
    assert parsed["action"] == "set_timer"


def test_default_factory_isolation():
    """Two IntentResults with default params must have isolated dicts."""
    a = IntentResult(raw_input="", raw_output="", parsed=None,
                     valid_json=False, valid_action=False, valid_slots=False)
    b = IntentResult(raw_input="", raw_output="", parsed=None,
                     valid_json=False, valid_action=False, valid_slots=False)
    a.params["x"] = 1
    assert "x" not in b.params, "default_factory dict shared between instances"
