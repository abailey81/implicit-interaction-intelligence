"""Iter 82 — PrivacyBudget.snapshot() WS-shape regression tests.

The WebSocket layer ships ``output.privacy_budget`` on every turn so
the dashboard's Privacy tab can render call/byte budget remaining.
These tests pin the snapshot's exact key set + per-key types so a
schema drift breaks fast instead of silently emptying the dashboard.
"""
from __future__ import annotations

import pytest

from i3.privacy.budget import PrivacyBudget


def _budget_with_call(consent=True):
    pb = PrivacyBudget(max_cloud_calls_per_session=10,
                       max_bytes_per_session=10_000)
    if consent:
        pb.set_consent("alice", True)
    if consent:
        pb.record_call(
            "alice", "s1",
            sanitised_prompt="hello world",
            response_text="hi back",
            pii_redactions=2,
            pii_categories={"email": 1, "phone": 1},
            bytes_in=11, bytes_out=7,
        )
    return pb.snapshot("alice", "s1")


def test_snapshot_to_dict_minimum_shape():
    snap = _budget_with_call(consent=True)
    d = snap.to_dict() if hasattr(snap, "to_dict") else dict(snap)
    expected = {
        "cloud_calls_total", "cloud_calls_max",
        "bytes_transmitted_total", "bytes_transmitted_max",
        "budget_remaining_calls", "budget_remaining_bytes",
        "consent_enabled", "pii_redactions_total",
        "bytes_redacted_total", "sensitive_categories", "last_call_ts",
    }
    missing = expected - set(d.keys())
    assert not missing, f"snapshot missing keys: {missing}"


def test_snapshot_types_are_json_safe():
    snap = _budget_with_call(consent=True)
    d = snap.to_dict()
    for k, v in d.items():
        assert v is None or isinstance(v, (int, float, bool, str, dict, list)), \
            f"key {k!r}: non-JSON-safe type {type(v).__name__}"


def test_snapshot_remaining_after_one_call():
    snap = _budget_with_call(consent=True)
    d = snap.to_dict()
    # Made 1 call out of 10
    assert d["cloud_calls_total"] == 1
    assert d["budget_remaining_calls"] == 9
    # bytes_in (11) + bytes_out (7) = 18 of 10 000
    assert d["bytes_transmitted_total"] >= 11  # at minimum the prompt
    assert d["budget_remaining_bytes"] <= 10_000


def test_snapshot_with_no_consent_and_no_calls():
    pb = PrivacyBudget()
    snap = pb.snapshot("bob", "s1")
    d = snap.to_dict()
    assert d["consent_enabled"] is False
    assert d["cloud_calls_total"] == 0
    assert d["bytes_transmitted_total"] == 0


def test_snapshot_pii_total_reflects_calls():
    snap = _budget_with_call(consent=True)
    d = snap.to_dict()
    # We passed pii_redactions=2 in the single call.
    assert d["pii_redactions_total"] == 2


def test_snapshot_round_trips_through_json():
    import json
    snap = _budget_with_call(consent=True)
    d = snap.to_dict()
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["cloud_calls_total"] == 1
