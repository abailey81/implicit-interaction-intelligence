"""Iter 60 — Privacy-budget circuit-breaker tests for the cloud arm.

Verifies that the iter-52/55 Gemini cascade arm honours the existing
PrivacyBudget invariants:

* No call without explicit user consent.
* No call once the per-session call budget is exhausted.
* No call once the per-session byte budget is exhausted.
* Reset clears the bucket cleanly.

These tests run on the budget module directly (no FastAPI / no
network) and complete in <0.1 s.
"""
from __future__ import annotations

import pytest

from i3.privacy.budget import PrivacyBudget


def test_default_consent_is_off():
    pb = PrivacyBudget()
    allowed, reason = pb.can_call("alice", "s1")
    assert allowed is False
    assert "consent" in reason.lower()


def test_consent_flip_unblocks():
    pb = PrivacyBudget()
    pb.set_consent("alice", True)
    allowed, _ = pb.can_call("alice", "s1")
    assert allowed is True


def test_call_budget_exhausts():
    pb = PrivacyBudget(max_cloud_calls_per_session=3,
                       max_bytes_per_session=10_000)
    pb.set_consent("alice", True)
    for _ in range(3):
        allowed, _ = pb.can_call("alice", "s1")
        assert allowed is True
        pb.record_call(
            "alice", "s1",
            sanitised_prompt="hi", response_text="ok",
            pii_redactions=0,
        )
    allowed, reason = pb.can_call("alice", "s1")
    assert allowed is False
    assert "call budget" in reason.lower()


def test_byte_budget_exhausts():
    pb = PrivacyBudget(max_cloud_calls_per_session=10,
                       max_bytes_per_session=20)  # tiny
    pb.set_consent("alice", True)
    pb.record_call(
        "alice", "s1",
        sanitised_prompt="x" * 25, response_text="y" * 25,
        pii_redactions=0,
    )
    allowed, reason = pb.can_call("alice", "s1")
    assert allowed is False
    assert "byte budget" in reason.lower()


def test_reset_session_clears_bucket():
    pb = PrivacyBudget(max_cloud_calls_per_session=2,
                       max_bytes_per_session=10_000)
    pb.set_consent("alice", True)
    pb.record_call("alice", "s1", sanitised_prompt="hi", response_text="ok",
                   pii_redactions=0)
    pb.record_call("alice", "s1", sanitised_prompt="hi", response_text="ok",
                   pii_redactions=0)
    allowed, _ = pb.can_call("alice", "s1")
    assert allowed is False
    pb.reset_session("alice", "s1")
    allowed, _ = pb.can_call("alice", "s1")
    assert allowed is True


def test_consent_per_user_isolation():
    pb = PrivacyBudget()
    pb.set_consent("alice", True)
    a_ok, _ = pb.can_call("alice", "s1")
    b_ok, b_reason = pb.can_call("bob", "s1")
    assert a_ok is True
    assert b_ok is False
    assert "consent" in b_reason.lower()


def test_snapshot_shape():
    pb = PrivacyBudget(max_cloud_calls_per_session=5,
                       max_bytes_per_session=1024)
    pb.set_consent("alice", True)
    pb.record_call("alice", "s1", sanitised_prompt="hello",
                   response_text="hi", pii_redactions=0)
    snap = pb.snapshot("alice", "s1")
    d = snap.to_dict() if hasattr(snap, "to_dict") else dict(snap)
    # Schema spot-check — these keys are what the WS layer ships
    for k in ("cloud_calls_total", "bytes_transmitted_total",
              "cloud_calls_max", "bytes_transmitted_max",
              "budget_remaining_calls", "budget_remaining_bytes"):
        assert k in d, f"missing {k!r} in snapshot.to_dict()"
    assert d["cloud_calls_total"] == 1
    assert d["budget_remaining_calls"] == 4
    assert d["consent_enabled"] is True
