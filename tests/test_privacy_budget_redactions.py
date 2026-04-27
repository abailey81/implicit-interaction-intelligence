"""Iter 113 — PrivacyBudget redaction + sensitive-category accumulation."""
from __future__ import annotations

import pytest

from i3.privacy.budget import PrivacyBudget


@pytest.fixture
def pb():
    p = PrivacyBudget(max_cloud_calls_per_session=10,
                      max_bytes_per_session=10_000)
    p.set_consent("alice", True)
    return p


def test_pii_redactions_total_aggregates(pb):
    pb.record_call("alice", "s1",
                   sanitised_prompt="hi", response_text="ok",
                   pii_redactions=2)
    pb.record_call("alice", "s1",
                   sanitised_prompt="hi", response_text="ok",
                   pii_redactions=3)
    snap = pb.snapshot("alice", "s1").to_dict()
    assert snap["pii_redactions_total"] == 5


def test_bytes_redacted_total_aggregates(pb):
    pb.record_call("alice", "s1",
                   sanitised_prompt="x", response_text="y",
                   pii_redactions=1, bytes_redacted=15)
    pb.record_call("alice", "s1",
                   sanitised_prompt="x", response_text="y",
                   pii_redactions=1, bytes_redacted=22)
    snap = pb.snapshot("alice", "s1").to_dict()
    assert snap["bytes_redacted_total"] == 37


def test_sensitive_categories_aggregate(pb):
    pb.record_call("alice", "s1",
                   sanitised_prompt="x", response_text="y",
                   pii_redactions=2,
                   pii_categories={"email": 1, "phone": 1})
    pb.record_call("alice", "s1",
                   sanitised_prompt="x", response_text="y",
                   pii_redactions=1,
                   pii_categories={"email": 1, "ssn": 1})
    snap = pb.snapshot("alice", "s1").to_dict()
    cats = snap["sensitive_categories"]
    assert cats.get("email") == 2
    assert cats.get("phone") == 1
    assert cats.get("ssn") == 1


def test_explicit_bytes_in_out_overrides_string_length(pb):
    pb.record_call("alice", "s1",
                   sanitised_prompt="x" * 100, response_text="y" * 100,
                   pii_redactions=0, bytes_in=10, bytes_out=20)
    snap = pb.snapshot("alice", "s1").to_dict()
    # Explicit bytes_in + bytes_out used, not the string lengths
    assert snap["bytes_transmitted_total"] == 30


def test_per_session_isolation():
    pb = PrivacyBudget(max_cloud_calls_per_session=10,
                       max_bytes_per_session=10_000)
    pb.set_consent("alice", True)
    pb.record_call("alice", "s1",
                   sanitised_prompt="hi", response_text="ok",
                   pii_redactions=2)
    snap_s2 = pb.snapshot("alice", "s2").to_dict()
    # Session s2 has no calls
    assert snap_s2["pii_redactions_total"] == 0
    assert snap_s2["cloud_calls_total"] == 0
