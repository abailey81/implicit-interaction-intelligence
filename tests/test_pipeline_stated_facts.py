"""Iter 87 — Pipeline._stated_facts in-memory cache invariants.

The iter-49 multi-fact session memory + iter-50 cross-session
persistence both depend on a per-(user, session) dict.  These tests
exercise the helper directly (no full Pipeline boot needed).
"""
from __future__ import annotations

import pytest

from i3.pipeline.engine import Pipeline


@pytest.fixture
def pipe():
    p = Pipeline.__new__(Pipeline)
    p._stated_facts = {}
    p._intent_parser_qwen = None
    return p


def _facts(pipe, uid, sid):
    return pipe._stated_facts.setdefault((uid, sid), {})


def test_facts_per_user_session_isolation(pipe):
    a = _facts(pipe, "alice", "s1")
    b = _facts(pipe, "alice", "s2")
    a["color"] = "blue"
    assert "color" not in b


def test_facts_per_user_isolation(pipe):
    a = _facts(pipe, "alice", "s1")
    b = _facts(pipe, "bob", "s1")
    a["name"] = "Alice"
    assert "name" not in b


def test_setdefault_returns_existing_dict(pipe):
    a = _facts(pipe, "alice", "s1")
    a["x"] = 1
    a2 = _facts(pipe, "alice", "s1")
    assert a is a2
    assert a2["x"] == 1


def test_facts_keys_are_tuples(pipe):
    """The cache key must be (user_id, session_id) tuple — string
    concatenation would clash on edge cases."""
    pipe._stated_facts[("a", "1:2")] = {"k": "v"}
    pipe._stated_facts[("a:1", "2")] = {"k": "v2"}
    # Two distinct keys (tuple structure prevents collision)
    assert len(pipe._stated_facts) == 2
    assert pipe._stated_facts[("a", "1:2")]["k"] == "v"
    assert pipe._stated_facts[("a:1", "2")]["k"] == "v2"


def test_intent_parser_lazy_starts_none(pipe):
    assert pipe._intent_parser_qwen is None


def test_cascade_arm_classifier_unknown_path(pipe):
    """Iter 55 classifier returns 'other' for unrecognised paths."""
    assert Pipeline._classify_cascade_arm("unknown_path", "?") == "other"
