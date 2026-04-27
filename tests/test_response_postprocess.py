"""Iter 101 — ResponsePostProcessor invariant tests.

Pins the cloud-response postprocess pipeline that adapts cloud-LLM
output to the AdaptationVector (length / vocabulary / non-empty
guarantee).
"""
from __future__ import annotations

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.postprocess import ResponsePostProcessor


@pytest.fixture
def pp():
    return ResponsePostProcessor()


def _vec(load=0.5, access=0.0):
    return AdaptationVector(
        cognitive_load=load,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=access,
    )


def test_returns_non_empty_string(pp):
    out = pp.process("hello world", _vec())
    assert isinstance(out, str)
    assert out.strip()


def test_empty_response_falls_back(pp):
    out = pp.process("", _vec())
    assert out  # the EMPTY_FALLBACK is non-empty


def test_whitespace_only_falls_back(pp):
    out = pp.process("   \t\n  ", _vec())
    assert out.strip()


def test_high_cognitive_load_shortens_response(pp):
    """High cognitive load (>0.7) caps sentences to ~2."""
    long = ("Sentence one. Sentence two. Sentence three. "
            "Sentence four. Sentence five. Sentence six.")
    out = pp.process(long, _vec(load=0.9))
    n = sum(1 for s in out.split(".") if s.strip())
    assert n <= 4, f"high-load response has {n} sentences: {out!r}"


def test_low_cognitive_load_preserves_length(pp):
    """Low load shouldn't aggressively trim."""
    full = "First. Second. Third."
    out = pp.process(full, _vec(load=0.1))
    assert "First" in out
    assert "Third" in out


def test_high_accessibility_does_not_crash(pp):
    """Accessibility > 0.5 enables simplification — must not crash."""
    out = pp.process("This is a typical sentence.", _vec(access=0.9))
    assert isinstance(out, str)


def test_idempotent_on_already_processed(pp):
    """Running process() twice produces the same result the second
    time (no further trimming when the text already meets the spec)."""
    s = "short response."
    once = pp.process(s, _vec())
    twice = pp.process(once, _vec())
    assert once == twice


def test_handles_no_terminator(pp):
    """A bare phrase without . or ? should round-trip safely."""
    out = pp.process("hello there", _vec())
    assert "hello" in out.lower()
