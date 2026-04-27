"""Iter 112 — explain_decomposer pattern + topic-keyword tests."""
from __future__ import annotations

import pytest

from i3.pipeline.explain_decomposer import (
    _EXPLAIN_PATTERNS,
    _topic_keywords,
)


def _matches(text: str) -> str | None:
    for pat in _EXPLAIN_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).strip().lower()
    return None


@pytest.mark.parametrize("text,expected_topic", [
    ("explain photosynthesis", "photosynthesis"),
    ("Explain attention mechanisms.", "attention mechanisms"),
    ("tell me about transformers", "transformers"),
    ("describe gradient descent", "gradient descent"),
    ("how does backprop work", "backprop"),
    ("what is recursion?", "recursion"),
    ("walk me through the kalman filter", "the kalman filter"),
])
def test_explain_pattern_matches(text, expected_topic):
    out = _matches(text)
    assert out == expected_topic, f"{text!r} -> {out!r} (want {expected_topic!r})"


@pytest.mark.parametrize("text", [
    "hello",
    "set timer for 5 minutes",
    "thanks",
    "yes",
    "no",
    "i love mondays",
])
def test_non_explain_text_does_not_match(text):
    out = _matches(text)
    assert out is None, f"non-explain text matched: {text!r} -> {out!r}"


def test_topic_keywords_strips_stopwords():
    kws = _topic_keywords("the the explanation of attention")
    assert "explanation" in kws
    assert "attention" in kws
    # 'the' / 'of' / 'explain' are stopwords
    assert "the" not in kws
    assert "of" not in kws


def test_topic_keywords_returns_set():
    out = _topic_keywords("hello world")
    assert isinstance(out, set)


def test_topic_keywords_handles_empty():
    assert _topic_keywords("") == set()
    assert _topic_keywords("   ") == set()


def test_topic_keywords_lowercases():
    out = _topic_keywords("Photosynthesis And Plants")
    # All-uppercase tokens stripped of stopwords; survivors lowercased
    assert "photosynthesis" in out
    assert "plants" in out
    # case-insensitive
    assert "Photosynthesis" not in out
