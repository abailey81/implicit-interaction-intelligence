"""Iter 83 — TopicSensitivityDetector per-category coverage.

Verifies the sensitivity router scores each category at the
documented weight, plus the iter-51 abuse-pattern fix
("sexually assaulted" / "raped" / "molested" inflections).
"""
from __future__ import annotations

import pytest

from i3.router.sensitivity import TopicSensitivityDetector


@pytest.fixture(scope="module")
def d():
    return TopicSensitivityDetector()


@pytest.mark.parametrize("text,min_score", [
    # confidential_info
    ("This is confidential", 0.85),
    ("under NDA agreement", 0.85),
    # abuse_safety (iter-51 inflection fix)
    ("I was sexually assaulted", 0.90),
    ("she was sexually abused", 0.90),
    ("he raped someone", 0.90),
    ("the suspect molested children", 0.90),
    ("domestic violence shelter", 0.90),
    # medical_records
    ("symptom of cancer", 0.85),
    ("MRI scan results", 0.85),
])
def test_detected_above_threshold(d, text, min_score):
    score = d.detect(text)
    assert score >= min_score, \
        f"text {text!r} scored {score:.2f}, expected >= {min_score}"


@pytest.mark.parametrize("text", [
    "hello there",
    "what is the weather",
    "set timer for 5 minutes",
    "explain photosynthesis",
])
def test_benign_text_at_min_score(d, text):
    score = d.detect(text)
    # Benign turns get the detector's min_score (the "no signal" default).
    assert score < 0.5


def test_empty_input(d):
    assert d.detect("") <= 0.5
    assert d.detect("   ") <= 0.5


def test_score_clamped_to_unit_interval(d):
    """Scores should always be in [min_score, 1.0]."""
    for t in ["confidential and assault and rape", "a" * 5000]:
        s = d.detect(t)
        assert 0.0 <= s <= 1.0


def test_detect_detailed_returns_categories(d):
    out = d.detect_detailed("she was sexually assaulted")
    assert isinstance(out, dict)
    assert "score" in out
    assert "matched_categories" in out
    assert "abuse_safety" in out["matched_categories"], \
        f"expected abuse_safety in matched, got {out['matched_categories']}"
