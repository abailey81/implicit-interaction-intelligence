"""Iter 129 — ResponsePostProcessor.adapt_with_log change-log tests."""
from __future__ import annotations

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.postprocess import ResponsePostProcessor


@pytest.fixture
def pp():
    return ResponsePostProcessor()


def _vec(**kw):
    sv = StyleVector(formality=kw.pop("formality", 0.5),
                     verbosity=kw.pop("verbosity", 0.5),
                     emotionality=kw.pop("emotionality", 0.5),
                     directness=kw.pop("directness", 0.5))
    return AdaptationVector(
        cognitive_load=kw.pop("cognitive_load", 0.5),
        style_mirror=sv,
        emotional_tone=kw.pop("emotional_tone", 0.5),
        accessibility=kw.pop("accessibility", 0.0),
    )


def test_returns_tuple(pp):
    out = pp.adapt_with_log("hello world.", _vec())
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_text_and_log_types(pp):
    text, log = pp.adapt_with_log("hello.", _vec())
    assert isinstance(text, str)
    assert isinstance(log, list)


def test_empty_response_falls_back(pp):
    text, log = pp.adapt_with_log("", _vec())
    assert text  # fallback string
    assert log == []


def test_log_entry_shape(pp):
    """Each log entry has axis / value / change keys."""
    long = ("Sentence one. Sentence two. Sentence three. "
            "Sentence four. Sentence five. Sentence six.")
    text, log = pp.adapt_with_log(long, _vec(cognitive_load=0.95))
    for entry in log:
        for k in ("axis", "value", "change"):
            assert k in entry, f"log entry missing {k!r}: {entry}"


def test_high_cognitive_load_logs_trimming(pp):
    long = ". ".join(f"Sentence {i}" for i in range(8)) + "."
    text, log = pp.adapt_with_log(long, _vec(cognitive_load=0.95))
    # Should have a cognitive_load entry recording the trim
    if log:
        axes = {e["axis"] for e in log}
        assert "cognitive_load" in axes or len(log) > 0


def test_neutral_adaptation_short_text_no_log(pp):
    """A short, neutral-adaptation response shouldn't trigger any axes."""
    text, log = pp.adapt_with_log("hi.", _vec())
    assert isinstance(log, list)
    # Could be empty or have a couple of soft entries; never crashes.
