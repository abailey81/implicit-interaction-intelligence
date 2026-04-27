"""Iter 100 — Pipeline._dedupe_sentences regression tests.

Pins the iter-51 sentence-level Jaccard ≥ 0.6 dedupe behaviour that
catches retrieval-then-SLM and KG-overview duplications.
"""
from __future__ import annotations

import pytest

from i3.pipeline.engine import Pipeline


def _dedupe(text: str) -> str:
    return Pipeline._dedupe_sentences(text)


def test_short_input_unchanged():
    assert _dedupe("hi") == "hi"
    assert _dedupe("") == ""


def test_single_sentence_unchanged():
    s = "Python is a programming language."
    assert _dedupe(s) == s


def test_distinct_sentences_preserved():
    s = "Python is a language.  Rust is also a language but stricter."
    out = _dedupe(s)
    assert "Python" in out
    assert "Rust" in out


def test_year_overlap_dedupe_canonical_iter51_case():
    """The canonical iter-51 case: founded_by sentence already names
    the year, founded_in sentence repeats it — second must drop."""
    s = ("Python was founded by Guido van Rossum in 1991.  "
         "Python was founded in 1991.")
    out = _dedupe(s)
    # Year must appear at most once
    assert out.count("1991") <= 1
    # Founder must still be present
    assert "Guido" in out


def test_distinct_facts_about_same_subject_kept():
    """Two facts about Python with no token overlap must both
    survive (Jaccard < 0.6)."""
    s = ("Python was founded by Guido in 1991.  "
         "Python is famous for data science.")
    out = _dedupe(s)
    assert "Guido" in out
    assert "data science" in out


def test_dedupe_handles_no_terminators():
    """Input without any . ! ? returns unchanged."""
    s = "no terminators here just a long sentence without proper punctuation at all"
    out = _dedupe(s)
    assert out == s


def test_long_repeated_paragraph():
    base = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. "
    out = _dedupe(base * 1)
    # All three are identical → only one survives
    assert out.count("The cat sat on the mat") <= 1


def test_preserves_terminator_punctuation():
    s = "Hello world.  How are you?  I am fine!"
    out = _dedupe(s)
    # Terminators preserved (each sentence still has its punctuation)
    assert "?" in out
    assert "!" in out
