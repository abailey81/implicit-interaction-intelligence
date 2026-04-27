"""Iter 102 — LinguisticAnalyzer feature-extraction invariants."""
from __future__ import annotations

import math

import pytest

from i3.interaction.linguistic import LinguisticAnalyzer


@pytest.fixture
def la():
    return LinguisticAnalyzer()


def test_type_token_ratio_unit_interval(la):
    for text in ["hello world", "the the the", "a", "", "complex sentence with vocabulary"]:
        v = la.type_token_ratio(text)
        assert 0.0 <= v <= 1.0


def test_type_token_ratio_repeats_low(la):
    repeat = la.type_token_ratio("hi hi hi hi hi")
    diverse = la.type_token_ratio("alpha beta gamma delta epsilon")
    assert repeat < diverse


def test_mean_word_length_positive(la):
    assert la.mean_word_length("hi there") > 0
    assert la.mean_word_length("") == 0


def test_count_syllables_basic(la):
    assert la.count_syllables("hello") >= 1
    assert la.count_syllables("syllabification") >= 4
    assert la.count_syllables("a") >= 1


def test_flesch_kincaid_grade_no_nan(la):
    for text in ["", "hi", "the quick brown fox jumps over the lazy dog",
                 "complicated multisyllabic profundity"]:
        g = la.flesch_kincaid_grade(text)
        assert isinstance(g, float)
        assert not math.isnan(g)


def test_question_ratio_in_unit(la):
    assert la.question_ratio("Hi. How are you?") in (0.5, pytest.approx(0.5))
    assert 0.0 <= la.question_ratio("hi") <= 1.0


def test_exclamation_ratio_in_unit(la):
    assert 0.0 <= la.exclamation_ratio("hi!") <= 1.0


def test_formality_score_in_unit(la):
    for text in ["yo whats up", "Greetings, may I inquire as to your wellbeing"]:
        v = la.formality_score(text)
        assert 0.0 <= v <= 1.0


def test_formality_higher_for_formal_text(la):
    informal = la.formality_score("yo whats up bro")
    formal = la.formality_score("Greetings; I trust this finds you well.")
    assert formal >= informal


def test_emoji_count(la):
    assert la.emoji_count("hi 😂") >= 1
    assert la.emoji_count("plain text") == 0


def test_sentiment_valence_in_signed_unit(la):
    for text in ["i love this", "i hate this", "neutral statement"]:
        v = la.sentiment_valence(text)
        assert -1.0 <= v <= 1.0


def test_sentence_split_basic(la):
    out = la.sentence_split("Hi. How are you? Fine!")
    assert isinstance(out, list)
    assert len(out) >= 2


def test_compute_all_returns_dict(la):
    feats = la.compute_all("Hello, how are you doing today?")
    assert isinstance(feats, dict)
    assert len(feats) >= 5
    for k, v in feats.items():
        assert isinstance(v, (int, float))
        assert not (math.isnan(v) if isinstance(v, float) else False)
