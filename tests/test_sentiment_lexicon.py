"""Tests for the on-device valence lexicon used by the perception layer.

The lexicon lives at ``i3/interaction/data/sentiment_lexicon.json`` and
is consumed by :class:`i3.interaction.sentiment.ValenceLexicon`.  These
tests cover shape invariants (no duplicates, valences in [-1, 1]),
loader behaviour, negation handling, and calibration of a small
golden set so regressions on the lexicon itself are caught early.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LEXICON_PATH = Path(__file__).resolve().parents[1] / "i3" / "interaction" / "data" / "sentiment_lexicon.json"


# ---------------------------------------------------------------------------
# Shape invariants — every one of these catches a real class of regression.
# ---------------------------------------------------------------------------


def _load_raw() -> dict:
    return json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))


def test_lexicon_file_exists():
    assert _LEXICON_PATH.is_file(), f"lexicon not found at {_LEXICON_PATH}"


def test_lexicon_has_expected_sections():
    data = _load_raw()
    for section in ("__meta__", "positive", "negative"):
        assert section in data, f"missing section: {section}"


def test_lexicon_has_meaningful_coverage():
    """Guard against accidental shrinking of the lexicon."""
    data = _load_raw()
    assert len(data["positive"]) >= 200, (
        f"positive lexicon too small: {len(data['positive'])}"
    )
    assert len(data["negative"]) >= 200, (
        f"negative lexicon too small: {len(data['negative'])}"
    )


def test_lexicon_values_in_unit_interval():
    data = _load_raw()
    for k, v in data["positive"].items():
        assert isinstance(v, (int, float)), f"positive[{k}] is not numeric"
        assert 0 < v <= 1, f"positive[{k}]={v} must be in (0, 1]"
    for k, v in data["negative"].items():
        assert isinstance(v, (int, float)), f"negative[{k}] is not numeric"
        assert -1 <= v < 0, f"negative[{k}]={v} must be in [-1, 0)"


def test_lexicon_no_cross_section_duplicates():
    """A word must not appear as both positive and negative."""
    data = _load_raw()
    pos = set(data["positive"].keys())
    neg = set(data["negative"].keys())
    dupes = pos & neg
    assert not dupes, f"words present in both sections: {sorted(dupes)}"


def test_lexicon_no_whitespace_keys():
    data = _load_raw()
    for section in ("positive", "negative"):
        for key in data[section]:
            assert key == key.strip(), f"whitespace in key {key!r}"
            assert key == key.lower(), f"non-lowercase key {key!r}"


def test_lexicon_meta_citations_present():
    """The lexicon must cite the research it was inspired by."""
    meta = _load_raw()["__meta__"]
    assert "citations" in meta
    assert any("VADER" in c for c in meta["citations"])
    # NRC Emotion Lexicon is typically cited as "Mohammad & Turney" /
    # "Word-Emotion Association Lexicon" — accept either token.
    assert any(
        "Word-Emotion" in c or "Mohammad" in c or "NRC" in c
        for c in meta["citations"]
    )


# ---------------------------------------------------------------------------
# Calibration golden set — specific examples that should always hold.
# These are deliberately robust to small valence tweaks: each assertion
# is an ordering or sign check rather than a precise number.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lexicon():
    # Lazy import so the stub-torch trick used in the monkeypatched
    # test suite still works.
    from i3.interaction.sentiment import ValenceLexicon

    return ValenceLexicon.default()


def _score(lex, sentence: str) -> float:
    return lex.score(sentence.lower().split())


def test_positive_sentences_score_positive(lexicon):
    for sentence in (
        "I am happy",
        "this is wonderful",
        "lovely beautiful amazing",
        "the code works perfectly",
        "I love this",
    ):
        assert _score(lexicon, sentence) > 0, sentence


def test_negative_sentences_score_negative(lexicon):
    for sentence in (
        "I am tired",
        "this is terrible",
        "broken crashed failure",
        "I hate this bug",
        "the system is down",
    ):
        assert _score(lexicon, sentence) < 0, sentence


def test_negation_flips_polarity(lexicon):
    """"not bad" should register as positive; "not great" as negative."""
    assert _score(lexicon, "not bad") > 0
    assert _score(lexicon, "not great") < 0
    assert _score(lexicon, "never good") < 0


def test_intensity_is_non_negative(lexicon):
    """Intensity is |score|-like and must never be negative."""
    for sentence in (
        "hello world",
        "I am deeply sad",
        "the cat sat on the mat",
        "amazing",
    ):
        assert lexicon.intensity(sentence.lower().split()) >= 0.0


def test_neutral_sentences_near_zero(lexicon):
    """A sentence with no affective tokens should score ~0."""
    for sentence in (
        "the cat sat on the mat",
        "open the file",
        "3 plus 4 equals 7",
    ):
        s = _score(lexicon, sentence)
        assert abs(s) < 0.15, f"{sentence!r} scored {s} — expected ~0"


def test_stronger_words_score_more_strongly(lexicon):
    """A stronger affective term should outweigh a milder one."""
    # "terrible" is catastrophic; "bad" is just mildly negative.
    assert _score(lexicon, "terrible") <= _score(lexicon, "bad")
    # "amazing" is stronger than "nice".
    assert _score(lexicon, "amazing") >= _score(lexicon, "nice")


def test_hci_developer_vocabulary_covered(lexicon):
    """The lexicon was expanded with HCI / developer terms — confirm."""
    for token in (
        "works",
        "fixed",
        "broken",
        "stuck",
        "crashed",
        "flaky",
        "outage",
    ):
        assert token in lexicon, f"{token!r} missing from lexicon"


def test_informal_interjections_covered(lexicon):
    """Informal / interjection vocabulary is part of v1.1.0 coverage."""
    for token in ("yay", "woohoo", "ugh", "meh", "yuck"):
        assert token in lexicon, f"{token!r} missing from lexicon"


def test_score_is_deterministic(lexicon):
    """Repeated calls yield identical scores."""
    sentence = "this is a moderately positive example".split()
    s1 = lexicon.score(sentence)
    s2 = lexicon.score(sentence)
    assert s1 == s2
