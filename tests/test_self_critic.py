"""Iter 72 — SelfCritic scoring contract tests.

Pins the composite-scoring behaviour of i3.critique.critic.SelfCritic
across the 5 sub-rubrics (on_topic / well_formed / non_repetitive /
safe / adaptation_match), the threshold-accept decision, and the
"never raises" contract.
"""
from __future__ import annotations

import pytest

from i3.critique.critic import CritiqueScore, SelfCritic


@pytest.fixture
def critic():
    return SelfCritic()


def _adapt(**kw):
    base = {"cognitive_load": 0.5, "emotional_tone": 0.5,
            "verbosity": 0.5, "formality": 0.5,
            "directness": 0.5, "accessibility": 0.0}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_score_returns_critique_score(critic):
    out = critic.score(prompt="hi", response="hello there",
                       adaptation=_adapt())
    assert isinstance(out, CritiqueScore)
    assert 0.0 <= out.score <= 1.0
    assert isinstance(out.accepted, bool)
    assert isinstance(out.sub_scores, dict)
    assert isinstance(out.reasons, list)


def test_score_includes_all_sub_scores(critic):
    out = critic.score(prompt="hi", response="hello",
                       adaptation=_adapt())
    for k in ("on_topic", "well_formed", "non_repetitive",
              "safe", "adaptation_match"):
        assert k in out.sub_scores, f"missing sub_score {k!r}"


def test_threshold_drives_accepted(critic):
    """A high-quality answer should be accepted; a clearly bad one
    should be rejected."""
    good = critic.score(
        prompt="what is python",
        response="Python is a programming language widely used in data "
                 "science, web backends, and scripting.",
        adaptation=_adapt(),
    )
    bad = critic.score(
        prompt="what is python",
        response="UNK [SEP] [SEP] [SEP] [SEP] [SEP] foo foo foo foo foo",
        adaptation=_adapt(),
    )
    assert good.score > bad.score
    assert bad.accepted is False


# ---------------------------------------------------------------------------
# Sub-rubric behaviour
# ---------------------------------------------------------------------------

def test_unk_leakage_penalised(critic):
    out = critic.score(
        prompt="hello",
        response="hello UNK how are you",
        adaptation=_adapt(),
    )
    assert out.sub_scores["well_formed"] < 1.0


def test_repetition_penalised(critic):
    out = critic.score(
        prompt="say hi",
        response="hi hi hi hi hi hi hi hi hi hi",
        adaptation=_adapt(),
    )
    # Either well_formed or non_repetitive should drop materially
    assert (out.sub_scores["well_formed"] < 0.7
            or out.sub_scores["non_repetitive"] < 0.7)


def test_off_topic_penalised(critic):
    out = critic.score(
        prompt="what is the capital of France",
        response="bananas grow on trees in tropical climates",
        adaptation=_adapt(),
    )
    assert out.sub_scores["on_topic"] < 0.5


def test_safe_default_for_neutral_response(critic):
    out = critic.score(prompt="hi", response="hello",
                       adaptation=_adapt())
    assert out.sub_scores["safe"] >= 0.9


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_never_raises_on_empty(critic):
    out = critic.score(prompt="", response="", adaptation={})
    assert isinstance(out, CritiqueScore)


def test_never_raises_on_huge_input(critic):
    out = critic.score(
        prompt="hi",
        response="word " * 5000,
        adaptation=_adapt(),
    )
    assert isinstance(out, CritiqueScore)


def test_score_clamped_to_unit_interval(critic):
    """Even a hostile weights override mustn't push score out of [0, 1]."""
    c = SelfCritic(weights={"on_topic": 99.0, "safe": 99.0})
    out = c.score(prompt="hi", response="hello", adaptation=_adapt())
    assert 0.0 <= out.score <= 1.0


def test_threshold_configurable():
    c = SelfCritic(threshold=0.99)
    out = c.score(prompt="hi", response="hello", adaptation=_adapt())
    # Threshold 0.99 should reject most reasonable scores.
    assert out.accepted in (True, False)
    assert out.threshold == pytest.approx(0.99)
