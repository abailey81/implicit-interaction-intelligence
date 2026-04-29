"""Iter 37/38 — comprehensive response-shaping emulation.

The user's complaint was that "the answers the model gives must
actually be shaped, and adapted".  Iter 34-36 fixed the input side
(adaptation vector dynamic range); iter 37 wired the output side
(directness + emotional_tone + emotionality now actually shape the
reply).

These tests prove the whole loop works: identical raw replies
produce VISIBLY DIFFERENT post-processed text across user states,
with the right axes firing for each scenario.
"""

from __future__ import annotations

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.postprocess import ResponsePostProcessor


@pytest.fixture
def pp() -> ResponsePostProcessor:
    return ResponsePostProcessor()


def _av(*, cl=0.5, formality=0.5, verbosity=0.5, emotionality=0.5,
        directness=0.5, emotional_tone=0.5, accessibility=0.0) -> AdaptationVector:
    return AdaptationVector(
        cognitive_load=cl,
        style_mirror=StyleVector(
            formality=formality,
            verbosity=verbosity,
            emotionality=emotionality,
            directness=directness,
        ),
        emotional_tone=emotional_tone,
        accessibility=accessibility,
    )


# ---------------------------------------------------------------------------
# Per-axis shaping
# ---------------------------------------------------------------------------


def test_cognitive_load_high_trims_to_one_sentence(pp) -> None:
    reply = ("This is the first sentence. Here is the second one. "
             "And a third sentence rounds it out.")
    out, log = pp.adapt_with_log(reply, _av(cl=0.85))
    # Should trim to a single sentence on high cognitive load.
    assert out.count(".") == 1
    assert any(e["axis"] == "cognitive_load" for e in log)


def test_cognitive_load_low_preserves_all_sentences(pp) -> None:
    reply = ("First. Second. Third. Fourth. Fifth.")
    out, log = pp.adapt_with_log(reply, _av(cl=0.10))
    # Low load -> 6-sentence cap, all preserved.
    assert out == reply
    assert not any(e["axis"] == "cognitive_load" for e in log)


def test_verbosity_low_strips_hedges(pp) -> None:
    reply = "I think this is, perhaps, somewhat important to consider."
    out, log = pp.adapt_with_log(reply, _av(verbosity=0.20))
    assert "I think" not in out
    assert "perhaps" not in out
    assert any(e["axis"] == "verbosity" for e in log)


def test_verbosity_high_appends_followup(pp) -> None:
    reply = "This is the answer."
    out, log = pp.adapt_with_log(reply, _av(verbosity=0.80))
    assert len(out) > len(reply)
    assert any(e["axis"] == "verbosity" for e in log)


def test_formality_high_expands_contractions(pp) -> None:
    reply = "I'm sure it's a great idea."
    out, log = pp.adapt_with_log(reply, _av(formality=0.80))
    assert "I'm" not in out and "it's" not in out
    assert "I am" in out and "it is" in out
    assert any(e["axis"] == "formality" for e in log)


def test_formality_low_contracts(pp) -> None:
    reply = "I am sure that it is a wonderful day."
    out, log = pp.adapt_with_log(reply, _av(formality=0.20))
    # The contraction map applies; "I am" -> "I'm", "it is" -> "it's".
    assert "I'm" in out or "it's" in out
    assert any(e["axis"] == "formality" for e in log)


def test_accessibility_simplifies_vocabulary(pp) -> None:
    reply = ("I will utilize this approach to facilitate understanding. "
             "Subsequently, we can demonstrate the effect.")
    out, log = pp.adapt_with_log(reply, _av(accessibility=0.7))
    assert "utilize" not in out and "facilitate" not in out
    assert "use" in out and "help" in out
    assert any(e["axis"] == "accessibility" for e in log)


# ---------------------------------------------------------------------------
# Iter 37 — directness, emotional_tone, emotionality
# ---------------------------------------------------------------------------


def test_directness_high_strips_soft_openers(pp) -> None:
    reply = "You might want to consider running the build before you push."
    out, log = pp.adapt_with_log(reply, _av(directness=0.85))
    assert "you might want to consider" not in out.lower()
    assert any(e["axis"] == "directness" for e in log)


def test_directness_low_preserves_softeners(pp) -> None:
    reply = "You might want to consider running the build before you push."
    out, log = pp.adapt_with_log(reply, _av(directness=0.30))
    assert out == reply
    assert not any(e["axis"] == "directness" for e in log)


def test_emotional_tone_high_strips_warmth(pp) -> None:
    reply = "Sure! Happy to help. The answer is 42!!"
    out, log = pp.adapt_with_log(reply, _av(emotional_tone=0.85))
    assert "Sure" not in out and "Happy to help" not in out
    assert "!!" not in out and "!" not in out
    assert any(e["axis"] == "emotional_tone" for e in log)


def test_emotional_tone_low_preserves_warmth(pp) -> None:
    reply = "Sure! Happy to help. The answer is 42!"
    out, log = pp.adapt_with_log(reply, _av(emotional_tone=0.20))
    assert out == reply
    assert not any(e["axis"] == "emotional_tone" for e in log)


def test_emotionality_low_strips_intensifiers(pp) -> None:
    reply = "This is absolutely incredibly the most amazingly fun project."
    out, log = pp.adapt_with_log(reply, _av(emotionality=0.20))
    assert "absolutely" not in out.lower()
    assert "incredibly" not in out.lower()
    assert "amazingly" not in out.lower()
    assert any(e["axis"] == "emotionality" for e in log)


def test_emotionality_high_preserves_intensifiers(pp) -> None:
    reply = "This is absolutely incredibly amazing."
    out, log = pp.adapt_with_log(reply, _av(emotionality=0.85))
    assert out == reply
    assert not any(e["axis"] == "emotionality" for e in log)


# ---------------------------------------------------------------------------
# End-to-end: same raw reply → meaningfully different output across user
# states.  Two scenarios should have BOTH different log entries AND
# different output text.
# ---------------------------------------------------------------------------


@pytest.fixture
def reply() -> str:
    return (
        "Sure! Absolutely incredibly happy to help. "
        "You might want to consider that perhaps multiple perspectives "
        "could provide additional context. "
        "Furthermore, I should mention that this is amazingly complex!"
    )


def test_calm_user_gets_full_unmodified_reply(pp, reply) -> None:
    out, log = pp.adapt_with_log(
        reply,
        _av(cl=0.15, verbosity=0.50, formality=0.50, emotionality=0.50,
            directness=0.50, emotional_tone=0.50, accessibility=0.0),
    )
    assert out == reply
    assert log == []


def test_stressed_user_gets_short_simplified_reply(pp, reply) -> None:
    # Use cognitive_load=0.65 so we get 2 sentences, leaving room for
    # other axes to also fire.
    out, log = pp.adapt_with_log(
        reply,
        _av(cl=0.65, verbosity=0.20, formality=0.40, emotionality=0.40,
            directness=0.40, emotional_tone=0.30, accessibility=0.6),
    )
    # Trimmed to ≤ 2 sentences.
    assert out.count(".") + out.count("!") <= 2
    # Visibly shorter.
    assert len(out) < len(reply) * 0.7
    # cognitive_load definitely fired; verbosity should also fire
    # because the trimmed text still contains hedges.
    fired = {e["axis"] for e in log}
    assert "cognitive_load" in fired


def test_direct_neutral_unemotional_user_gets_clinical_reply(pp, reply) -> None:
    out, log = pp.adapt_with_log(
        reply,
        _av(cl=0.40, verbosity=0.50, formality=0.55, emotionality=0.20,
            directness=0.85, emotional_tone=0.85, accessibility=0.0),
    )
    # Soft openers stripped, exclamation neutralised, intensifiers removed.
    assert "you might want to consider" not in out.lower()
    assert "Sure!" not in out and "absolutely" not in out.lower()
    assert "!" not in out
    fired = {e["axis"] for e in log}
    assert fired >= {"directness", "emotional_tone", "emotionality"}


def test_warm_expressive_indirect_user_keeps_full_warmth(pp, reply) -> None:
    out, log = pp.adapt_with_log(
        reply,
        _av(cl=0.30, verbosity=0.55, formality=0.40, emotionality=0.85,
            directness=0.30, emotional_tone=0.20, accessibility=0.0),
    )
    # Nothing stripped on this profile.
    assert "Sure" in out
    assert "Absolutely" in out
    assert "you might want to consider" in out.lower()


def test_formal_direct_verbose_user_gets_followup_added(pp, reply) -> None:
    out, log = pp.adapt_with_log(
        reply,
        _av(cl=0.50, verbosity=0.85, formality=0.85, emotionality=0.50,
            directness=0.85, emotional_tone=0.50, accessibility=0.0),
    )
    # verbosity high -> followup appended.
    assert any(e["axis"] == "verbosity" and "appended" in e["change"]
               for e in log)
    # directness high -> softener stripped.
    assert "you might want to consider" not in out.lower()


# ---------------------------------------------------------------------------
# Cross-state determinism + invariants
# ---------------------------------------------------------------------------


def test_same_input_same_output(pp, reply) -> None:
    """Two runs with the same adaptation produce byte-identical text."""
    av = _av(cl=0.85, verbosity=0.20, formality=0.40, emotionality=0.40,
             directness=0.85, emotional_tone=0.85, accessibility=0.6)
    a, _ = pp.adapt_with_log(reply, av)
    b, _ = pp.adapt_with_log(reply, av)
    assert a == b


def test_output_always_starts_with_capital(pp) -> None:
    """Strip rules can leave a lowercase first letter; the final
    capitalisation pass must catch every case."""
    test_inputs = [
        "Sure! you might want to consider this.",
        "Absolutely you might want to consider this.",
        "Wonderful! perhaps you should think about it.",
        "Awesome you might consider this.",
    ]
    av = _av(directness=0.85, emotional_tone=0.85, emotionality=0.20)
    for raw in test_inputs:
        out, _ = pp.adapt_with_log(raw, av)
        assert out and out[0].isupper(), (
            f"output starts with lowercase: {out!r} (from {raw!r})"
        )


def test_output_is_non_empty_under_aggressive_stripping(pp) -> None:
    """Even when every axis fires aggressively, output is non-empty."""
    av = _av(cl=0.95, verbosity=0.10, formality=0.85, emotionality=0.10,
             directness=0.95, emotional_tone=0.95, accessibility=0.95)
    out, _ = pp.adapt_with_log("Sure! Absolutely incredibly happy to help!", av)
    assert out
    assert len(out) >= 3


def test_log_axes_are_strings(pp, reply) -> None:
    """Every log entry has a string axis name."""
    av = _av(cl=0.85, verbosity=0.20, directness=0.85,
             emotional_tone=0.85, emotionality=0.20)
    _, log = pp.adapt_with_log(reply, av)
    for e in log:
        assert isinstance(e["axis"], str)
        assert isinstance(e["change"], str)


# ---------------------------------------------------------------------------
# Spread test — measure dynamic range across many synthetic user states.
# ---------------------------------------------------------------------------


def test_many_user_states_produce_distinct_outputs(pp, reply) -> None:
    """20 user states sampled across the adaptation cube → at least
    8 distinct output texts.  Demonstrates the system actually
    differentiates across the user-state space."""
    import itertools

    seen = set()
    for cl, verbosity, directness, tone in itertools.product(
        (0.20, 0.55, 0.85),
        (0.20, 0.50, 0.80),
        (0.30, 0.60, 0.85),
        (0.20, 0.50, 0.85),
    ):
        av = _av(cl=cl, verbosity=verbosity, directness=directness,
                 emotional_tone=tone, accessibility=0.6 if cl > 0.7 else 0.0)
        out, _ = pp.adapt_with_log(reply, av)
        seen.add(out)

    # At least 8 distinct outputs across 81 combinations.
    assert len(seen) >= 8, (
        f"only {len(seen)} distinct outputs across 81 user states — "
        f"the post-processor isn't sufficiently differentiating"
    )
