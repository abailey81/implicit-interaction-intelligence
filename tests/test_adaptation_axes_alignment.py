"""Iter 47-48, 53-55 regression tests — adaptation axes alignment.

These tests pin the post-iter-43 unified semantic across the
adaptation pipeline:

* CognitiveLoadAdapter / prompt-builder / post-processor all agree
  that high cl => stressed => brief reply (iter 53).
* AccessibilityAdapter fires from any single strong difficulty
  signal (iter 47-48 max() + 0.5 threshold).
* Accessibility forces a 1-sentence cap regardless of cl tier
  (iter 54).
* Cloud prompt-builder and post-processor use the same threshold
  bands for verbosity / formality (iter 55).

If a future refactor re-introduces a semantic inversion or threshold
mismatch, these tests fire immediately.
"""

from __future__ import annotations

import pytest

from i3.adaptation.controller import AdaptationController
from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.postprocess import ResponsePostProcessor
from i3.cloud.prompt_builder import PromptBuilder
from i3.config import AdaptationConfig
from i3.interaction.types import InteractionFeatureVector
from i3.user_model.types import DeviationMetrics


def _zero_dev() -> DeviationMetrics:
    return DeviationMetrics(
        current_vs_baseline=0.0, current_vs_session=0.0,
        engagement_score=0.5, magnitude=0.0,
        iki_deviation=0.0, length_deviation=0.0, vocab_deviation=0.0,
        formality_deviation=0.0, speed_deviation=0.0,
        engagement_deviation=0.0, complexity_deviation=0.0,
        pattern_deviation=0.0,
    )


def _fv(**overrides) -> InteractionFeatureVector:
    base = {n: 0.0 for n in [
        "mean_iki", "std_iki", "mean_burst_length", "mean_pause_duration",
        "backspace_ratio", "composition_speed", "pause_before_send",
        "editing_effort", "message_length", "type_token_ratio",
        "mean_word_length", "flesch_kincaid", "question_ratio",
        "formality", "emoji_density", "sentiment_valence",
        "length_trend", "latency_trend", "vocab_trend",
        "engagement_velocity", "topic_coherence", "session_progress",
        "time_deviation", "response_depth",
        "iki_deviation", "length_deviation", "vocab_deviation",
        "formality_deviation", "speed_deviation", "engagement_deviation",
        "complexity_deviation", "pattern_deviation",
    ]}
    base.update(overrides)
    return InteractionFeatureVector(**base)


# ---------------------------------------------------------------------------
# Iter 47-48 — accessibility fires from any single strong signal
# ---------------------------------------------------------------------------


@pytest.fixture
def ctrl() -> AdaptationController:
    return AdaptationController(AdaptationConfig())


def test_accessibility_fires_from_editing_effort_alone(ctrl) -> None:
    """A user with high editing_effort but no other difficulty signals
    must fire accessibility — the iter 48 max() aggregator means a
    single strong signal is sufficient evidence."""
    av = ctrl.compute(_fv(editing_effort=0.8), _zero_dev())
    assert av.accessibility >= 0.5, (
        f"editing_effort=0.8 alone should fire accessibility; "
        f"got {av.accessibility}"
    )


def test_accessibility_fires_from_backspace_ratio_alone(ctrl) -> None:
    """High backspace_ratio alone fires accessibility (iter 48)."""
    av = ctrl.compute(_fv(backspace_ratio=0.6), _zero_dev())
    assert av.accessibility >= 0.5, (
        f"backspace_ratio=0.6 alone should fire accessibility; "
        f"got {av.accessibility}"
    )


def test_accessibility_does_not_fire_on_mild_signals(ctrl) -> None:
    """Mild stress signals (each well below 0.5) must NOT fire
    accessibility — the iter 47 threshold is 0.5, not free-for-all."""
    av = ctrl.compute(
        _fv(editing_effort=0.3, backspace_ratio=0.2),
        _zero_dev(),
    )
    assert av.accessibility == 0.0, (
        f"mild stress (edit=0.3, bsr=0.2) should not fire accessibility; "
        f"got {av.accessibility}"
    )


# ---------------------------------------------------------------------------
# Iter 53 — cognitive_load semantic alignment between prompt + post-processor
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


def _av(cl: float = 0.5, **kw) -> AdaptationVector:
    return AdaptationVector(
        cognitive_load=cl,
        style_mirror=StyleVector(
            formality=kw.get("formality", 0.5),
            verbosity=kw.get("verbosity", 0.5),
            emotionality=kw.get("emotionality", 0.5),
            directness=kw.get("directness", 0.5),
        ),
        emotional_tone=kw.get("emotional_tone", 0.5),
        accessibility=kw.get("accessibility", 0.0),
    )


def test_high_cl_prompt_asks_for_brief_reply(builder) -> None:
    """Iter 53: high cognitive_load means STRESSED.  The prompt-
    builder must instruct the LLM to keep replies tight, not to
    'use sophisticated vocabulary'."""
    prompt = builder.build_system_prompt(_av(cl=0.9))
    p_lower = prompt.lower()
    # Must NOT ask for richness on a stressed user.
    assert "sophisticated" not in p_lower, (
        f"high cl should not ask for sophisticated vocab; got {prompt}"
    )
    # Must ask for a short / single-sentence reply.
    assert "single concise sentence" in p_lower or "single sentence" in p_lower


def test_low_cl_prompt_allows_richer_reply(builder) -> None:
    """Iter 53: low cognitive_load means user has spare bandwidth —
    the prompt may ask for 4-6 sentence depth."""
    prompt = builder.build_system_prompt(_av(cl=0.1))
    p_lower = prompt.lower()
    assert "spare cognitive bandwidth" in p_lower or "richer" in p_lower


# ---------------------------------------------------------------------------
# Iter 54 — accessibility forces 1-sentence cap regardless of cl
# ---------------------------------------------------------------------------


@pytest.fixture
def pp() -> ResponsePostProcessor:
    return ResponsePostProcessor()


def test_accessibility_user_gets_one_sentence_at_moderate_cl(pp) -> None:
    """Iter 54: a user whose accessibility fires gets a 1-sentence
    reply even when cognitive_load is moderate.  The LLM is asked to
    produce <= 15 words, but the post-processor enforces it
    deterministically."""
    raw = ("Sure! Here is a detailed response. The first thing to note is "
           "that this is complex. Furthermore, you should consider the "
           "implications. Finally, take your time.")
    shaped, log = pp.adapt_with_log(raw, _av(cl=0.4, accessibility=0.7))
    sentence_count = shaped.count(".") + shaped.count("!") + shaped.count("?")
    assert sentence_count <= 1, (
        f"accessibility=0.7 + cl=0.4 should produce <= 1 sentence; "
        f"got {sentence_count} sentences in {shaped!r}"
    )


def test_normal_user_at_moderate_cl_keeps_multiple_sentences(pp) -> None:
    """The iter 54 accessibility override must not affect normal users
    at the same cl level."""
    raw = ("Sure! Here is a detailed response. The first thing to note is "
           "that this is complex. Furthermore, you should consider the "
           "implications. Finally, take your time.")
    shaped, _ = pp.adapt_with_log(raw, _av(cl=0.4, accessibility=0.0))
    sentence_count = shaped.count(".") + shaped.count("!") + shaped.count("?")
    assert sentence_count >= 3, (
        f"accessibility=0 + cl=0.4 should preserve multiple sentences; "
        f"got {sentence_count} sentences in {shaped!r}"
    )


# ---------------------------------------------------------------------------
# Iter 55 — verbosity / formality threshold alignment between prompt + pp
# ---------------------------------------------------------------------------


def test_verbosity_threshold_alignment_low(builder, pp) -> None:
    """Iter 55: when verbosity is just below 0.35 (post-processor's
    hedge-strip threshold), the prompt-builder must ALSO ask the LLM
    to be concise — otherwise the LLM hedges and the post-processor
    silently strips, wasting tokens."""
    prompt = builder.build_system_prompt(_av(verbosity=0.32))
    assert "concise" in prompt.lower() or "skip hedges" in prompt.lower()


def test_formality_threshold_alignment_high(builder) -> None:
    """Iter 55: prompt-builder formality > 0.65 matches the post-
    processor's contraction-expansion threshold."""
    # Just above threshold — must ask for formal tone.
    prompt = builder.build_system_prompt(_av(formality=0.66))
    assert "professional" in prompt.lower() or "formal" in prompt.lower()
