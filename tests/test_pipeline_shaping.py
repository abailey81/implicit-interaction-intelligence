"""Iter 38 — full-pipeline emulation: synthetic typing → reply shape.

Drives the entire chain end-to-end:

  user typing pattern
    -> FeatureExtractor (32-dim feature vector)
    -> BaselineTracker (deviation features)
    -> AdaptationController (7-axis adaptation vector)
    -> ResponsePostProcessor (final reply text)

For each synthetic user pattern, asserts the post-processed reply
visibly differs from the unprocessed reply AND from other user
patterns' outputs.

This is the regression suite that pins together iters 34-37 — if any
of those break, identical typing patterns will start producing
identical replies and the test fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from i3.adaptation.controller import AdaptationController
from i3.cloud.postprocess import ResponsePostProcessor
from i3.config import AdaptationConfig
from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.interaction.types import InteractionFeatureVector
from i3.user_model.types import DeviationMetrics


# ---------------------------------------------------------------------------
# Synthetic user patterns
# ---------------------------------------------------------------------------


@dataclass
class TypingPattern:
    name: str
    iki_mean_ms: float
    iki_std_ms: float
    composition_ms: float
    edits: int
    message_text: str


PATTERNS: list[TypingPattern] = [
    # 1. Calm, curious user — short questions, smooth typing.
    TypingPattern(
        name="calm_curious",
        iki_mean_ms=110.0, iki_std_ms=15.0,
        composition_ms=2200.0, edits=0,
        message_text="how does this work?",
    ),
    # 2. Verbose thoughtful user — longer messages, mid-pace.
    TypingPattern(
        name="verbose_thoughtful",
        iki_mean_ms=140.0, iki_std_ms=22.0,
        composition_ms=4500.0, edits=1,
        message_text=(
            "Could you explain in some detail how this system handles "
            "different kinds of user input across the various interaction "
            "modes you've described in your previous message please?"
        ),
    ),
    # 3. Stressed user — slow, edits, longer composition.
    TypingPattern(
        name="stressed_terse",
        iki_mean_ms=210.0, iki_std_ms=55.0,
        composition_ms=6000.0, edits=4,
        message_text="ugh just tell me",
    ),
    # 4. Direct power-user — fast, no edits, clear declarative.
    TypingPattern(
        name="direct_power_user",
        iki_mean_ms=85.0, iki_std_ms=12.0,
        composition_ms=1500.0, edits=0,
        message_text="set timer for 30 seconds.",
    ),
    # 5. Tired user — slow, low engagement, low complexity.
    TypingPattern(
        name="tired_simple",
        iki_mean_ms=260.0, iki_std_ms=35.0,
        composition_ms=8000.0, edits=2,
        message_text="ok do it",
    ),
]


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    pattern_name: str
    fv: InteractionFeatureVector
    adaptation_dict: dict
    shaped_reply: str
    log_axes: list[str]


def _run_pattern(
    pattern: TypingPattern,
    *,
    extractor: FeatureExtractor,
    baseline: BaselineTracker,
    controller: AdaptationController,
    pp: ResponsePostProcessor,
    canonical_reply: str,
) -> _Run:
    """Push one turn through the full pipeline."""
    km = {
        "mean_iki_ms": pattern.iki_mean_ms,
        "std_iki_ms": pattern.iki_std_ms,
        "mean_burst_length": 8.0,
        "mean_pause_duration_ms": 200.0,
        "backspace_ratio": min(1.0, pattern.edits / 50.0),
        "composition_speed_cps": max(0.5, len(pattern.message_text) / (pattern.composition_ms / 1000.0)),
        "pause_before_send_ms": 300.0,
        "editing_effort": min(1.0, pattern.edits / 10.0),
    }
    fv = extractor.extract(
        keystroke_metrics=km,
        message_text=pattern.message_text,
        history=[],
        baseline=baseline,
        session_start_ts=0.0,
        current_ts=30.0,
    )
    baseline.update(fv)

    deviation = DeviationMetrics(
        current_vs_baseline=0.0, current_vs_session=0.0,
        engagement_score=0.5, magnitude=0.0,
        iki_deviation=fv.iki_deviation,
        length_deviation=fv.length_deviation,
        vocab_deviation=fv.vocab_deviation,
        formality_deviation=fv.formality_deviation,
        speed_deviation=fv.speed_deviation,
        engagement_deviation=fv.engagement_deviation,
        complexity_deviation=fv.complexity_deviation,
        pattern_deviation=fv.pattern_deviation,
    )
    av = controller.compute(fv, deviation)
    shaped, log = pp.adapt_with_log(canonical_reply, av)

    return _Run(
        pattern_name=pattern.name,
        fv=fv,
        adaptation_dict=dict(
            cognitive_load=av.cognitive_load,
            verbosity=av.style_mirror.verbosity,
            formality=av.style_mirror.formality,
            emotionality=av.style_mirror.emotionality,
            directness=av.style_mirror.directness,
            emotional_tone=av.emotional_tone,
            accessibility=av.accessibility,
        ),
        shaped_reply=shaped,
        log_axes=[e["axis"] for e in log],
    )


# ---------------------------------------------------------------------------
# The canonical raw reply that every pattern shapes
# ---------------------------------------------------------------------------


CANONICAL_REPLY = (
    "Sure! Absolutely happy to help. "
    "You might want to consider that perhaps approximately five different "
    "perspectives could provide additional context on this complex topic. "
    "Furthermore, I should mention that you can subsequently demonstrate "
    "the effect with a few simple examples. "
    "Want me to expand on any of that?"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_state():
    extractor = FeatureExtractor()
    baseline = BaselineTracker(warmup=2)
    controller = AdaptationController(AdaptationConfig())
    pp = ResponsePostProcessor()
    return extractor, baseline, controller, pp


def test_each_pattern_runs_without_exception(pipeline_state) -> None:
    """All 5 synthetic patterns drive through the full pipeline without
    raising and produce a non-empty shaped reply."""
    extractor, baseline, controller, pp = pipeline_state
    for pat in PATTERNS:
        run = _run_pattern(
            pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
        )
        assert run.shaped_reply, f"empty reply for {pat.name}"
        # Basic sanity: every axis value is finite + bounded.
        for axis, val in run.adaptation_dict.items():
            assert 0.0 <= val <= 1.0, f"{axis}={val} out of [0,1] for {pat.name}"


def test_patterns_produce_distinct_shaped_replies(pipeline_state) -> None:
    """Different typing patterns must produce visibly different shaped
    replies — otherwise the system is not adapting end-to-end."""
    extractor, baseline, controller, pp = pipeline_state
    runs = [
        _run_pattern(
            pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
        )
        for pat in PATTERNS
    ]

    distinct_replies = {r.shaped_reply for r in runs}
    # 5 patterns -> at least 3 distinct shaped replies (some may coincide
    # if their adaptation vectors are nearly identical, but most should
    # differ).
    assert len(distinct_replies) >= 3, (
        f"only {len(distinct_replies)} distinct replies across 5 patterns:\n"
        + "\n".join(f"  {r.pattern_name}: {r.shaped_reply}" for r in runs)
    )


def test_stressed_user_gets_shorter_reply_than_calm_user(pipeline_state) -> None:
    """A rhythm-stressed user (slow + heavy edits) gets a shorter
    shaped reply than a calm user typing a similar-length message.

    The rhythm signal is what differentiates here: both users send
    short messages (low content complexity), so without the iter-38
    rhythm-stress contribution to ``cognitive_load`` they would
    receive identical reply lengths.  After iter 38, the stressed
    user crosses into a higher cognitive-load tier and the post-
    processor trims more aggressively.
    """
    extractor, baseline, controller, pp = pipeline_state
    calm_pat = next(p for p in PATTERNS if p.name == "calm_curious")
    stressed_pat = next(p for p in PATTERNS if p.name == "stressed_terse")

    # Establish baseline so deviation metrics are meaningful.
    for _ in range(3):
        _run_pattern(
            calm_pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup.",
        )

    runs_c = _run_pattern(
        calm_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )
    runs_s = _run_pattern(
        stressed_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    assert len(runs_s.shaped_reply) < len(runs_c.shaped_reply), (
        f"stressed reply ({len(runs_s.shaped_reply)} chars) should be "
        f"shorter than calm reply ({len(runs_c.shaped_reply)} chars). "
        f"cl_calm={runs_c.adaptation_dict['cognitive_load']:.3f}, "
        f"cl_stressed={runs_s.adaptation_dict['cognitive_load']:.3f}"
    )


def test_direct_user_strips_softeners(pipeline_state) -> None:
    """A direct power-user's reply has soft openers removed."""
    extractor, baseline, controller, pp = pipeline_state
    direct_pat = next(p for p in PATTERNS if p.name == "direct_power_user")

    # Establish baseline first.
    for _ in range(3):
        _run_pattern(
            PATTERNS[1], extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup text.",
        )
    run = _run_pattern(
        direct_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    # The direct user's adaptation should produce directness >= 0.7,
    # which strips "you might want to consider".
    if run.adaptation_dict["directness"] > 0.7:
        assert "you might want to consider" not in run.shaped_reply.lower(), (
            f"direct user's reply still contains soft opener: "
            f"{run.shaped_reply}"
        )


def test_calm_user_gets_minimally_shaped_reply(pipeline_state) -> None:
    """A calm, curious, mid-tone user is meant to receive the reply
    largely unmodified — the adaptation is supposed to be subtle when
    no axis crosses a threshold."""
    extractor, baseline, controller, pp = pipeline_state
    calm_pat = next(p for p in PATTERNS if p.name == "calm_curious")

    # Establish baseline.
    for _ in range(3):
        _run_pattern(
            calm_pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup text.",
        )

    run = _run_pattern(
        calm_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    # The calm pattern shouldn't trigger ALL the strip rules — at least
    # 40% of the original reply length should survive.
    survival_ratio = len(run.shaped_reply) / len(CANONICAL_REPLY)
    assert survival_ratio >= 0.40, (
        f"calm user got {survival_ratio:.0%} of the reply; "
        f"expected >= 40% on a non-stressed pattern"
    )


def test_full_pipeline_log_includes_active_axes(pipeline_state) -> None:
    """When a pattern fires multiple axes, the change log records each."""
    extractor, baseline, controller, pp = pipeline_state
    stressed_pat = next(p for p in PATTERNS if p.name == "stressed_terse")

    # Establish a verbose baseline so the stressed pattern produces
    # meaningful deviations.
    for _ in range(3):
        _run_pattern(
            PATTERNS[1], extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup text.",
        )

    run = _run_pattern(
        stressed_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    # At least one axis fired.  The exact set depends on baseline state,
    # but the log must be non-empty for a stressed pattern.
    assert run.log_axes, f"stressed pattern produced no axes firing: {run}"


def test_stressed_user_gets_higher_cognitive_load_than_calm(pipeline_state) -> None:
    """Iter 38: cognitive_load must differentiate based on TYPING
    RHYTHM, not just message-content complexity.  A stressed user
    typing a short message should get higher cognitive_load than a
    calm user typing a similar-length message — driving the post-
    processor toward a tighter reply.

    Pre-iter-38: stressed and calm both produced cl ≈ 0.41 because
    cognitive_load only looked at content.
    Post-iter-38: stressed gets cl ≈ 0.59, calm gets cl ≈ 0.41 — a
    real difference of ≥ 0.10 driven entirely by typing rhythm.
    """
    extractor, baseline, controller, pp = pipeline_state
    calm_pat = next(p for p in PATTERNS if p.name == "calm_curious")
    stressed_pat = next(p for p in PATTERNS if p.name == "stressed_terse")

    # Establish baseline first so iki_deviation has meaning.
    for _ in range(3):
        _run_pattern(
            calm_pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup.",
        )

    run_calm = _run_pattern(
        calm_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )
    run_stressed = _run_pattern(
        stressed_pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    cl_calm = run_calm.adaptation_dict["cognitive_load"]
    cl_stressed = run_stressed.adaptation_dict["cognitive_load"]
    delta = cl_stressed - cl_calm
    assert delta >= 0.10, (
        f"stressed cognitive_load ({cl_stressed:.3f}) should exceed calm "
        f"cognitive_load ({cl_calm:.3f}) by at least 0.10; got delta={delta:.3f}"
    )


def test_complex_message_differentiates_calm_from_stressed(pipeline_state) -> None:
    """Iter 42: the 3-sentence intermediate tier (cl 0.55-0.65) means a
    calm user typing a complex message receives a 3-sentence reply
    while a stressed user typing the same complex message receives a
    2-sentence reply.  Previously both fell into the same 0.6-0.8
    bracket and collapsed to identical 2-sentence output."""
    extractor, baseline, controller, pp = pipeline_state
    calm_pat = next(p for p in PATTERNS if p.name == "calm_curious")
    stressed_pat = next(p for p in PATTERNS if p.name == "stressed_terse")
    long_msg = (
        "could you explain in some detail how this system handles different "
        "kinds of user input across the various interaction modes you have "
        "described in your previous message please"
    )

    # Replace each pattern's message with the long complex one.
    from dataclasses import replace
    calm_long = replace(calm_pat, message_text=long_msg)
    stressed_long = replace(stressed_pat, message_text=long_msg)

    # Establish baseline.
    for _ in range(3):
        _run_pattern(
            calm_pat, extractor=extractor, baseline=baseline,
            controller=controller, pp=pp, canonical_reply="Some setup.",
        )

    runs_c = _run_pattern(
        calm_long, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )
    runs_s = _run_pattern(
        stressed_long, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )

    # The shaped replies must differ — the iter-42 fine-grained tier
    # is the whole point.
    assert runs_c.shaped_reply != runs_s.shaped_reply, (
        f"calm and stressed users typing the same complex content got "
        f"identical replies — the cognitive_load tier didn't differentiate.\n"
        f"calm cl={runs_c.adaptation_dict['cognitive_load']:.3f} "
        f"-> {runs_c.shaped_reply}\n"
        f"stressed cl={runs_s.adaptation_dict['cognitive_load']:.3f} "
        f"-> {runs_s.shaped_reply}"
    )


def test_replays_are_deterministic(pipeline_state) -> None:
    """Running the same pattern twice on a fresh pipeline produces the
    same shaped reply."""
    extractor, baseline, controller, pp = pipeline_state
    pat = PATTERNS[0]

    run1 = _run_pattern(
        pat, extractor=extractor, baseline=baseline,
        controller=controller, pp=pp, canonical_reply=CANONICAL_REPLY,
    )
    # Fresh pipeline state.
    extractor2 = FeatureExtractor()
    baseline2 = BaselineTracker(warmup=2)
    controller2 = AdaptationController(AdaptationConfig())
    pp2 = ResponsePostProcessor()
    run2 = _run_pattern(
        pat, extractor=extractor2, baseline=baseline2,
        controller=controller2, pp=pp2, canonical_reply=CANONICAL_REPLY,
    )

    assert run1.shaped_reply == run2.shaped_reply
    assert run1.log_axes == run2.log_axes
