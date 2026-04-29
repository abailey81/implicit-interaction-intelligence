"""Iter 61 — chaos / fuzz regression test for the adaptation pipeline.

Drives 100 deterministic-pseudo-random pathological inputs through
the full FeatureExtractor + BaselineTracker + AdaptationController +
ResponsePostProcessor stack and asserts that:

  * The pipeline never raises.
  * Every adaptation axis is a finite float in [0, 1].
  * The shaped reply is non-empty (the post-processor's empty-fallback
    catches any otherwise-empty output).

This pins the iter-61 robustness invariant — if a future refactor
re-introduces NaN propagation or a sentence-trim that produces empty
text, this test fires immediately.

The full 1000-iteration version lives at
``D:/tmp/chaos_fuzz_emulation.py``.
"""

from __future__ import annotations

import math
import random

import pytest

from i3.adaptation.controller import AdaptationController
from i3.cloud.postprocess import ResponsePostProcessor
from i3.config import AdaptationConfig
from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.user_model.types import DeviationMetrics


CANONICAL = (
    "Sure! Absolutely happy to help. "
    "You might want to consider that perhaps approximately five different "
    "perspectives could provide additional context."
)


PATHOLOGICAL_MESSAGES = [
    "",                                              # empty
    "a",                                             # single char
    "?",                                             # just punctuation
    " " * 50,                                        # all whitespace
    "a" * 500,                                       # repeated single char
    "aaa bbb " * 30,                                 # repeated phrase
    "🎉🎊🎈" * 5,                                     # emoji-heavy
    "asdf;jkl;asdf;jkl",                             # nonsense + punctuation
    "WHY ISN'T THIS WORKING I'VE TRIED EVERYTHING",  # all caps
    "uhh i think maybe perhaps probably this isnt working",  # hedge stack
    "yo lol idk gonna brb",                          # slang stack
    "Pursuant to the matter, kindly advise.",        # formal
    "let me think...",                               # ellipsis
    "hello\n\nworld\n\n",                            # multi-line
    "set timer FOR 30 SECONDS NOW please",           # mixed case
    "did you mean 'a' or 'b'??!?!?!",                # punctuation chaos
    "1 + 1 = 2, right? 2 * 3 = 6",                   # math
    " " + "hi " * 100,                               # repeated word
    "hello there  general  kenobi",                  # multi-spaces
]


def _gen_km(rng: random.Random) -> dict:
    iki = rng.choice([
        rng.uniform(20, 800),
        rng.uniform(0, 1),
        rng.uniform(2000, 5000),
        0,
    ])
    std = rng.choice([rng.uniform(0, 100), rng.uniform(500, 2000), 0])
    edits = rng.choice([0, 1, 2, 5, 10, 50, 100])
    return {
        "mean_iki_ms": iki,
        "std_iki_ms": std,
        "mean_burst_length": rng.uniform(0, 200),
        "mean_pause_duration_ms": rng.uniform(0, 30000),
        "backspace_ratio": rng.uniform(0, 1),
        "composition_speed_cps": rng.uniform(0, 100),
        "pause_before_send_ms": rng.uniform(0, 60000),
        "editing_effort": min(1.0, edits / 10.0),
    }


@pytest.fixture
def fresh_pipeline():
    return (
        FeatureExtractor(),
        BaselineTracker(warmup=2),
        AdaptationController(AdaptationConfig()),
        ResponsePostProcessor(),
    )


@pytest.mark.parametrize("seed_offset", range(100))
def test_pathological_input_does_not_crash_or_produce_garbage(
    seed_offset: int,
) -> None:
    """100 deterministic-pseudo-random pathological inputs must each
    produce a valid AdaptationVector + non-empty shaped reply."""
    rng = random.Random(20260429 + seed_offset)
    msg = rng.choice(PATHOLOGICAL_MESSAGES)
    km = _gen_km(rng)

    ext = FeatureExtractor()
    baseline = BaselineTracker(warmup=2)
    ctrl = AdaptationController(AdaptationConfig())
    pp = ResponsePostProcessor()

    fv = ext.extract(
        keystroke_metrics=km, message_text=msg, history=[],
        baseline=baseline, session_start_ts=0.0, current_ts=30.0,
    )
    baseline.update(fv)
    dev = DeviationMetrics(
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
    av = ctrl.compute(fv, dev)
    shaped, _ = pp.adapt_with_log(CANONICAL, av)

    # Every axis must be a finite float in [0, 1].
    for name, val in [
        ("cognitive_load", av.cognitive_load),
        ("formality", av.style_mirror.formality),
        ("verbosity", av.style_mirror.verbosity),
        ("emotionality", av.style_mirror.emotionality),
        ("directness", av.style_mirror.directness),
        ("emotional_tone", av.emotional_tone),
        ("accessibility", av.accessibility),
    ]:
        assert math.isfinite(val), (
            f"seed_offset={seed_offset}, msg={msg!r}: "
            f"{name} non-finite: {val}"
        )
        assert 0.0 <= val <= 1.0, (
            f"seed_offset={seed_offset}, msg={msg!r}: "
            f"{name} out of [0,1]: {val}"
        )

    # Shaped reply must be non-empty (post-processor empty-fallback
    # catches any otherwise-empty output).
    assert shaped, (
        f"seed_offset={seed_offset}, msg={msg!r}: empty shaped reply"
    )
    assert shaped.strip(), (
        f"seed_offset={seed_offset}, msg={msg!r}: whitespace-only reply"
    )
