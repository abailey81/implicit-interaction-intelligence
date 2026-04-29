"""Iter 62 — pin reachability of every user-state label.

The Live State Badge classifies each turn into one of 6 discrete
states: ``calm``, ``focused``, ``stressed``, ``tired``, ``distracted``,
``warming up``.

A user reported only seeing 2 states ("calm" / "focused") in a real
session — which was correct (they were a calm/focused user).  But it
raised the question: are the OTHER 4 states actually reachable, or
has a calibration shift made them dead labels?

This test crafts a deterministic signal pattern for each target state
and asserts the classifier returns that label.  If a future
calibration tweak makes any state unreachable, the corresponding test
fires immediately.
"""

from __future__ import annotations

import pytest

from i3.affect.state_classifier import classify_user_state


# Each tuple: (target_state, summary, kwargs for classify_user_state)
SCENARIOS = [
    (
        "warming up",
        "First message of session, baseline not yet established",
        dict(
            adaptation={"cognitive_load": 0.4, "style_mirror": {"formality": 0.5}},
            composition_time_ms=2500, edit_count=0,
            iki_mean=120, iki_std=20, engagement_score=0.5,
            deviation_from_baseline=0.0,
            baseline_established=False, messages_in_session=1,
        ),
    ),
    (
        "calm",
        "Mid-session, normal IKI, low load, low edits",
        dict(
            adaptation={"cognitive_load": 0.30, "style_mirror": {"formality": 0.45}},
            composition_time_ms=2200, edit_count=0,
            iki_mean=110, iki_std=15, engagement_score=0.55,
            deviation_from_baseline=0.0,
            baseline_established=True, messages_in_session=10,
        ),
    ),
    (
        "focused",
        "Cognitive sweet spot 0.55, raised formality 0.75",
        dict(
            adaptation={"cognitive_load": 0.55, "style_mirror": {"formality": 0.75}},
            composition_time_ms=3000, edit_count=0,
            iki_mean=120, iki_std=15, engagement_score=0.65,
            deviation_from_baseline=0.0,
            baseline_established=True, messages_in_session=10,
        ),
    ),
    (
        "stressed",
        "High load + multiple edits + elevated IKI variance + long composition",
        dict(
            adaptation={"cognitive_load": 0.85, "style_mirror": {"formality": 0.4}},
            composition_time_ms=7000, edit_count=4,
            iki_mean=200, iki_std=60, engagement_score=0.4,
            deviation_from_baseline=0.0,
            baseline_established=True, messages_in_session=10,
        ),
    ),
    (
        "tired",
        "Long composition + slow IKI + low engagement",
        dict(
            adaptation={"cognitive_load": 0.55, "style_mirror": {"formality": 0.5}},
            composition_time_ms=8500, edit_count=2,
            iki_mean=240, iki_std=30, engagement_score=0.3,
            deviation_from_baseline=0.0,
            baseline_established=True, messages_in_session=10,
        ),
    ),
    (
        "distracted",
        "High IKI variance + normal IKI mean + intermittent edits",
        dict(
            adaptation={"cognitive_load": 0.50, "style_mirror": {"formality": 0.5}},
            composition_time_ms=4500, edit_count=2,
            iki_mean=120, iki_std=70, engagement_score=0.45,
            deviation_from_baseline=0.0,
            baseline_established=True, messages_in_session=10,
        ),
    ),
]


@pytest.mark.parametrize("target_state,summary,kwargs", SCENARIOS,
                         ids=[s[0] for s in SCENARIOS])
def test_state_is_reachable(
    target_state: str,
    summary: str,
    kwargs: dict,
) -> None:
    """Each of the 6 user states must be reachable with a crafted
    signal pattern.  If any state becomes unreachable (e.g. a future
    calibration shift removes its scoring band), this test fires."""
    label = classify_user_state(**kwargs)
    assert label.state == target_state, (
        f"expected state={target_state!r}, got {label.state!r} "
        f"(secondary={label.secondary_state}, conf={label.confidence:.3f})\n"
        f"scenario: {summary}"
    )
    # Confidence must be a finite probability.
    assert 0.0 <= label.confidence <= 1.0, (
        f"confidence {label.confidence} out of [0,1]"
    )
