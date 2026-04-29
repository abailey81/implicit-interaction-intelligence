"""Calibration tests for ``i3.affect.state_classifier.classify_user_state``.

Iter 3 lifts the softmax temperature so borderline cases produce a
runner-up label and confidence calibrates honestly.  These tests
encode the post-iter-3 expectations.
"""

from __future__ import annotations

import pytest

from i3.affect.state_classifier import UserStateLabel, classify_user_state


def _label(**overrides) -> UserStateLabel:
    """Defaults are a clean "calm" baseline; override to probe other states."""
    base = dict(
        adaptation={
            "cognitive_load": 0.25,
            "formality": 0.5,
            "verbosity": 0.5,
            "accessibility": 0.0,
        },
        composition_time_ms=2000.0,
        edit_count=0,
        iki_mean=110.0,
        iki_std=15.0,
        engagement_score=0.6,
        deviation_from_baseline=0.0,
        baseline_established=True,
        messages_in_session=10,
    )
    base.update(overrides)
    return classify_user_state(**base)


# ---------------------------------------------------------------------------
# Determinism (same inputs → same label, every time)
# ---------------------------------------------------------------------------


def test_repeated_calls_are_stable() -> None:
    args = dict(
        adaptation={"cognitive_load": 0.65, "formality": 0.7, "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=3000.0,
        edit_count=1,
        iki_mean=115.0,
        iki_std=20.0,
        engagement_score=0.6,
        deviation_from_baseline=0.0,
        baseline_established=True,
        messages_in_session=10,
    )
    a = classify_user_state(**args)
    b = classify_user_state(**args)
    c = classify_user_state(**args)
    assert a.state == b.state == c.state
    assert a.confidence == b.confidence == c.confidence
    assert a.secondary_state == b.secondary_state == c.secondary_state


# ---------------------------------------------------------------------------
# Calibration: clean signals → confident, borderline signals → less confident
# ---------------------------------------------------------------------------


def test_clean_calm_signal_produces_high_but_not_saturated_confidence() -> None:
    """A clearly-calm user should land state=calm with confidence in [0.55, 0.95].

    iter 3 caps over-confidence: with temperature=0.35, even a clean
    win sits comfortably under 1.0.
    """
    label = _label()  # calm defaults
    assert label.state == "calm"
    assert 0.55 <= label.confidence <= 0.95


def test_clean_stressed_signal_produces_high_but_not_saturated_confidence() -> None:
    label = _label(
        adaptation={"cognitive_load": 0.85, "formality": 0.4, "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=6000.0,
        edit_count=5,
        iki_mean=180.0,
        iki_std=70.0,
        engagement_score=0.4,
    )
    assert label.state == "stressed"
    assert 0.55 <= label.confidence <= 0.95


def test_borderline_calm_focused_surfaces_secondary_state() -> None:
    """Cognitive load just inside the calm/focused boundary surfaces both.

    iter 3: with temperature=0.35 and gap-threshold 0.20, a user
    typing calmly with raised formality lands as calm with focused
    as the runner-up — the badge UI shows "calm/focused" rather
    than feigning certainty about one.
    """
    label = _label(
        adaptation={"cognitive_load": 0.45, "formality": 0.7, "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=2500.0,
        edit_count=0,
        iki_mean=120.0,
        iki_std=20.0,
        engagement_score=0.55,
    )
    # Either calm or focused is acceptable as the top label.
    assert label.state in {"calm", "focused"}
    # The other one must surface as secondary.
    assert label.secondary_state in {"calm", "focused"}
    assert label.secondary_state != label.state


def test_stressed_distracted_ambiguous_surfaces_secondary_state() -> None:
    """High IKI σ + high cognitive load is the canonical stressed/distracted.

    iter 3: must surface the runner-up so the UI can show
    "distracted/stressed" combined.
    """
    label = _label(
        # cl=0.75 fires both "high cognitive load" (stressed) and the
        # _band(0.35, 0.7) tail (distracted, attenuated).  iki_std=70
        # sits above the stressed peak (=60) AND inside the distracted
        # ramp (30-80).  Composition=5500 is into stressed territory.
        adaptation={"cognitive_load": 0.75, "formality": 0.4, "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=5500.0,
        edit_count=3,
        iki_mean=130.0,
        iki_std=70.0,
        engagement_score=0.45,
        deviation_from_baseline=0.4,
    )
    assert label.state in {"stressed", "distracted"}
    assert label.secondary_state in {"stressed", "distracted"}
    assert label.secondary_state != label.state


# ---------------------------------------------------------------------------
# Confidence floor sanity (no NaN, no negative)
# ---------------------------------------------------------------------------


def test_confidence_is_in_zero_one_interval() -> None:
    """Confidence is always a probability in [0, 1]."""
    test_cases = [
        # Clean signals
        dict(),
        dict(adaptation={"cognitive_load": 0.85, "formality": 0.4, "verbosity": 0.5, "accessibility": 0.0},
             composition_time_ms=5500.0, edit_count=4, iki_mean=180.0, iki_std=55.0, engagement_score=0.5),
        # Degenerate engagement
        dict(engagement_score=0.0),
        dict(engagement_score=1.0),
        # Pathological: zero composition time, zero edits
        dict(composition_time_ms=0.0, edit_count=0, iki_mean=0.0, iki_std=0.0),
    ]
    for case in test_cases:
        label = _label(**case)
        assert 0.0 <= label.confidence <= 1.0, f"out of range for case: {case}"


# ---------------------------------------------------------------------------
# Warm-up still wins on turn 1
# ---------------------------------------------------------------------------


def test_warming_up_still_wins_on_first_message() -> None:
    """Iter 3 must not regress the warm-up short-circuit."""
    label = _label(
        baseline_established=False,
        messages_in_session=0,
    )
    assert label.state == "warming up"
    assert label.confidence > 0.5  # warmup_strong = 1.4 raw → high prob


def test_warming_up_decays_after_baseline_established() -> None:
    """Once the baseline is established, warm-up no longer dominates."""
    label = _label(
        baseline_established=True,
        messages_in_session=10,
    )
    assert label.state != "warming up"


# ---------------------------------------------------------------------------
# Contributing signals correctness (post-calibration)
# ---------------------------------------------------------------------------


def test_contributing_signals_match_chosen_state() -> None:
    """Contributing signals come from the chosen state's signal map."""
    label = _label(
        adaptation={"cognitive_load": 0.85, "formality": 0.4, "verbosity": 0.5, "accessibility": 0.0},
        composition_time_ms=5500.0,
        edit_count=4,
        iki_mean=180.0,
        iki_std=55.0,
        engagement_score=0.5,
    )
    assert label.state == "stressed"
    # All contributing labels come from the stressed signal bank.
    expected_labels = {
        "high cognitive load",
        "elevated edit count",
        "elevated IKI variance",
        "long composition",
    }
    assert set(label.contributing_signals) <= expected_labels
    assert 1 <= len(label.contributing_signals) <= 3


def test_contributing_signals_are_non_empty_on_clean_states() -> None:
    """Every clean classification surfaces at least one contributing signal."""
    for adaptation_cl, expected_subset in [
        (0.25, "calm"),
        (0.85, "stressed"),
    ]:
        label = _label(
            adaptation={"cognitive_load": adaptation_cl, "formality": 0.5, "verbosity": 0.5, "accessibility": 0.0},
            composition_time_ms=5500.0 if adaptation_cl > 0.5 else 2000.0,
            edit_count=4 if adaptation_cl > 0.5 else 0,
            iki_mean=180.0 if adaptation_cl > 0.5 else 110.0,
            iki_std=55.0 if adaptation_cl > 0.5 else 15.0,
        )
        assert label.contributing_signals, f"no signals for cl={adaptation_cl}"


# ---------------------------------------------------------------------------
# Robustness to noisy adaptation dicts
# ---------------------------------------------------------------------------


def test_missing_adaptation_keys_default_safely() -> None:
    label = _label(adaptation={})  # empty dict
    # Defaults: cognitive_load=0.5, formality=0.5, verbosity=0.5, accessibility=0.0.
    # State should be deterministic regardless.
    assert label.state in {"calm", "focused", "stressed", "tired", "distracted", "warming up"}
    assert 0.0 <= label.confidence <= 1.0


def test_nan_adaptation_values_default_safely() -> None:
    label = _label(adaptation={"cognitive_load": float("nan"), "formality": float("inf")})
    assert label.state in {"calm", "focused", "stressed", "tired", "distracted", "warming up"}
    assert 0.0 <= label.confidence <= 1.0


def test_nested_style_mirror_format_supported() -> None:
    """The classifier accepts both flat and engine-nested adaptation dicts."""
    flat = _label(adaptation={"formality": 0.7, "cognitive_load": 0.3})
    nested = _label(adaptation={"cognitive_load": 0.3, "style_mirror": {"formality": 0.7}})
    assert flat.state == nested.state
    # Confidence should be approximately equal (allow for float-precision noise).
    assert abs(flat.confidence - nested.confidence) < 1e-9
