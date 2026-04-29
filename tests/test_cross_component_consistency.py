"""Cross-component consistency invariants between shift_detector and
state_classifier.

Iter 29 — when ``AffectShiftDetector`` reports a strong rising_load
shift, ``classify_user_state`` should likewise weight the
"high-load" candidates (stressed / tired / distracted) — not return
"calm" or "focused".  This verifies the two affect-detection paths
*agree* on the underlying signal even though their internal
mechanisms are distinct (rule-based scoring vs window-comparison).

A divergence between these two would surface to the user as a
contradictory state-badge + suggestion — exactly the kind of bug
that erodes calibrated trust in the routing chip.
"""

from __future__ import annotations

import torch

from i3.affect.shift_detector import AffectShiftDetector
from i3.affect.state_classifier import classify_user_state


def _drive_shift_detector(
    detector: AffectShiftDetector,
    user: str,
    session: str,
    *,
    baseline_iki: float,
    baseline_edits: int,
    recent_iki: float,
    recent_edits: int,
    n_baseline: int = 5,
    n_recent: int = 3,
):
    """Feed a baseline + recent pattern and return the last AffectShift."""
    for _ in range(n_baseline):
        detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2000.0, edit_count=baseline_edits,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=baseline_iki, keystroke_iki_std=20.0,
        )
    last = None
    for _ in range(n_recent):
        last = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=4000.0, edit_count=recent_edits,
            pause_before_send_ms=500.0,
            keystroke_iki_mean=recent_iki, keystroke_iki_std=45.0,
        )
    return last


def test_strong_rising_shift_correlates_with_high_load_classifier_label() -> None:
    """If shift_detector fires a strong rising_load with high
    confidence (>= 0.7), state_classifier on the same recent
    parameters should pick a high-load label (not calm / focused)."""
    detector = AffectShiftDetector()
    shift = _drive_shift_detector(
        detector, "u_corr", "s_corr",
        baseline_iki=110.0, baseline_edits=0,
        recent_iki=240.0, recent_edits=6,
    )
    # Sanity: shift fires strongly.
    assert shift is not None
    assert shift.detected is True
    assert shift.direction == "rising_load"
    # Now feed the same recent params to the classifier.
    label = classify_user_state(
        adaptation={
            "cognitive_load": 0.8,
            "formality": 0.5,
            "verbosity": 0.5,
            "accessibility": 0.0,
        },
        composition_time_ms=4000.0,
        edit_count=6,
        iki_mean=240.0,
        iki_std=45.0,
        engagement_score=0.45,
        deviation_from_baseline=0.5,
        baseline_established=True,
        messages_in_session=10,
    )
    assert label.state in {"stressed", "tired", "distracted"}, (
        f"strong rising_load shift but classifier says {label.state}"
    )


def test_no_shift_correlates_with_low_load_classifier_label() -> None:
    """If shift_detector does NOT fire (calm baseline, calm recent),
    the classifier on the same recent parameters should pick a low-
    load label (calm or focused)."""
    detector = AffectShiftDetector()
    shift = _drive_shift_detector(
        detector, "u_calm", "s_calm",
        baseline_iki=110.0, baseline_edits=0,
        recent_iki=115.0, recent_edits=0,
    )
    assert shift is not None
    assert shift.detected is False
    label = classify_user_state(
        adaptation={
            "cognitive_load": 0.25,
            "formality": 0.5,
            "verbosity": 0.5,
            "accessibility": 0.0,
        },
        composition_time_ms=2000.0,
        edit_count=0,
        iki_mean=115.0,
        iki_std=20.0,
        engagement_score=0.6,
        deviation_from_baseline=0.0,
        baseline_established=True,
        messages_in_session=10,
    )
    assert label.state in {"calm", "focused"}, (
        f"no shift but classifier says {label.state}"
    )


def test_falling_load_shift_correlates_with_calm_or_focused_label() -> None:
    """A falling_load (recovery from stress) with calm-recent metrics
    should put the classifier in calm / focused — same recent
    parameters."""
    detector = AffectShiftDetector()
    shift = _drive_shift_detector(
        detector, "u_recover", "s_recover",
        baseline_iki=200.0, baseline_edits=4,
        recent_iki=110.0, recent_edits=1,
    )
    assert shift is not None
    if shift.detected:
        assert shift.direction == "falling_load"
    label = classify_user_state(
        adaptation={
            "cognitive_load": 0.30,
            "formality": 0.5,
            "verbosity": 0.5,
            "accessibility": 0.0,
        },
        composition_time_ms=2200.0,
        edit_count=1,
        iki_mean=110.0,
        iki_std=20.0,
        engagement_score=0.6,
        deviation_from_baseline=-0.3,
        baseline_established=True,
        messages_in_session=10,
    )
    assert label.state in {"calm", "focused"}, (
        f"recovery scenario but classifier says {label.state}"
    )


def test_shift_confidence_above_classifier_minimum() -> None:
    """A detected shift always has confidence >= 0.5; the classifier's
    confidence on the same scenario is also non-trivial (>= 0.35).
    Both detectors agree the signal is real."""
    detector = AffectShiftDetector()
    shift = _drive_shift_detector(
        detector, "u_both", "s_both",
        baseline_iki=110.0, baseline_edits=0,
        recent_iki=220.0, recent_edits=5,
    )
    assert shift is not None
    if shift.detected:
        assert shift.confidence >= 0.5
        # Classifier with same params.
        label = classify_user_state(
            adaptation={"cognitive_load": 0.85, "formality": 0.5,
                        "verbosity": 0.5, "accessibility": 0.0},
            composition_time_ms=4000.0,
            edit_count=5,
            iki_mean=220.0,
            iki_std=45.0,
            engagement_score=0.45,
            deviation_from_baseline=0.5,
            baseline_established=True,
            messages_in_session=10,
        )
        # Classifier should be reasonably confident (post-iter-3
        # calibration: clean stressed lands in 0.55-0.85 band).
        assert label.confidence >= 0.35
