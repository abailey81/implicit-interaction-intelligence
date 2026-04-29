"""Full-pipeline integration test (100-turn session, multi-archetype).

Iter 18 — drives the FeatureExtractor + BaselineTracker +
state_classifier + AffectShiftDetector + KeystrokeAuthenticator end-
to-end on a 100-turn synthetic session that transitions through
multiple typing patterns.  Asserts every invariant from iters 1–17
holds *jointly* — i.e. one component's output never leaves another
component in an invalid state.

The synthetic trajectory:

* Turns 0–19   — calm baseline (speed_typist archetype).
* Turns 20–39 — rising load (gradual transition to anxious_typist).
* Turns 40–59 — sustained stress (anxious_typist).
* Turns 60–79 — recovery (back to calm).
* Turns 80–99 — second stress wave (different archetype).

Invariants asserted at every turn:

* All 32 fv fields are finite.
* state classifier returns a valid label + confidence in [0, 1].
* shift detector confidence is 0 or in [0.5, 1.0] consistently with
  detected.
* iki_delta_pct, edit_delta_pct, magnitude all finite.
* keystroke authenticator never raises; status always a valid state.
* Per-user baseline mean stays bounded.

This is the gold-standard validation for the precision sweep.
"""

from __future__ import annotations

import math
import random

import torch

from i3.affect.shift_detector import AffectShiftDetector
from i3.affect.state_classifier import classify_user_state
from i3.biometric.keystroke_auth import KeystrokeAuthenticator
from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.interaction.types import InteractionFeatureVector


def _phase_for_turn(turn: int) -> dict:
    """Synthetic 100-turn phase plan."""
    if turn < 20:
        # Calm baseline.
        return dict(iki_mean=110.0, iki_std=18.0, comp=2000.0, edits=0, eng=0.65)
    if turn < 40:
        # Rising load — gradual transition.
        progress = (turn - 20) / 20  # 0 → 1
        return dict(
            iki_mean=110.0 + progress * 90.0,
            iki_std=18.0 + progress * 30.0,
            comp=2000.0 + progress * 4000.0,
            edits=int(progress * 4),
            eng=0.65 - progress * 0.20,
        )
    if turn < 60:
        # Sustained stress.
        return dict(iki_mean=200.0, iki_std=48.0, comp=6000.0, edits=4, eng=0.45)
    if turn < 80:
        # Recovery — typing FASTER than baseline (user is energised /
        # focused).  Falls below baseline IKI, edits at zero, so the
        # falling_load condition fires.  (Iter 7's fixed-baseline
        # anchor means "recovery from stress" needs to dip BELOW the
        # original baseline to register as a falling shift; mere
        # return-to-baseline produces neutral.)
        return dict(iki_mean=85.0, iki_std=15.0, comp=1700.0, edits=0, eng=0.70)
    # Second stress wave (different shape — high variance, not just slow).
    return dict(iki_mean=140.0, iki_std=70.0, comp=4500.0, edits=2, eng=0.50)


def test_full_pipeline_100_turn_integration() -> None:
    user_id = "u_integration"
    session_id = "s_integration"

    extractor = FeatureExtractor()
    baseline = BaselineTracker(warmup=3)
    detector = AffectShiftDetector()
    authenticator = KeystrokeAuthenticator(enrolment_target=5)
    history: list[InteractionFeatureVector] = []

    # Stable embedding for the authenticator (so its template is
    # deterministic).
    template_emb = torch.zeros(64)
    template_emb[:32] = 0.5

    rng = random.Random(42)

    state_labels: list[str] = []
    confidences: list[float] = []
    shift_detections: list[bool] = []
    shift_directions: list[str] = []
    auth_states: list[str] = []

    base_ts = 0.0
    for turn in range(100):
        phase = _phase_for_turn(turn)

        # Add tiny per-turn jitter.
        iki_jitter = rng.uniform(-0.05, 0.05) * phase["iki_mean"]
        iki_mean = max(20.0, phase["iki_mean"] + iki_jitter)

        km = {
            "mean_iki_ms": iki_mean,
            "std_iki_ms": max(5.0, phase["iki_std"] + rng.uniform(-3.0, 3.0)),
            "mean_burst_length": 8.0,
            "mean_pause_duration_ms": 200.0,
            "backspace_ratio": phase["edits"] / 50.0,
            "composition_speed_cps": 4.0,
            "pause_before_send_ms": 300.0,
            "editing_effort": min(1.0, phase["edits"] / 10.0),
        }

        # 1. FEATURE EXTRACTION.
        fv = extractor.extract(
            keystroke_metrics=km,
            message_text="the quick brown fox jumps over the lazy dog",
            history=history[-8:],
            baseline=baseline,
            session_start_ts=base_ts,
            current_ts=(turn + 1) * 30.0,
        )
        history.append(fv)
        baseline.update(fv)

        # All 32 fields must be finite.
        for fname in fv.__dataclass_fields__:
            v = getattr(fv, fname)
            if isinstance(v, (int, float)):
                assert math.isfinite(v), (
                    f"non-finite {fname}={v} on turn {turn} phase {phase}"
                )

        # 2. STATE CLASSIFICATION.
        cl = max(0.0, min(1.0, 0.5 * fv.editing_effort + 0.5 * fv.iki_deviation + 0.4))
        adaptation = {
            "cognitive_load": cl,
            "formality": float(fv.formality),
            "verbosity": 0.5,
            "accessibility": 0.0,
        }
        label = classify_user_state(
            adaptation=adaptation,
            composition_time_ms=phase["comp"],
            edit_count=phase["edits"],
            iki_mean=iki_mean,
            iki_std=phase["iki_std"],
            engagement_score=phase["eng"],
            deviation_from_baseline=0.0,
            baseline_established=baseline.is_established,
            messages_in_session=turn + 1,
        )
        state_labels.append(label.state)
        confidences.append(label.confidence)
        # Invariants.
        assert label.state in {
            "calm", "focused", "stressed", "tired", "distracted", "warming up"
        }
        assert math.isfinite(label.confidence)
        assert 0.0 <= label.confidence <= 1.0

        # 3. SHIFT DETECTION.
        shift = detector.observe(
            user_id=user_id,
            session_id=session_id,
            embedding=template_emb + 0.01 * torch.randn(64),
            composition_time_ms=phase["comp"],
            edit_count=phase["edits"],
            pause_before_send_ms=300.0,
            keystroke_iki_mean=iki_mean,
            keystroke_iki_std=phase["iki_std"],
        )
        shift_detections.append(shift.detected)
        shift_directions.append(shift.direction)
        # Invariants.
        assert math.isfinite(shift.magnitude)
        assert math.isfinite(shift.iki_delta_pct)
        assert math.isfinite(shift.edit_delta_pct)
        assert math.isfinite(shift.confidence)
        if shift.detected:
            assert 0.5 <= shift.confidence <= 1.0
        else:
            assert shift.confidence == 0.0
        assert shift.direction in {"rising_load", "falling_load", "neutral"}

        # 4. IDENTITY LOCK / KEYSTROKE AUTH.
        match = authenticator.observe(
            user_id,
            embedding=template_emb,
            iki_mean=iki_mean,
            iki_std=phase["iki_std"],
            composition_time_ms=phase["comp"],
            edit_count=phase["edits"],
        )
        auth_states.append(match.state)
        assert match.state in {
            "unregistered", "registering", "registered", "verifying", "mismatch"
        }
        assert math.isfinite(match.similarity)
        assert math.isfinite(match.confidence)
        assert 0.0 <= match.confidence <= 1.0

    # ---- post-run assertions ----

    # Phase 1 (calm baseline) — no high-load labels among turns 5..19.
    early_calm = state_labels[5:20]
    bad = {"stressed", "tired", "distracted"}
    assert not (bad & set(early_calm)), (
        f"calm baseline phase produced high-load labels: {early_calm}"
    )

    # Phase 2/3 (rising load + sustained stress) — at least one
    # rising_load shift fires.
    stress_directions = shift_directions[20:60]
    assert "rising_load" in stress_directions, (
        f"expected a rising_load shift during turns 20-59: {stress_directions}"
    )

    # Phase 4 (recovery) — at least one falling_load shift fires.
    recovery_directions = shift_directions[60:80]
    assert "falling_load" in recovery_directions, (
        f"expected a falling_load shift during turns 60-79: {recovery_directions}"
    )

    # Phase 5 (second stress wave, different shape) — distracted-ish
    # labels should appear.
    late_states = state_labels[80:100]
    high_load_late = {"stressed", "distracted", "tired"}
    assert high_load_late & set(late_states), (
        f"expected high-load labels in turns 80-99: {late_states}"
    )

    # Identity Lock should have fully registered by mid-session.
    mid_states = auth_states[10:90]
    assert "registered" in mid_states or "verifying" in mid_states, (
        f"expected the authenticator to reach registered/verifying: "
        f"{set(mid_states)}"
    )

    # Per-user baseline mean for mean_iki should be bounded
    # (sanity check Welford remained stable).
    assert math.isfinite(baseline.get_mean("mean_iki"))
    assert baseline.get_mean("mean_iki") >= 0.0
    assert baseline.get_std("mean_iki") >= 0.0


# ---------------------------------------------------------------------------
# Iter 23 — full-pipeline Hypothesis fuzzing
# ---------------------------------------------------------------------------

import hypothesis  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@given(
    turns=st.lists(
        st.tuples(
            st.floats(min_value=20.0, max_value=600.0),   # iki_mean
            st.floats(min_value=5.0, max_value=120.0),    # iki_std
            st.floats(min_value=500.0, max_value=15000.0),  # composition
            st.integers(min_value=0, max_value=10),       # edits
            st.floats(min_value=0.0, max_value=1.0),      # engagement
        ),
        min_size=5,
        max_size=40,
    ),
)
def test_full_pipeline_invariants_under_hypothesis_fuzzing(turns) -> None:
    """Iter 23 — random sequences of (iki_mean, iki_std, composition,
    edits, engagement) tuples driven through the full pipeline must
    never crash and must keep every output invariant satisfied."""
    extractor = FeatureExtractor()
    baseline = BaselineTracker(warmup=3)
    detector = AffectShiftDetector()
    authenticator = KeystrokeAuthenticator(enrolment_target=5)
    history: list[InteractionFeatureVector] = []
    template_emb = torch.zeros(64); template_emb[:32] = 0.5

    for turn_idx, (iki_m, iki_s, comp, edits, eng) in enumerate(turns):
        km = {
            "mean_iki_ms": iki_m,
            "std_iki_ms": iki_s,
            "mean_burst_length": 8.0,
            "mean_pause_duration_ms": 200.0,
            "backspace_ratio": min(1.0, edits / 50.0),
            "composition_speed_cps": 4.0,
            "pause_before_send_ms": 300.0,
            "editing_effort": min(1.0, edits / 10.0),
        }
        fv = extractor.extract(
            keystroke_metrics=km,
            message_text="the quick brown fox",
            history=history[-8:],
            baseline=baseline,
            session_start_ts=0.0,
            current_ts=(turn_idx + 1) * 30.0,
        )
        history.append(fv); baseline.update(fv)

        for fname in fv.__dataclass_fields__:
            v = getattr(fv, fname)
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"non-finite {fname}={v}"

        cl = max(0.0, min(1.0, 0.5 * fv.editing_effort + 0.5 * fv.iki_deviation + 0.4))
        label = classify_user_state(
            adaptation={"cognitive_load": cl, "formality": float(fv.formality),
                        "verbosity": 0.5, "accessibility": 0.0},
            composition_time_ms=comp, edit_count=edits,
            iki_mean=iki_m, iki_std=iki_s,
            engagement_score=eng, deviation_from_baseline=0.0,
            baseline_established=baseline.is_established,
            messages_in_session=turn_idx + 1,
        )
        assert label.state in {"calm", "focused", "stressed", "tired", "distracted", "warming up"}
        assert 0.0 <= label.confidence <= 1.0

        shift = detector.observe(
            user_id="u_fuzz", session_id="s_fuzz",
            embedding=template_emb,
            composition_time_ms=comp, edit_count=edits,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=iki_m, keystroke_iki_std=iki_s,
        )
        assert math.isfinite(shift.magnitude)
        assert math.isfinite(shift.confidence)
        if shift.detected:
            assert 0.5 <= shift.confidence <= 1.0
        else:
            assert shift.confidence == 0.0

        match = authenticator.observe(
            "u_fuzz", embedding=template_emb,
            iki_mean=iki_m, iki_std=iki_s,
            composition_time_ms=comp, edit_count=edits,
        )
        assert match.state in {"unregistered", "registering", "registered", "verifying", "mismatch"}
        assert 0.0 <= match.confidence <= 1.0
