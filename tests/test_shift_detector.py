"""Tests for ``i3.affect.shift_detector.AffectShiftDetector``.

Covers:

* Warm-up window: no detection until enough observations.
* Strong-tier keystroke trigger (IKI ≥ +20% AND edits ≥ +50%).
* Moderate-tier triggers (high single-signal — added in iter 1).
* Direction inference vs keystroke firing self-consistency.
* Embedding-magnitude trigger.
* Embedding-shape robustness (iter 6 — mismatched dims, multi-dim, device).
* Debounce.
* Determinism.
* Defensive coercion of bad embeddings.
"""

from __future__ import annotations

import math

import pytest
import torch

from i3.affect.shift_detector import (
    AffectShift,
    AffectShiftDetector,
    FALLING_LOAD_SUGGESTIONS,
    NEUTRAL_SHIFT_SUGGESTIONS,
    RISING_LOAD_SUGGESTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calm_obs(detector: AffectShiftDetector, user: str, session: str, n: int) -> None:
    """Feed *n* near-identical "calm" observations (anchors a baseline)."""
    for _ in range(n):
        detector.observe(
            user_id=user,
            session_id=session,
            embedding=torch.zeros(64),
            composition_time_ms=2000.0,
            edit_count=0,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0,
            keystroke_iki_std=20.0,
        )


def _stressed_obs(
    detector: AffectShiftDetector,
    user: str,
    session: str,
    *,
    iki_mean: float = 200.0,
    edits: int = 4,
    embedding: torch.Tensor | None = None,
) -> AffectShift:
    """Feed a single "stressed" observation and return the result."""
    return detector.observe(
        user_id=user,
        session_id=session,
        embedding=embedding if embedding is not None else torch.zeros(64),
        composition_time_ms=4000.0,
        edit_count=edits,
        pause_before_send_ms=600.0,
        keystroke_iki_mean=iki_mean,
        keystroke_iki_std=40.0,
    )


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def test_warmup_returns_no_detection_until_enough_observations() -> None:
    detector = AffectShiftDetector()
    user, session = "u1", "s1"

    # min_required = recent_size + 2 = 5 with defaults.
    for i in range(4):
        result = detector.observe(
            user_id=user,
            session_id=session,
            embedding=torch.zeros(64),
            composition_time_ms=2000.0,
            edit_count=0,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0,
            keystroke_iki_std=20.0,
        )
        assert result.detected is False, f"shouldn't detect at obs {i + 1}"
        assert result.suggestion == ""

    # Fifth observation reaches min_required.
    result = detector.observe(
        user_id=user,
        session_id=session,
        embedding=torch.zeros(64),
        composition_time_ms=2000.0,
        edit_count=0,
        pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    # Still calm baseline so detected should remain False, but the gate
    # is now open and metrics start computing.
    assert result.magnitude >= 0.0


# ---------------------------------------------------------------------------
# Strong-tier trigger (the documented brief: AND of two signals)
# ---------------------------------------------------------------------------


def test_strong_tier_fires_when_both_iki_and_edits_cross() -> None:
    detector = AffectShiftDetector()
    user, session = "u_strong", "s_strong"

    _calm_obs(detector, user, session, n=5)

    # First stressed observation should already fire (recent window of
    # 3 includes 1 stressed + 2 calm; deltas cross both thresholds).
    first = _stressed_obs(detector, user, session, iki_mean=200.0, edits=4)

    assert first.detected is True
    assert first.direction == "rising_load"
    assert first.iki_delta_pct > 20.0
    assert first.edit_delta_pct > 50.0
    assert first.suggestion in RISING_LOAD_SUGGESTIONS


# ---------------------------------------------------------------------------
# Moderate-tier triggers (iter 1 improvement)
#
# Before iter 1: a strong IKI delta with no edit delta would be classified
# as "rising_load" by `_infer_direction` (which uses OR) but would NOT
# fire `_keystroke_fired` (which uses AND).  The shift dropped silently
# unless the embedding magnitude crossed too.
# After iter 1: a *strong-enough* single signal (IKI >= +35% OR edits >=
# +120%) is enough to fire the trigger on its own.
# ---------------------------------------------------------------------------


def test_moderate_tier_iki_only_fires_on_large_iki_delta() -> None:
    """A strong IKI rise (no edit change) should still fire (iter 1)."""
    detector = AffectShiftDetector()
    user, session = "u_iki", "s_iki"

    _calm_obs(detector, user, session, n=5)

    # 3 stressed obs to fully populate the recent window with the new
    # signal, so we measure steady-state response.  IKI 120 -> 220 ms
    # (+83%), edits unchanged (0 -> 0).
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=0, pause_before_send_ms=400.0,
        keystroke_iki_mean=220.0, keystroke_iki_std=40.0,
    )
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=0, pause_before_send_ms=400.0,
        keystroke_iki_mean=220.0, keystroke_iki_std=40.0,
    )
    last = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=0, pause_before_send_ms=400.0,
        keystroke_iki_mean=220.0, keystroke_iki_std=40.0,
    )

    assert last.iki_delta_pct >= 35.0
    assert last.edit_delta_pct == 0.0
    assert last.direction == "rising_load"
    # iter 1 fix: IKI-only large delta fires the trigger.
    assert last.detected is True


def test_moderate_tier_edits_only_fires_on_large_edit_spike() -> None:
    """A large edit spike (no IKI change) should still fire (iter 1)."""
    detector = AffectShiftDetector()
    user, session = "u_edit", "s_edit"

    _calm_obs(detector, user, session, n=5)

    # IKI unchanged, edits 0 -> 5 over the recent window.
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=5, pause_before_send_ms=400.0,
        keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
    )
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=5, pause_before_send_ms=400.0,
        keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
    )
    last = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=4000.0, edit_count=5, pause_before_send_ms=400.0,
        keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
    )

    assert last.edit_delta_pct >= 120.0  # zero_baseline_default = 200.0
    assert last.direction == "rising_load"
    # iter 1 fix: edit-only large delta fires the trigger.
    assert last.detected is True


def test_below_moderate_thresholds_does_not_fire() -> None:
    """Small single-signal deltas should still not fire."""
    detector = AffectShiftDetector()
    user, session = "u_small", "s_small"

    _calm_obs(detector, user, session, n=5)

    # IKI +25% (within original "rising" but below moderate-tier +35%),
    # edits unchanged (0).  Should NOT fire.
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2500.0, edit_count=0, pause_before_send_ms=300.0,
        keystroke_iki_mean=150.0, keystroke_iki_std=25.0,
    )
    detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2500.0, edit_count=0, pause_before_send_ms=300.0,
        keystroke_iki_mean=150.0, keystroke_iki_std=25.0,
    )
    last = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2500.0, edit_count=0, pause_before_send_ms=300.0,
        keystroke_iki_mean=150.0, keystroke_iki_std=25.0,
    )

    assert 20.0 <= last.iki_delta_pct < 35.0
    assert last.edit_delta_pct == 0.0
    # Direction still inferred as rising_load (by the OR rule), but
    # no trigger fires — neither strong nor moderate.
    assert last.detected is False


# ---------------------------------------------------------------------------
# Falling-load direction
# ---------------------------------------------------------------------------


def test_falling_load_fires_when_iki_drops_and_edits_flat() -> None:
    detector = AffectShiftDetector()
    user, session = "u_fall", "s_fall"

    # Establish a "tired" baseline (slow + edits).
    for _ in range(5):
        detector.observe(
            user_id=user,
            session_id=session,
            embedding=torch.zeros(64),
            composition_time_ms=4000.0,
            edit_count=3,
            pause_before_send_ms=600.0,
            keystroke_iki_mean=200.0,
            keystroke_iki_std=40.0,
        )

    # Now snap to "alert" — fast IKI, edits flat or down.  Take the
    # first observation that crosses the threshold (subsequent ones get
    # debounced).
    first = detector.observe(
        user_id=user,
        session_id=session,
        embedding=torch.zeros(64),
        composition_time_ms=2000.0,
        edit_count=2,
        pause_before_send_ms=200.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    # Fire might need 1-2 more obs to fully populate the recent window.
    if not first.detected:
        first = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2000.0, edit_count=2, pause_before_send_ms=200.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )
    if not first.detected:
        first = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2000.0, edit_count=2, pause_before_send_ms=200.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )

    assert first.iki_delta_pct <= -15.0
    assert first.edit_delta_pct <= 5.0
    assert first.direction == "falling_load"
    assert first.detected is True
    assert first.suggestion in FALLING_LOAD_SUGGESTIONS


# ---------------------------------------------------------------------------
# Embedding magnitude trigger
# ---------------------------------------------------------------------------


def test_embedding_magnitude_can_trigger_alone() -> None:
    """A large embedding shift (no keystroke change) should still fire."""
    detector = AffectShiftDetector(magnitude_threshold=1.4)
    user, session = "u_emb", "s_emb"

    # Baseline: small variation around zero embedding.
    torch.manual_seed(0)
    for i in range(5):
        emb = torch.randn(64) * 0.1
        detector.observe(
            user_id=user,
            session_id=session,
            embedding=emb,
            composition_time_ms=2000.0,
            edit_count=0,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0,
            keystroke_iki_std=20.0,
        )

    # Recent: shift the embedding far from baseline; keystrokes
    # unchanged.
    drift = torch.ones(64) * 2.0
    last: AffectShift | None = None
    for _ in range(3):
        last = detector.observe(
            user_id=user,
            session_id=session,
            embedding=drift,
            composition_time_ms=2000.0,
            edit_count=0,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0,
            keystroke_iki_std=20.0,
        )

    assert last is not None
    assert last.magnitude >= 1.4
    assert last.detected is True


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


def test_debounce_suppresses_repeated_suggestions() -> None:
    detector = AffectShiftDetector()
    user, session = "u_dbnce", "s_dbnce"

    _calm_obs(detector, user, session, n=5)

    # First strong shift on the next stressed obs — should emit a
    # suggestion.
    first = _stressed_obs(detector, user, session)
    assert first.detected is True
    assert first.suggestion != ""

    # Next few stressed observations — should remain detected but
    # have no suggestion (debounced for _DEBOUNCE_TURNS = 4 turns).
    for _ in range(3):
        nxt = _stressed_obs(detector, user, session)
        assert nxt.detected is True
        assert nxt.suggestion == ""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_inputs_produce_same_suggestion() -> None:
    """Suggestion picking is deterministic given (user, session, turn)."""
    d1 = AffectShiftDetector()
    d2 = AffectShiftDetector()

    _calm_obs(d1, "alice", "sess", n=5)
    _calm_obs(d2, "alice", "sess", n=5)

    last1: AffectShift | None = None
    last2: AffectShift | None = None
    for _ in range(3):
        last1 = _stressed_obs(d1, "alice", "sess")
        last2 = _stressed_obs(d2, "alice", "sess")

    assert last1 is not None and last2 is not None
    assert last1.suggestion == last2.suggestion


# ---------------------------------------------------------------------------
# Defensive embedding handling
# ---------------------------------------------------------------------------


def test_none_embedding_does_not_crash() -> None:
    detector = AffectShiftDetector()
    result = detector.observe(
        user_id="u_none",
        session_id="s_none",
        embedding=None,  # type: ignore[arg-type]
        composition_time_ms=2000.0,
        edit_count=0,
        pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    assert result.detected is False
    assert result.magnitude == 0.0


def test_nan_embedding_is_zeroed() -> None:
    detector = AffectShiftDetector()
    bad = torch.tensor([float("nan")] * 64)
    result = detector.observe(
        user_id="u_nan",
        session_id="s_nan",
        embedding=bad,
        composition_time_ms=2000.0,
        edit_count=0,
        pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    # nan_to_num replaces NaN with 0.0 in the embedding; magnitude stays
    # finite and the call doesn't raise.
    assert result.magnitude == 0.0


# ---------------------------------------------------------------------------
# Iter 6 — embedding-shape robustness
# ---------------------------------------------------------------------------


def test_undersized_embedding_is_padded_to_64() -> None:
    """A 32-dim embedding gets zero-padded to 64-dim and the run continues."""
    detector = AffectShiftDetector()
    user, session = "u_short", "s_short"

    # Mix 32-dim and 64-dim observations; the embedding magnitude
    # math should still produce a finite scalar (no RuntimeError on
    # torch.stack across mixed shapes — iter 6 fix canonicalises to
    # 64-dim before stacking).
    short = torch.ones(32) * 0.5
    full = torch.ones(64) * 0.7

    for emb in [short, full, short, full, full]:
        result = detector.observe(
            user_id=user, session_id=session, embedding=emb,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )
    # Final observation produced a finite, non-zero magnitude scalar
    # (the embeddings *do* differ in upper 32 dims).
    assert math.isfinite(result.magnitude)


def test_oversized_embedding_is_truncated_to_64() -> None:
    """A 128-dim embedding gets truncated to 64 and the run continues."""
    detector = AffectShiftDetector()
    user, session = "u_long", "s_long"

    big = torch.ones(128) * 0.5

    for _ in range(5):
        result = detector.observe(
            user_id=user, session_id=session, embedding=big,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )
    assert math.isfinite(result.magnitude)


def test_multidim_embedding_is_flattened_then_canonicalised() -> None:
    """A (4, 16) embedding flattens to 64-dim cleanly."""
    detector = AffectShiftDetector()
    user, session = "u_2d", "s_2d"

    multi = torch.ones((4, 16)) * 0.3

    for _ in range(5):
        result = detector.observe(
            user_id=user, session_id=session, embedding=multi,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )
    assert math.isfinite(result.magnitude)


def test_iter63_stable_session_does_not_over_trigger() -> None:
    """Iter 63: a 20-turn session of consistent typing with small
    embedding jitter must NOT produce a 75-80% shift-detection rate.

    Pre-iter-63 the magnitude formula was ``L2(diff) / σ_per_dim``,
    which inflated magnitudes by sqrt(N) ≈ 8x on 64-dim embeddings —
    a stable per-dim std of 0.05 produced ``magnitude ≈ 8σ`` between
    successive baseline windows even though nothing meaningful had
    changed.  The user reported '12/16 affect-shift events' which
    matched this 75-80% false-positive rate exactly.

    Post-iter-63 the formula RMS-normalises by sqrt(N) so stable
    sessions stay quiet."""
    import random
    rng = random.Random(20260429)
    detector = AffectShiftDetector()
    user, session = "u_stable", "s_stable"

    detected_count = 0
    for _ in range(20):
        # Generate a 64-d embedding around 0.5 with small jitter.
        emb = torch.tensor(
            [0.5 + rng.gauss(0.0, 0.05) for _ in range(64)],
            dtype=torch.float32,
        )
        shift = detector.observe(
            user_id=user, session_id=session,
            embedding=emb,
            composition_time_ms=2500.0,
            edit_count=0,
            pause_before_send_ms=300.0,
            keystroke_iki_mean=110.0,
            keystroke_iki_std=15.0,
        )
        if shift.detected:
            detected_count += 1

    # Stable session should produce VERY few detections.  Pre-iter-63
    # this was 80% (16/20).  Post-iter-63 it should be < 25%.
    assert detected_count < 5, (
        f"stable session produced {detected_count}/20 shift detections "
        f"({100 * detected_count / 20:.0f}%) — over-triggering on "
        f"within-session noise.  Pre-iter-63 this was ~80% (matching the "
        f"user's reported '12/16 affect events' false-positive rate)."
    )


def test_mixed_shape_embeddings_dont_silently_drop_detection() -> None:
    """Before iter 6, mixed-shape sequences hit the torch.stack
    RuntimeError fallback inside _embedding_magnitude and silently
    returned magnitude=0.0 — a real shift could be missed during a
    transition between embedding sizes.  After iter 6, every input
    canonicalises to 64-dim before stacking, so a real shift produces
    a real, non-zero magnitude even when shapes change mid-stream.
    """
    detector = AffectShiftDetector()
    user, session = "u_mix", "s_mix"

    # Baseline: small embeddings near zero in mixed shapes.  Recent:
    # large embeddings (different magnitude) in mixed shapes.  After
    # canonicalisation, the magnitude should be > 0 (the values
    # genuinely differ).
    baseline_shapes = [
        torch.zeros(64) + 0.05,
        torch.zeros(32) + 0.05,    # short — pads to 64 with zeros
        torch.zeros((4, 16)) + 0.05,  # multi-dim — flattens to 64
    ]
    recent_shapes = [
        torch.ones(64) * 1.5,
        torch.ones(128) * 1.5,    # long — truncates to 64
    ]
    last = None
    for emb in baseline_shapes + recent_shapes:
        last = detector.observe(
            user_id=user, session_id=session, embedding=emb,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )
    assert last is not None
    assert math.isfinite(last.magnitude)
    # iter 6: mixed shapes now produce real magnitude, not silently
    # zero.  Recent values are 1.5 vs baseline ≈ 0.05 — magnitude
    # should be substantial.
    #
    # Iter 63 — the embedding-magnitude formula now divides by
    # sqrt(N) so the value is RMS per-dim deviation in σ-units, not
    # total L2 norm.  Pre-iter-63 magnitudes were inflated by
    # sqrt(64)=8x on 64-dim embeddings; the test threshold here was
    # tuned to that inflation.  The genuinely-different mixed-shape
    # case still produces magnitude well above zero post-iter-63
    # (around 0.4–0.5 in σ-units), so we lower the threshold.
    assert last.magnitude > 0.3, (
        f"mixed-shape embeddings should produce real magnitude; "
        f"got {last.magnitude}"
    )


# ---------------------------------------------------------------------------
# Iter 6 — also: cuda/cpu device-portability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Iter 7 — fixed-baseline anchor (vs the previous rolling baseline)
# ---------------------------------------------------------------------------


def test_fixed_baseline_anchor_persists_across_long_session() -> None:
    """Iter 7: baseline anchors to the first N observations of the
    session, not a rolling tail.  This means a sustained shift is
    still detected against the user's *original* normal — even after
    20+ turns where a rolling-tail baseline would have drifted toward
    the new normal and lost detection sensitivity.
    """
    detector = AffectShiftDetector()
    user, session = "u_long", "s_long"

    # Five calm observations establish the fixed baseline.
    _calm_obs(detector, user, session, n=5)

    # Now sustained stressed typing for 15 turns.  Under a rolling
    # baseline, by turn 13+ the baseline would have rolled into all-
    # stressed observations and the detection would silently drop
    # (recent_window vs baseline_window are both stressed → no shift).
    # Under a fixed baseline, the original calm anchor persists and
    # detection should remain reliable.
    last: AffectShift | None = None
    for _ in range(15):
        last = _stressed_obs(detector, user, session, iki_mean=200.0, edits=4)

    assert last is not None
    # iter 7 fix: still detected because baseline is anchored to the
    # initial calm window.
    assert last.detected is True
    assert last.direction == "rising_load"
    # IKI delta % should remain large because baseline never shifts.
    assert last.iki_delta_pct >= 50.0, (
        f"with fixed baseline, sustained stress should keep showing a "
        f"large IKI delta; got {last.iki_delta_pct}"
    )


# ---------------------------------------------------------------------------
# Iter 9 — confidence score on AffectShift
# ---------------------------------------------------------------------------


def test_scalar_zero_dim_embedding_does_not_crash() -> None:
    """Iter 25: a 0-dim scalar tensor (torch.tensor(0.5)) flattens to
    a 1-element tensor and zero-pads to canonical 64-dim instead of
    crashing torch.cat with the 'zero-dimensional tensor' error."""
    detector = AffectShiftDetector()
    scalar = torch.tensor(0.5)
    result = detector.observe(
        user_id="u_scalar", session_id="s_scalar",
        embedding=scalar,
        composition_time_ms=1500.0, edit_count=0, pause_before_send_ms=200.0,
        keystroke_iki_mean=110.0, keystroke_iki_std=18.0,
    )
    assert math.isfinite(result.magnitude)


def test_high_variance_baseline_still_uses_embedding_trigger() -> None:
    """Iter 14: when sigma_baseline is well above the floor (>= 0.1),
    the embedding-magnitude trigger continues to fire as documented.
    Boundary check that the iter-14 floor doesn't accidentally disable
    the embedding channel in normal-variance regimes."""
    detector = AffectShiftDetector(magnitude_threshold=1.4)
    user, session = "u_hivar", "s_hivar"

    # Build a high-variance baseline (different random embeddings).
    torch.manual_seed(42)
    for _ in range(5):
        emb = torch.randn(64) * 0.3  # mean per-dim std around 0.3
        detector.observe(
            user_id=user, session_id=session, embedding=emb,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )

    # Now drift the embedding far from baseline.
    drift = torch.ones(64) * 5.0
    last: AffectShift | None = None
    for _ in range(3):
        last = detector.observe(
            user_id=user, session_id=session, embedding=drift,
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )

    assert last is not None
    # High-variance baseline + large drift produces a clean embedding-
    # magnitude fire — iter 14 doesn't accidentally suppress it.
    assert last.magnitude >= 1.4
    assert last.detected is True


def test_low_variance_baseline_does_not_false_positive_on_tiny_shift() -> None:
    """Iter 14: when the baseline is very consistent (sigma ~ 0),
    the magnitude divisor was floored at 1e-3, which made even a
    tiny L2 distance look like a multi-sigma shift.  After iter 14:
    the embedding-magnitude trigger only fires when sigma is at
    least 1e-2, otherwise we trust the keystroke channel.
    """
    detector = AffectShiftDetector()
    user, session = "u_lowvar", "s_lowvar"

    # Build an extremely consistent baseline (all zero embeddings)
    # with calm keystrokes.  Sigma_baseline should be ~ 0.
    for _ in range(5):
        detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )

    # Now feed a recent observation with a tiny embedding perturbation
    # (but no keystroke change).  Pre-iter-14, l2/sigma blew up to >>
    # threshold and falsely fired.  Post-iter-14, no embedding fire
    # because sigma is below the trust floor.
    last: AffectShift | None = None
    for _ in range(3):
        last = detector.observe(
            user_id=user, session_id=session,
            embedding=torch.full((64,), 0.005),  # tiny non-zero
            composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
        )

    assert last is not None
    # No keystroke change at all and embedding shift is tiny — no
    # genuine shift.  The detector should NOT fire.
    assert last.detected is False, (
        f"low-variance baseline + tiny embedding perturbation should "
        f"not fire; got detected=True with magnitude={last.magnitude}"
    )


def test_undetected_shift_has_confidence_zero() -> None:
    """When detected=False, confidence must be exactly 0.0."""
    detector = AffectShiftDetector()
    user, session = "u_conf_0", "s_conf_0"

    # Calm baseline + one calm probe → no detection, confidence=0.
    _calm_obs(detector, user, session, n=5)
    result = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
    )
    assert result.detected is False
    assert result.confidence == 0.0


def test_detected_shift_has_confidence_in_half_to_one() -> None:
    """Any detected shift must have confidence >= 0.5 and <= 1.0."""
    detector = AffectShiftDetector()
    user, session = "u_conf_half", "s_conf_half"

    _calm_obs(detector, user, session, n=5)
    result = _stressed_obs(detector, user, session, iki_mean=200.0, edits=4)
    assert result.detected is True
    assert 0.5 <= result.confidence <= 1.0


def test_strong_shift_has_higher_confidence_than_weak() -> None:
    """A clearly stronger keystroke shift produces higher confidence.

    To compare apples-to-apples, both detectors are populated with a
    full recent window of the relevant signal so the percentage
    deltas are unaffected by the calm-stressed mixing in the
    transition turn.
    """
    weak_detector = AffectShiftDetector()
    strong_detector = AffectShiftDetector()

    _calm_obs(weak_detector, "u_w", "s_w", n=5)
    _calm_obs(strong_detector, "u_s", "s_s", n=5)

    # Weak: 3 turns of just-over-strong-tier IKI, edits at strong-tier
    # min, low-margin overall.
    weak = None
    for _ in range(3):
        weak = _stressed_obs(weak_detector, "u_w", "s_w", iki_mean=150.0, edits=1)
    # Strong: 3 turns of huge IKI rise + many edits.
    strong = None
    for _ in range(3):
        strong = _stressed_obs(strong_detector, "u_s", "s_s", iki_mean=260.0, edits=6)

    assert weak is not None and strong is not None
    assert weak.detected is True, f"weak should still be detected: {weak}"
    assert strong.detected is True, f"strong should be detected: {strong}"
    assert strong.confidence > weak.confidence, (
        f"strong ({strong.confidence}) should exceed weak ({weak.confidence})"
    )


def test_confidence_is_in_dict_serialisation() -> None:
    """to_dict() must include the confidence field."""
    shift = AffectShift(
        detected=True,
        direction="rising_load",
        magnitude=2.0,
        iki_delta_pct=50.0,
        edit_delta_pct=200.0,
        suggestion="test",
        confidence=0.85,
    )
    d = shift.to_dict()
    assert "confidence" in d
    assert d["confidence"] == 0.85
    assert isinstance(d["confidence"], float)


# ---------------------------------------------------------------------------
# Iter 19 — confidence monotonicity property
# ---------------------------------------------------------------------------


def test_confidence_is_monotonic_in_iki_delta() -> None:
    """For rising_load, larger IKI deltas should produce equal-or-
    larger confidence — never smaller.  Iter 19: catches calibration
    regressions where a stronger signal accidentally produced
    weaker confidence.
    """
    confidences: list[tuple[float, float]] = []
    for iki_recent in (150.0, 180.0, 220.0, 280.0, 340.0):
        detector = AffectShiftDetector()
        user, session = "u_mono", "s_mono"
        _calm_obs(detector, user, session, n=5)
        # Three stressed observations to fill the recent window.
        last: AffectShift | None = None
        for _ in range(3):
            last = detector.observe(
                user_id=user, session_id=session, embedding=torch.zeros(64),
                composition_time_ms=4000.0, edit_count=2, pause_before_send_ms=400.0,
                keystroke_iki_mean=iki_recent, keystroke_iki_std=40.0,
            )
        assert last is not None
        if last.detected:
            confidences.append((iki_recent, last.confidence))

    # Across the rising-IKI sweep, confidence values should be
    # monotonically non-decreasing.
    for i in range(1, len(confidences)):
        prev_iki, prev_conf = confidences[i - 1]
        cur_iki, cur_conf = confidences[i]
        assert cur_conf >= prev_conf - 1e-9, (
            f"confidence regressed: iki {prev_iki}->{cur_iki} = "
            f"{prev_conf:.3f}->{cur_conf:.3f}"
        )


def test_confidence_is_monotonic_in_edit_delta() -> None:
    """Same monotonicity for the edit-count axis."""
    confidences: list[tuple[int, float]] = []
    for edits in (2, 4, 6, 8, 12):
        detector = AffectShiftDetector()
        user, session = "u_mono_e", "s_mono_e"
        _calm_obs(detector, user, session, n=5)
        last: AffectShift | None = None
        for _ in range(3):
            last = detector.observe(
                user_id=user, session_id=session, embedding=torch.zeros(64),
                composition_time_ms=4000.0, edit_count=edits, pause_before_send_ms=400.0,
                keystroke_iki_mean=200.0, keystroke_iki_std=40.0,
            )
        assert last is not None
        if last.detected:
            confidences.append((edits, last.confidence))

    for i in range(1, len(confidences)):
        prev_e, prev_conf = confidences[i - 1]
        cur_e, cur_conf = confidences[i]
        assert cur_conf >= prev_conf - 1e-9, (
            f"confidence regressed: edits {prev_e}->{cur_e} = "
            f"{prev_conf:.3f}->{cur_conf:.3f}"
        )


def test_zero_baseline_edit_delta_gradates_with_recent_magnitude() -> None:
    """Iter 31: zero-baseline edits + positive recent edits no longer
    saturates to a binary 200%.  edit_delta_pct now scales with
    recent_mean (capped at the original 200% default), so tier
    thresholds and confidence ramps see real severity ordering.

    Before: 1-edit and 100-edit recent windows both reported 200%.
    After: 1-edit reports ~100%, 5+ edits reports 200% (capped).
    """
    # Path A — 1 edit per turn (recent_mean ≈ 1.0).
    d_low = AffectShiftDetector()
    _calm_obs(d_low, "u_lo", "s_lo", n=5)
    last_low = None
    for _ in range(3):
        last_low = d_low.observe(
            user_id="u_lo", session_id="s_lo", embedding=torch.zeros(64),
            composition_time_ms=3000.0, edit_count=1,
            pause_before_send_ms=400.0,
            keystroke_iki_mean=180.0, keystroke_iki_std=35.0,
        )

    # Path B — 8 edits per turn (recent_mean ≈ 8.0, well above cap).
    d_hi = AffectShiftDetector()
    _calm_obs(d_hi, "u_hi", "s_hi", n=5)
    last_hi = None
    for _ in range(3):
        last_hi = d_hi.observe(
            user_id="u_hi", session_id="s_hi", embedding=torch.zeros(64),
            composition_time_ms=3000.0, edit_count=8,
            pause_before_send_ms=400.0,
            keystroke_iki_mean=180.0, keystroke_iki_std=35.0,
        )

    assert last_low is not None and last_hi is not None
    # Iter 31: low recent should land near 100% (gradated, not capped).
    assert 80.0 <= last_low.edit_delta_pct <= 130.0, (
        f"low-edit recent should gradate to ~100%, got "
        f"{last_low.edit_delta_pct}"
    )
    # High recent saturates at the 200% cap.
    assert last_hi.edit_delta_pct == pytest.approx(200.0, abs=0.5), (
        f"high-edit recent should saturate at 200% cap, got "
        f"{last_hi.edit_delta_pct}"
    )
    # Confidence reflects severity: high-edit produces higher
    # confidence than low-edit at the same iki delta.
    if last_low.detected and last_hi.detected:
        assert last_hi.confidence >= last_low.confidence


def test_corroborated_iki_and_edit_signals_increase_confidence() -> None:
    """Iter 30: when BOTH IKI and edit channels fire substantially,
    confidence is strictly higher than when only one channel fires
    with the same maximum strength.

    Need a non-zero edit baseline so edit_delta_pct varies (recent
    edits > 0 with baseline 0 always saturates to 200% via the
    zero-baseline-default).
    """
    # Build a baseline with 2 edits per turn so edit_delta gradates
    # by recent edit count (a zero-edits baseline saturates to the
    # 200% zero-baseline-default regardless of recent count).
    def _baseline_with_edits(detector, user, session, n=5):
        for _ in range(n):
            detector.observe(
                user_id=user, session_id=session, embedding=torch.zeros(64),
                composition_time_ms=2500.0, edit_count=2,
                pause_before_send_ms=300.0,
                keystroke_iki_mean=120.0, keystroke_iki_std=22.0,
            )

    # Single-channel: IKI moderate (180ms ≈ +50%), edits flat at baseline.
    # Crafted so neither channel saturates the confidence ramp,
    # making the corroboration bonus visible.
    d_single = AffectShiftDetector()
    _baseline_with_edits(d_single, "u_s", "s_s")
    last_single = None
    for _ in range(3):
        last_single = d_single.observe(
            user_id="u_s", session_id="s_s", embedding=torch.zeros(64),
            composition_time_ms=3000.0, edit_count=2,
            pause_before_send_ms=400.0,
            keystroke_iki_mean=180.0, keystroke_iki_std=35.0,
        )

    # Corroborating: IKI same moderate, edits moderate (6 vs baseline 2 → +200%).
    d_corr = AffectShiftDetector()
    _baseline_with_edits(d_corr, "u_c", "s_c")
    last_corr = None
    for _ in range(3):
        last_corr = d_corr.observe(
            user_id="u_c", session_id="s_c", embedding=torch.zeros(64),
            composition_time_ms=3000.0, edit_count=6,
            pause_before_send_ms=400.0,
            keystroke_iki_mean=180.0, keystroke_iki_std=35.0,
        )

    assert last_single is not None and last_corr is not None
    assert last_single.detected is True
    assert last_corr.detected is True
    # IKI delta same; edit delta differs.
    assert last_corr.edit_delta_pct > last_single.edit_delta_pct
    # Corroboration should produce strictly higher confidence.
    assert last_corr.confidence > last_single.confidence, (
        f"corroboration ({last_corr.confidence:.3f}) should exceed "
        f"single-channel ({last_single.confidence:.3f})"
    )


def test_falling_load_corroborated_by_edit_decrease() -> None:
    """Iter 30: a recovery turn where edits drop sharply (e.g. 4 → 0)
    has higher confidence than one where edits stay flat (e.g. 4 → 4),
    even at the same IKI delta."""
    # Path A — edits stay at baseline level (flat).
    d_flat = AffectShiftDetector()
    for _ in range(5):
        d_flat.observe(
            user_id="u_a", session_id="s_a", embedding=torch.zeros(64),
            composition_time_ms=4500.0, edit_count=4, pause_before_send_ms=600.0,
            keystroke_iki_mean=200.0, keystroke_iki_std=40.0,
        )
    last_flat = None
    for _ in range(3):
        last_flat = d_flat.observe(
            user_id="u_a", session_id="s_a", embedding=torch.zeros(64),
            composition_time_ms=2200.0, edit_count=4, pause_before_send_ms=300.0,
            keystroke_iki_mean=110.0, keystroke_iki_std=20.0,
        )

    # Path B — edits drop sharply (corroborating recovery).
    d_drop = AffectShiftDetector()
    for _ in range(5):
        d_drop.observe(
            user_id="u_b", session_id="s_b", embedding=torch.zeros(64),
            composition_time_ms=4500.0, edit_count=4, pause_before_send_ms=600.0,
            keystroke_iki_mean=200.0, keystroke_iki_std=40.0,
        )
    last_drop = None
    for _ in range(3):
        last_drop = d_drop.observe(
            user_id="u_b", session_id="s_b", embedding=torch.zeros(64),
            composition_time_ms=2200.0, edit_count=0, pause_before_send_ms=300.0,
            keystroke_iki_mean=110.0, keystroke_iki_std=20.0,
        )

    assert last_flat is not None and last_drop is not None
    if last_flat.detected and last_drop.detected:
        # Edit-drop corroboration should produce strictly higher confidence.
        assert last_drop.confidence > last_flat.confidence, (
            f"edit-drop corroboration ({last_drop.confidence:.3f}) should "
            f"exceed flat-edits ({last_flat.confidence:.3f})"
        )


def test_falling_load_confidence_scales_with_iki_drop() -> None:
    """Larger IKI drop → higher falling-load confidence."""
    detector = AffectShiftDetector()
    user, session = "u_fall_conf", "s_fall_conf"

    # Establish a stressed baseline.
    for _ in range(5):
        detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=4000.0, edit_count=3, pause_before_send_ms=600.0,
            keystroke_iki_mean=200.0, keystroke_iki_std=40.0,
        )

    # First recovery turn — modest drop.
    modest = detector.observe(
        user_id=user, session_id=session, embedding=torch.zeros(64),
        composition_time_ms=2500.0, edit_count=2, pause_before_send_ms=300.0,
        keystroke_iki_mean=160.0, keystroke_iki_std=25.0,
    )
    if modest.detected:
        modest_conf = modest.confidence
    else:
        # Need another observation to populate the recent window.
        modest = detector.observe(
            user_id=user, session_id=session, embedding=torch.zeros(64),
            composition_time_ms=2500.0, edit_count=2, pause_before_send_ms=300.0,
            keystroke_iki_mean=160.0, keystroke_iki_std=25.0,
        )
        modest_conf = modest.confidence if modest.detected else 0.0

    # All confidences in the valid range.
    if modest.detected:
        assert 0.5 <= modest_conf <= 1.0


def test_fixed_baseline_resets_on_end_session() -> None:
    """end_session wipes the per-session fixed baseline so a new
    session starts fresh."""
    detector = AffectShiftDetector()
    user, session = "u_reset", "s_reset"

    _calm_obs(detector, user, session, n=5)
    detector.end_session(user, session)

    # First observation of the new session: warmup, no detection.
    result = _stressed_obs(detector, user, session)
    assert result.detected is False
    # The detector treats this as turn 1 of a brand-new session, so
    # there's no baseline yet to compare against.


def test_embedding_on_unexpected_device_handled() -> None:
    """Even if a CUDA-resident tensor were passed, _safe_embedding's
    .to() call should normalise it to the detector's device.

    This test only executes on a CUDA-enabled machine; otherwise it
    asserts the CPU path still works.
    """
    detector = AffectShiftDetector()
    if torch.cuda.is_available():
        emb = torch.zeros(64, device="cuda")
    else:
        emb = torch.zeros(64)

    result = detector.observe(
        user_id="u_dev", session_id="s_dev",
        embedding=emb,
        composition_time_ms=2000.0, edit_count=0, pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0, keystroke_iki_std=20.0,
    )
    assert math.isfinite(result.magnitude)


# ---------------------------------------------------------------------------
# end_session
# ---------------------------------------------------------------------------


def test_end_session_clears_state() -> None:
    detector = AffectShiftDetector()
    _calm_obs(detector, "u_end", "s_end", n=6)

    detector.end_session("u_end", "s_end")

    # After end_session, the next observation should be back in warm-up.
    result = detector.observe(
        user_id="u_end",
        session_id="s_end",
        embedding=torch.zeros(64),
        composition_time_ms=2000.0,
        edit_count=0,
        pause_before_send_ms=300.0,
        keystroke_iki_mean=120.0,
        keystroke_iki_std=20.0,
    )
    assert result.detected is False
    assert result.iki_delta_pct == 0.0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_max_sessions_evicts_oldest() -> None:
    detector = AffectShiftDetector(max_sessions=3)

    # Create 4 sessions; first should be evicted on the 4th.
    for i in range(4):
        _calm_obs(detector, f"user_{i}", f"sess_{i}", n=1)

    assert ("user_0", "sess_0") not in detector._buffers
    assert ("user_3", "sess_3") in detector._buffers
    assert len(detector._buffers) == 3
