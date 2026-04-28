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
    assert last.magnitude > 0.5, (
        f"mixed-shape embeddings should produce real magnitude; got {last.magnitude}"
    )


# ---------------------------------------------------------------------------
# Iter 6 — also: cuda/cpu device-portability
# ---------------------------------------------------------------------------


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
