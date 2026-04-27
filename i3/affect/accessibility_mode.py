"""Accessibility-mode auto-switch state machine.

Second deliverable of the Live State Badge / Accessibility Mode
showpiece for the Huawei R&D UK / HMI Lab pitch.  Where the state
classifier paints a discrete label every turn, this controller tracks
a *sticky* per-session bool: when sustained motor-difficulty /
dyslexia signals are observed we activate accessibility mode, and we
only deactivate after a recovery window of consistently calmer turns.

Why "sticky"
~~~~~~~~~~~~
A naive turn-by-turn rule would flap on a single calmer turn and
yank the font scale, vocabulary, and TTS rate around mid-session.
Real motor / cognitive accessibility needs are stable over minutes —
the state machine therefore requires *N consecutive elevated turns*
to activate and *M consecutive calm turns* to deactivate, with
N=3 and M=4 by default.  This matches the WCAG 2.x guidance to
prefer infrequent state changes over jittery responsiveness.

Activation rule
~~~~~~~~~~~~~~~
Sustained over the last 3 turns:

* mean ``edit_count`` ≥ 1.5, AND
* mean ``iki_mean`` > 1.4× the user's session ``baseline_iki``, AND
* reading-level proxy (recent response token entropy / cognitive_load
  proxy) ≤ a low threshold (i.e. the user is producing terse,
  low-complexity input, consistent with motor difficulty),

OR an explicit ``force=True`` from the manual toggle endpoint.

Deactivation rule
~~~~~~~~~~~~~~~~~
After 4 consecutive turns where ALL of:

* edit_count ≤ 1, AND
* iki_mean within ±15% of baseline, AND
* normal pause-before-send (cognitive_load < 0.55).

Implementation discipline
~~~~~~~~~~~~~~~~~~~~~~~~~
* Pure Python.  No torch.  CPU-only.
* 1000-session LRU cap so a churning multi-tenant server cannot grow
  unbounded.
* Per-session entry retains ``last_active`` so the controller can
  emit rising-edge / falling-edge flags for the UI animation.
* Every per-turn observation is bounded in ``[0, ∞)`` so a hostile
  client cannot poison the rolling means with absurd values.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class AccessibilityModeState:
    """Result of one :meth:`AccessibilityController.observe` call.

    Attributes:
        active: Whether accessibility mode is currently engaged.
        activated_this_turn: Rising edge — became active on this turn.
        deactivated_this_turn: Falling edge — became inactive on this
            turn.
        confidence: ``[0, 1]`` confidence in the classification.
            Computed from the strength of the rolling-window
            elevated-turn evidence.
        reason: Human-readable explanation, e.g. "elevated edit-rate
            sustained over 3 turns", surfaced in the UI strip.
        sentence_cap: Hard cap on response sentences while active —
            1 / 2 / 3 depending on signal severity.  Always 3 when
            inactive.
        simplify_vocab: Always True while active; the engine uses
            this to push ``adaptation.accessibility`` toward 1.0.
        tts_rate_multiplier: 0.6 while active, 1.0 normal.  Read by
            the front-end TTS player.
        font_scale: 1.25 while active, 1.0 normal.  Read by the
            front-end CSS layer to grow chat bubbles.
    """

    active: bool
    activated_this_turn: bool = False
    deactivated_this_turn: bool = False
    confidence: float = 0.0
    reason: str = ""
    sentence_cap: int = 3
    simplify_vocab: bool = False
    tts_rate_multiplier: float = 1.0
    font_scale: float = 1.0

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the WS layer."""
        return {
            "active": bool(self.active),
            "activated_this_turn": bool(self.activated_this_turn),
            "deactivated_this_turn": bool(self.deactivated_this_turn),
            "confidence": float(self.confidence),
            "reason": str(self.reason),
            "sentence_cap": int(self.sentence_cap),
            "simplify_vocab": bool(self.simplify_vocab),
            "tts_rate_multiplier": float(self.tts_rate_multiplier),
            "font_scale": float(self.font_scale),
        }


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    """One turn's worth of metrics retained in the per-session ring."""

    edit_count: float
    iki_mean: float
    iki_std: float
    cognitive_load: float
    accessibility_axis: float


@dataclass
class _SessionState:
    """All per-session bookkeeping for the controller."""

    turns: Deque[_Turn] = field(default_factory=lambda: deque(maxlen=6))
    active: bool = False
    last_active: bool = False  # for rising / falling edge detection
    forced: bool | None = None  # None = auto, True/False = manual override
    last_reason: str = ""
    # Per-session running mean of iki_mean over the first ~5 turns,
    # used as the baseline against which we measure "1.4× elevation".
    # Updated lazily as the user types so the baseline is always
    # *that user's* normal, not a fleet average.
    baseline_iki: float = 0.0
    baseline_samples: int = 0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class AccessibilityController:
    """State machine for accessibility-mode auto-activation.

    Activation rule
    ---------------
    Sustained over the last 3 turns:
      - mean ``edit_count`` >= 1.5 per turn, AND
      - mean ``iki_mean`` > 1.4× the user's session baseline IKI, AND
      - reading-level proxy (low ``cognitive_load`` mean ≥ 0.55,
        or explicit ``accessibility_axis`` mean ≥ 0.4) — i.e. the
        adaptation layer agrees the user is producing terse / simple
        input,
    OR explicit ``force=True`` (user clicked the manual toggle).

    Deactivation rule
    -----------------
    After 4 consecutive turns where ALL of:
      - ``edit_count`` <= 1,
      - ``iki_mean`` within ±15% of baseline,
      - ``cognitive_load`` < 0.55 (normal pause-before-send proxy).

    Once active, it remains active until the deactivation rule fires —
    i.e. the mode is "sticky" so it doesn't flap on a single calmer
    turn.

    LRU-bounded at 1000 sessions.
    """

    # Activation thresholds.
    _ACT_WINDOW: int = 3                  # consecutive elevated turns
    _ACT_MEAN_EDITS: float = 1.5          # mean edits over window
    _ACT_IKI_RATIO: float = 1.4           # IKI > 1.4× baseline
    _ACT_COGLOAD_MIN: float = 0.55        # cognitive_load floor
    _ACT_ACCESS_MIN: float = 0.40         # accessibility-axis floor

    # Deactivation thresholds.
    _DEACT_WINDOW: int = 4                # consecutive calm turns
    _DEACT_MAX_EDITS: float = 1.0
    _DEACT_IKI_TOL: float = 0.15          # ±15% of baseline
    _DEACT_COGLOAD_MAX: float = 0.55

    # Per-user max sessions tracked.
    _MAX_SESSIONS: int = 1000

    # Tunables for the activation strength → adapt-knob translation.
    _AGGRESSIVE_EDIT_THR: float = 3.5     # mean edits ≥ 3.5 → cap=1
    _MODERATE_EDIT_THR: float = 2.0       # mean edits ≥ 2.0 → cap=2

    def __init__(self, max_sessions: int | None = None) -> None:
        self.max_sessions = int(max_sessions or self._MAX_SESSIONS)
        self._sessions: OrderedDict[tuple[str, str], _SessionState] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        user_id: str,
        session_id: str,
        *,
        edit_count: float,
        iki_mean: float,
        iki_std: float,
        baseline_iki: float | None = None,
        cognitive_load: float = 0.5,
        accessibility_axis: float = 0.0,
        force: bool | None = None,
    ) -> AccessibilityModeState:
        """Push one turn of metrics, run the state machine, return the new state.

        Args:
            user_id: Opaque user identifier.
            session_id: Opaque session identifier.
            edit_count: Backspaces + deletes during composition.
            iki_mean: Mean inter-keystroke interval (ms) for this turn.
            iki_std: Std-dev of inter-keystroke intervals (ms).
            baseline_iki: Optional override for the user's IKI
                baseline.  When ``None`` (the common case) the
                controller maintains its own baseline as a running
                mean of the first 5 turns of the session.
            cognitive_load: Adaptation cognitive_load axis.
            accessibility_axis: Adaptation accessibility axis.
            force: Manual override.  ``True`` activates immediately
                (sticky), ``False`` deactivates immediately, ``None``
                lets the auto rule decide.  Once a forced value is
                provided it is remembered for subsequent ``observe``
                calls so a single ``force=True`` doesn't get undone
                by the very next auto-eval.

        Returns:
            A fresh :class:`AccessibilityModeState`.
        """
        key = (str(user_id), str(session_id))
        state = self._get_or_create(key)

        # --- Defensive coercion ---------------------------------------
        ec = max(0.0, _safe_float(edit_count))
        im = max(0.0, _safe_float(iki_mean))
        is_ = max(0.0, _safe_float(iki_std))
        cl = max(0.0, min(1.0, _safe_float(cognitive_load, 0.5)))
        ax = max(0.0, min(1.0, _safe_float(accessibility_axis, 0.0)))
        baseline_override = (
            None
            if baseline_iki is None
            else max(0.0, _safe_float(baseline_iki, 0.0))
        )

        # --- Manual override sticks until the next manual call --------
        if force is not None:
            state.forced = bool(force)

        # --- Update the per-session baseline IKI (running mean of
        #     up to 5 *quiet* turns the controller saw). --------------
        # Quiet = edit_count ≤ 1 AND cognitive_load below the
        # deactivation ceiling.  Restricting baseline learning to
        # quiet turns prevents a few panicky early turns from
        # poisoning the running mean and locking the user into
        # accessibility mode forever (a real bug surfaced by the
        # 13-turn probe).
        is_quiet = (
            ec <= self._DEACT_MAX_EDITS
            and cl < self._DEACT_COGLOAD_MAX
        )
        if baseline_override is not None and baseline_override > 0:
            state.baseline_iki = baseline_override
            # Treat an externally-supplied baseline as authoritative.
            state.baseline_samples = max(state.baseline_samples, 1)
        elif is_quiet and state.baseline_samples < 5 and im > 0:
            # Incremental running mean over quiet turns only.
            n = state.baseline_samples + 1
            state.baseline_iki = (
                (state.baseline_iki * state.baseline_samples) + im
            ) / n
            state.baseline_samples = n

        # --- Append to the rolling window -----------------------------
        state.turns.append(
            _Turn(
                edit_count=ec,
                iki_mean=im,
                iki_std=is_,
                cognitive_load=cl,
                accessibility_axis=ax,
            )
        )

        prev_active = state.last_active

        # --- Decide the new activation state --------------------------
        if state.forced is True:
            state.active = True
            state.last_reason = "manual override (force=True)"
            confidence = 1.0
        elif state.forced is False:
            state.active = False
            state.last_reason = "manual override (force=False)"
            confidence = 1.0
        else:
            # Auto rule.
            new_active, reason, confidence = self._auto_rule(state)
            state.active = new_active
            if reason:
                state.last_reason = reason

        rising = (not prev_active) and state.active
        falling = prev_active and (not state.active)
        state.last_active = state.active

        return self._build_state(
            state,
            confidence=confidence,
            rising=rising,
            falling=falling,
        )

    def end_session(self, user_id: str, session_id: str) -> None:
        """Drop all state for a session.  Idempotent."""
        key = (str(user_id), str(session_id))
        self._sessions.pop(key, None)

    def current(self, user_id: str, session_id: str) -> AccessibilityModeState | None:
        """Return the most recent state for a session, or ``None``.

        Used by the manual-toggle endpoint to read back the post-
        toggle state without forcing another ``observe`` cycle.
        """
        key = (str(user_id), str(session_id))
        state = self._sessions.get(key)
        if state is None:
            return None
        # Touch for LRU ordering.
        self._sessions.move_to_end(key)
        return self._build_state(state, confidence=1.0, rising=False, falling=False)

    def force(
        self,
        user_id: str,
        session_id: str,
        *,
        force: bool | None,
    ) -> AccessibilityModeState:
        """Apply a manual override and return the resulting state.

        Implemented as a thin wrapper over :meth:`observe` with
        zero-valued metrics so the manual toggle can be triggered
        without the user having typed yet.  The pre-existing
        per-session rolling window is preserved.
        """
        key = (str(user_id), str(session_id))
        state = self._get_or_create(key)

        prev_active = state.last_active

        if force is None:
            # Clearing the manual override: re-evaluate via the auto rule.
            # When the previously-forced active state has no rolling
            # data to justify itself, drop back to inactive cleanly.
            state.forced = None
            if not state.turns:
                state.active = False
                state.last_reason = ""
                confidence = 1.0
            else:
                new_active, reason, confidence = self._auto_rule(state)
                state.active = new_active
                if reason:
                    state.last_reason = reason
        else:
            state.forced = bool(force)
            state.active = bool(force)
            state.last_reason = (
                "manual override (force=True)"
                if force
                else "manual override (force=False)"
            )
            confidence = 1.0

        rising = (not prev_active) and state.active
        falling = prev_active and (not state.active)
        state.last_active = state.active

        return self._build_state(
            state,
            confidence=confidence,
            rising=rising,
            falling=falling,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, key: tuple[str, str]) -> _SessionState:
        """Fetch (or create + LRU-evict) the per-session state."""
        existing = self._sessions.get(key)
        if existing is not None:
            self._sessions.move_to_end(key)
            return existing
        while len(self._sessions) >= self.max_sessions:
            evicted_key, _ = self._sessions.popitem(last=False)
            logger.debug(
                "AccessibilityController evicted oldest session: user=%s session=%s",
                evicted_key[0],
                evicted_key[1],
            )
        new_state = _SessionState()
        self._sessions[key] = new_state
        return new_state

    def _auto_rule(
        self, state: _SessionState
    ) -> tuple[bool, str, float]:
        """Apply the activation / deactivation rule.

        Returns:
            ``(new_active, reason, confidence)``.
        """
        turns = list(state.turns)
        if not turns:
            return state.active, state.last_reason, 0.0

        # ---- Activation evaluation -----------------------------------
        if not state.active:
            window = turns[-self._ACT_WINDOW :]
            if len(window) < self._ACT_WINDOW:
                return False, state.last_reason, 0.0
            mean_edits = sum(t.edit_count for t in window) / len(window)
            mean_iki = sum(t.iki_mean for t in window) / len(window)
            mean_cl = sum(t.cognitive_load for t in window) / len(window)
            mean_ax = sum(t.accessibility_axis for t in window) / len(window)

            base = state.baseline_iki if state.baseline_iki > 0 else max(mean_iki, 1.0)
            iki_ratio = mean_iki / base if base > 0 else 1.0

            elevated_edits = mean_edits >= self._ACT_MEAN_EDITS
            elevated_iki = iki_ratio >= self._ACT_IKI_RATIO
            simple_input = (
                mean_cl >= self._ACT_COGLOAD_MIN
                or mean_ax >= self._ACT_ACCESS_MIN
            )

            if elevated_edits and elevated_iki and simple_input:
                # Confidence is the geometric mean of how far each
                # signal exceeded its threshold (capped at 1).  A
                # near-miss case scores ~0.5; a strongly elevated
                # case scores ≥ 0.85.
                e_score = min(1.0, mean_edits / (self._ACT_MEAN_EDITS * 2))
                i_score = min(1.0, (iki_ratio - 1.0) / (self._ACT_IKI_RATIO * 0.6))
                c_score = min(
                    1.0,
                    max(
                        mean_cl / (self._ACT_COGLOAD_MIN * 1.5),
                        mean_ax / max(self._ACT_ACCESS_MIN, 1e-3),
                    ),
                )
                confidence = max(0.0, min(1.0, (e_score * i_score * c_score) ** (1.0 / 3.0)))
                reason = (
                    f"elevated edit-rate sustained over {self._ACT_WINDOW} turns "
                    f"(mean edits {mean_edits:.1f}, IKI +{(iki_ratio - 1) * 100:.0f}% vs baseline)"
                )
                return True, reason, confidence
            return False, state.last_reason, 0.0

        # ---- Deactivation evaluation ---------------------------------
        window = turns[-self._DEACT_WINDOW :]
        if len(window) < self._DEACT_WINDOW:
            return True, state.last_reason, 0.7

        base = state.baseline_iki if state.baseline_iki > 0 else 0.0
        all_calm = True
        for t in window:
            if t.edit_count > self._DEACT_MAX_EDITS:
                all_calm = False
                break
            if base > 0:
                ratio = abs(t.iki_mean - base) / base
                if ratio > self._DEACT_IKI_TOL:
                    all_calm = False
                    break
            if t.cognitive_load >= self._DEACT_COGLOAD_MAX:
                all_calm = False
                break

        if all_calm:
            reason = (
                f"recovery window: {self._DEACT_WINDOW} consecutive calm turns "
                f"(edits ≤ 1, IKI within ±{int(self._DEACT_IKI_TOL * 100)}% of baseline)"
            )
            return False, reason, 0.85

        # Still elevated — stay active.  Confidence reflects how many
        # turns of the recovery window were calm.
        calm_count = 0
        for t in window:
            if t.edit_count <= self._DEACT_MAX_EDITS:
                calm_count += 1
        confidence = max(0.55, 1.0 - (calm_count / max(1, self._DEACT_WINDOW)) * 0.3)
        return True, state.last_reason, confidence

    def _build_state(
        self,
        state: _SessionState,
        *,
        confidence: float,
        rising: bool,
        falling: bool,
    ) -> AccessibilityModeState:
        """Translate the internal _SessionState into the public dataclass."""
        if state.active:
            # Use the most recent rolling-window mean to pick how
            # aggressively to truncate.  Falls back to "moderate" when
            # the window has fewer than 3 turns.
            turns = list(state.turns)[-self._ACT_WINDOW :]
            mean_edits = (
                sum(t.edit_count for t in turns) / len(turns) if turns else 0.0
            )
            if mean_edits >= self._AGGRESSIVE_EDIT_THR:
                sentence_cap = 1
            elif mean_edits >= self._MODERATE_EDIT_THR:
                sentence_cap = 2
            else:
                sentence_cap = 3
            return AccessibilityModeState(
                active=True,
                activated_this_turn=rising,
                deactivated_this_turn=False,
                confidence=float(max(0.0, min(1.0, confidence))),
                reason=state.last_reason,
                sentence_cap=sentence_cap,
                simplify_vocab=True,
                tts_rate_multiplier=0.6,
                font_scale=1.25,
            )
        return AccessibilityModeState(
            active=False,
            activated_this_turn=False,
            deactivated_this_turn=falling,
            confidence=float(max(0.0, min(1.0, confidence))),
            reason=state.last_reason,
            sentence_cap=3,
            simplify_vocab=False,
            tts_rate_multiplier=1.0,
            font_scale=1.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | int, default: float = 0.0) -> float:
    """Coerce *value* to a finite float with fallback."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


__all__ = [
    "AccessibilityModeState",
    "AccessibilityController",
]
