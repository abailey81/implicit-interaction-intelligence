"""Mid-conversation affect-shift detection from keystroke dynamics.

Why this is the I3 pitch piece (Huawei R&D UK / HMI Lab brief)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The TCN encoder produces a 64-dim user-state embedding per turn.  As
the user keeps typing, that embedding moves through state-space.  An
**affect shift** is when the embedding's L2 distance to the user's
recent-baseline mean exceeds a threshold — interpreted as "the user
just changed mood / cognitive load / fatigue mid-conversation".

When detected, the assistant proactively appends a short, polite
check-in to the response (e.g. "I notice you've been typing 30%
slower with more edits over the last 2 minutes — would you like me
to break this answer into shorter pieces?").  This is the kind of
behaviour that *only* a model watching keystrokes can do; ChatGPT
cannot.  It is the one-sentence pitch I3 walks into the Huawei
interview with.

Two-signal detector (robust to encoder warm-up noise)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The TCN can produce somewhat noisy 64-dim embeddings during the
session warm-up window (≤ 5 turns).  Using the embedding alone as
the trigger therefore *misfires* on early turns and *misses* late
shifts where the embedding has already drifted.  The detector here
therefore fuses two signals:

1. **Embedding magnitude** (primary, when warm enough):
   ``magnitude = L2(recent - baseline) / max(σ_baseline, 1e-3)``.
2. **Keystroke metrics** (always available; primary fallback):
   ``IKI_recent / IKI_baseline > 1.2`` AND ``edits_recent /
   edits_baseline > 1.5`` for ``rising_load``; the symmetric
   condition for ``falling_load``.

A shift fires when **either** the embedding crosses
``magnitude_threshold`` (default 1.4σ) *or* the keystroke condition
is met.  In practice the keystroke heuristic is the one that fires
reliably during the demo, with the embedding magnitude acting as a
sanity-check / corroboration.

Direction inference
~~~~~~~~~~~~~~~~~~~
Direction (``rising_load`` / ``falling_load`` / ``neutral``) is
derived from the keystroke metrics, *not* from the embedding alone:

* **rising_load**: IKI mean +20% vs baseline AND/OR edits +50%.
* **falling_load**: IKI mean -15% vs baseline AND edits flat or
  decreasing.
* **neutral**: shift detected but no clean directional signal
  (e.g. embedding moved but keystroke metrics did not).

State management
~~~~~~~~~~~~~~~~
Per-session rolling history of the last ``window_size=8``
observations, keyed on ``(user_id, session_id)``.  The map is
LRU-bounded at 1000 sessions so a long-running multi-tenant server
cannot grow unbounded.  Sessions can be reset with
:meth:`AffectShiftDetector.end_session`.

Constraints
~~~~~~~~~~~
* Pure Python + torch, no new deps, CPU-only.
* The detector NEVER changes the routing decision or the SLM call
  — it only inspects post-hoc and decorates the response.
* Single-turn / no-history flows still work: the detector simply
  reports ``detected=False`` until enough observations accumulate.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suggestion banks
# ---------------------------------------------------------------------------

RISING_LOAD_SUGGESTIONS: tuple[str, ...] = (
    "I notice your typing has slowed and you've made more edits over the past few turns — would you like me to break this answer into shorter pieces?",
    "Your typing pattern shifted toward more deliberate / corrected entries. Want me to keep responses tighter?",
    "Picking up a rising-load signal — happy to switch to one-sentence answers if that's easier.",
)
"""Templates used when ``direction == 'rising_load'``."""

FALLING_LOAD_SUGGESTIONS: tuple[str, ...] = (
    "Your typing pattern just shifted — happy to dig deeper if you've got more bandwidth now.",
    "Looks like you're moving faster — want me to expand on any of this with more detail?",
    "Sensing more headroom on your end — let me know if you want a longer answer.",
)
"""Templates used when ``direction == 'falling_load'``."""

NEUTRAL_SHIFT_SUGGESTIONS: tuple[str, ...] = (
    "Your interaction pattern shifted notably from the start of this session — let me know if I should adjust how I'm answering.",
)
"""Templates used when a shift is detected but direction is ambiguous."""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class AffectShift:
    """Result of a single :meth:`AffectShiftDetector.observe` call.

    Attributes:
        detected: Whether a meaningful affect shift was detected on
            this turn.  ``False`` during session warm-up or when the
            shift is below threshold.
        direction: One of ``"rising_load"``, ``"falling_load"``,
            ``"neutral"``.  Inferred from keystroke metrics, not the
            embedding.
        magnitude: Embedding-derived shift magnitude in σ units
            (``L2(recent - baseline) / σ_baseline``).  ``0.0`` if not
            enough observations to compute a baseline.
        iki_delta_pct: Mean inter-keystroke interval in the recent
            window vs the baseline window, expressed as a signed
            percentage (e.g. ``+25.0`` = 25% slower).  ``0.0`` if no
            baseline yet.
        edit_delta_pct: Edit count, recent vs baseline, signed %.
        suggestion: The proactive suggestion text to append to the
            assistant's response, or ``""`` if no shift / debounced.
    """

    detected: bool
    direction: str
    magnitude: float
    iki_delta_pct: float
    edit_delta_pct: float
    suggestion: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the WebSocket layer."""
        return {
            "detected": bool(self.detected),
            "direction": str(self.direction),
            "magnitude": float(self.magnitude),
            "iki_delta_pct": float(self.iki_delta_pct),
            "edit_delta_pct": float(self.edit_delta_pct),
            "suggestion": str(self.suggestion),
        }


# ---------------------------------------------------------------------------
# Per-session ring buffer
# ---------------------------------------------------------------------------


@dataclass
class _Observation:
    """One turn's worth of signals retained in the per-session ring."""

    embedding: torch.Tensor          # (64,) float32
    composition_time_ms: float
    edit_count: int
    pause_before_send_ms: float
    keystroke_iki_mean: float
    keystroke_iki_std: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AffectShiftDetector:
    """Tracks per-user rolling embedding history and detects affect shifts.

    See the module-level docstring for the full theoretical background.

    Window logic
    ------------
    * Keep the last ``window_size=8`` observations per
      ``(user_id, session_id)``.
    * **Baseline** = the *first* min(N, len) observations the
      detector saw in this session (the warm-up window — anchors
      "what does this user normally look like?").
    * **Recent** = the last ``recent_size=3`` observations (the
      probe window — "what are they doing right now?").
    * Embedding shift magnitude = ``L2(recent_mean - baseline_mean) /
      max(σ_baseline, 1e-3)``.
    * Threshold: ``magnitude >= magnitude_threshold`` (default 1.4σ).

    The detector's per-session state is stored in an ``OrderedDict``
    of size ``max_sessions`` (LRU-evicted) so a long-running server
    cannot grow unbounded if clients churn session ids.
    """

    # Direction thresholds — keystroke-only.
    _RISING_IKI_PCT: float = 20.0      # IKI must rise ≥ +20% over baseline
    _RISING_EDIT_PCT: float = 50.0     # OR edits must rise ≥ +50%
    _FALLING_IKI_PCT: float = -15.0    # IKI must drop ≤ -15%
    # Falling: edits must be flat or decreasing (≤ +5% jitter is allowed).
    _FALLING_EDIT_MAX_PCT: float = 5.0

    # Pure-keystroke fallback (fires even when embedding magnitude is
    # below threshold — the brief calls this "the one that fires
    # reliably during the demo").
    _KS_RISING_IKI_PCT: float = 20.0    # 1.2× ratio
    _KS_RISING_EDIT_PCT: float = 50.0   # 1.5× ratio

    # Debounce: don't append a suggestion more than once every
    # ``_DEBOUNCE_TURNS`` turns of the same session.
    _DEBOUNCE_TURNS: int = 4

    def __init__(
        self,
        window_size: int = 8,
        recent_size: int = 3,
        magnitude_threshold: float = 1.4,
        max_sessions: int = 1000,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if recent_size < 1 or recent_size >= window_size:
            raise ValueError("recent_size must satisfy 1 <= recent_size < window_size")
        self.window_size = int(window_size)
        self.recent_size = int(recent_size)
        self.magnitude_threshold = float(magnitude_threshold)
        self.max_sessions = int(max_sessions)

        # Per-session ring buffers + counters.
        self._buffers: OrderedDict[tuple[str, str], Deque[_Observation]] = OrderedDict()
        self._turn_index: dict[tuple[str, str], int] = {}
        # Number of turns since the last suggestion was emitted on
        # this session.  ``-1`` means "never emitted yet" — we always
        # allow the first.
        self._turns_since_suggestion: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        user_id: str,
        session_id: str,
        *,
        embedding: torch.Tensor,
        composition_time_ms: float,
        edit_count: int,
        pause_before_send_ms: float,
        keystroke_iki_mean: float,
        keystroke_iki_std: float,
    ) -> AffectShift:
        """Append an observation and return whether a shift was detected.

        Args:
            user_id: Opaque user identifier.
            session_id: Opaque session identifier.  Resetting the
                session via :meth:`end_session` wipes the rolling
                history.
            embedding: 64-dim user-state embedding from the TCN
                encoder.  Higher-dim tensors are flattened; ``None``
                or wrong-rank inputs are coerced to a zero vector.
            composition_time_ms: Total composition time for this
                turn (ms).  Reported by the client.
            edit_count: Number of edits (backspaces + cuts) during
                composition.
            pause_before_send_ms: Hesitation between last keystroke
                and send.
            keystroke_iki_mean: Mean inter-keystroke interval over
                this turn, in milliseconds.
            keystroke_iki_std: Std-dev of inter-keystroke intervals
                over this turn.

        Returns:
            An :class:`AffectShift`.  ``detected=False`` is the
            normal case during warm-up.  Same inputs always produce
            the same suggestion text (deterministic hashing).
        """
        key = (str(user_id), str(session_id))
        buf = self._get_or_create_buffer(key)

        # ---- Coerce embedding to a 1-D float32 tensor ----------------
        emb_safe = self._safe_embedding(embedding)

        obs = _Observation(
            embedding=emb_safe,
            composition_time_ms=float(max(0.0, composition_time_ms)),
            edit_count=int(max(0, edit_count)),
            pause_before_send_ms=float(max(0.0, pause_before_send_ms)),
            keystroke_iki_mean=float(max(0.0, keystroke_iki_mean)),
            keystroke_iki_std=float(max(0.0, keystroke_iki_std)),
        )
        buf.append(obs)
        # Increment debounce counter every turn.
        self._turns_since_suggestion[key] = self._turns_since_suggestion.get(key, 99) + 1
        self._turn_index[key] = self._turn_index.get(key, -1) + 1

        # ---- Need at least baseline + recent observations ------------
        # Warm-up: until the buffer holds at least ``recent_size + 2``
        # entries we cannot meaningfully separate baseline from
        # recent, so we report no shift.
        min_required = self.recent_size + 2
        if len(buf) < min_required:
            return AffectShift(
                detected=False,
                direction="neutral",
                magnitude=0.0,
                iki_delta_pct=0.0,
                edit_delta_pct=0.0,
                suggestion="",
            )

        # ---- Split buffer into baseline + recent windows ------------
        observations = list(buf)
        # Baseline = everything BEFORE the recent window.
        baseline_window = observations[: -self.recent_size]
        recent_window = observations[-self.recent_size :]

        # ---- Embedding shift magnitude ------------------------------
        magnitude = self._embedding_magnitude(baseline_window, recent_window)

        # ---- Keystroke metric deltas (signed %) ---------------------
        iki_delta_pct = self._signed_pct(
            recent=[o.keystroke_iki_mean for o in recent_window],
            baseline=[o.keystroke_iki_mean for o in baseline_window],
        )
        edit_delta_pct = self._signed_pct(
            recent=[float(o.edit_count) for o in recent_window],
            baseline=[float(o.edit_count) for o in baseline_window],
            # Edits are integers and the baseline often contains
            # zeros; treat a zero baseline with non-zero recent as a
            # large positive delta rather than dividing by zero.
            zero_baseline_default=200.0 if any(o.edit_count > 0 for o in recent_window) else 0.0,
        )

        # ---- Direction inference (keystroke-driven) -----------------
        direction = self._infer_direction(iki_delta_pct, edit_delta_pct)

        # ---- Detection (either signal can trigger) -------------------
        embedding_fired = magnitude >= self.magnitude_threshold
        keystroke_fired = self._keystroke_fired(direction, iki_delta_pct, edit_delta_pct)

        if not (embedding_fired or keystroke_fired):
            return AffectShift(
                detected=False,
                direction="neutral",
                magnitude=float(magnitude),
                iki_delta_pct=float(iki_delta_pct),
                edit_delta_pct=float(edit_delta_pct),
                suggestion="",
            )

        # ---- Debounce: at most one suggestion per N turns -----------
        if self._turns_since_suggestion.get(key, 99) <= self._DEBOUNCE_TURNS:
            # Detected, but suppress the suggestion text.
            return AffectShift(
                detected=True,
                direction=direction,
                magnitude=float(magnitude),
                iki_delta_pct=float(iki_delta_pct),
                edit_delta_pct=float(edit_delta_pct),
                suggestion="",
            )

        # ---- Build deterministic suggestion --------------------------
        suggestion = self._pick_suggestion(
            user_id=user_id,
            session_id=session_id,
            turn_index=self._turn_index[key],
            direction=direction,
        )

        # Reset the debounce counter so the next 4 turns suppress.
        self._turns_since_suggestion[key] = 0

        return AffectShift(
            detected=True,
            direction=direction,
            magnitude=float(magnitude),
            iki_delta_pct=float(iki_delta_pct),
            edit_delta_pct=float(edit_delta_pct),
            suggestion=suggestion,
        )

    def end_session(self, user_id: str, session_id: str) -> None:
        """Drop all rolling state for a session.

        Idempotent — calling on an unknown session is a no-op.
        """
        key = (str(user_id), str(session_id))
        self._buffers.pop(key, None)
        self._turn_index.pop(key, None)
        self._turns_since_suggestion.pop(key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_buffer(
        self, key: tuple[str, str]
    ) -> Deque[_Observation]:
        """Return the per-session ring buffer, creating + LRU-evicting as needed."""
        buf = self._buffers.get(key)
        if buf is not None:
            self._buffers.move_to_end(key)
            return buf
        # Evict oldest if at capacity.
        while len(self._buffers) >= self.max_sessions:
            evicted_key, _ = self._buffers.popitem(last=False)
            self._turn_index.pop(evicted_key, None)
            self._turns_since_suggestion.pop(evicted_key, None)
            logger.debug(
                "AffectShiftDetector evicted oldest session: user=%s session=%s",
                evicted_key[0],
                evicted_key[1],
            )
        buf = deque(maxlen=self.window_size)
        self._buffers[key] = buf
        self._turn_index[key] = -1
        # Initialise debounce counter to a sentinel that lets the
        # first detection through (i.e. > _DEBOUNCE_TURNS).
        self._turns_since_suggestion[key] = self._DEBOUNCE_TURNS + 1
        return buf

    @staticmethod
    def _safe_embedding(embedding: torch.Tensor | None) -> torch.Tensor:
        """Coerce arbitrary embedding inputs to a flat float32 1-D tensor.

        Falls back to a 64-dim zero tensor on bad input rather than
        raising — the detector is decorative; it must never break the
        pipeline.
        """
        if embedding is None:
            return torch.zeros(64, dtype=torch.float32)
        try:
            t = embedding.detach().to(dtype=torch.float32, copy=False)
        except Exception:
            return torch.zeros(64, dtype=torch.float32)
        if t.ndim > 1:
            t = t.flatten()
        if t.numel() == 0:
            return torch.zeros(64, dtype=torch.float32)
        # Replace NaN / inf with zeros so the L2 distance stays finite.
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t

    @staticmethod
    def _embedding_magnitude(
        baseline: list[_Observation],
        recent: list[_Observation],
    ) -> float:
        """L2(recent_mean - baseline_mean) / max(σ_baseline, 1e-3)."""
        if not baseline or not recent:
            return 0.0
        # Stack the embeddings, padding with zeros if dims disagree
        # (defensive; in practice every embedding is 64-d).
        try:
            b_stack = torch.stack([o.embedding for o in baseline])
            r_stack = torch.stack([o.embedding for o in recent])
        except RuntimeError:
            return 0.0
        baseline_mean = b_stack.mean(dim=0)
        recent_mean = r_stack.mean(dim=0)
        diff = recent_mean - baseline_mean
        l2 = float(torch.linalg.norm(diff).item())

        # σ_baseline = mean per-dim std-dev across the baseline
        # window.  When the window has only one entry we fall back
        # to a small floor so we don't divide by zero.
        if b_stack.shape[0] >= 2:
            sigma = float(b_stack.std(dim=0, unbiased=False).mean().item())
        else:
            sigma = 0.0
        sigma = max(sigma, 1e-3)
        magnitude = l2 / sigma
        if not math.isfinite(magnitude):
            return 0.0
        return float(magnitude)

    @staticmethod
    def _signed_pct(
        recent: list[float],
        baseline: list[float],
        *,
        zero_baseline_default: float = 0.0,
    ) -> float:
        """Return signed % change of mean(recent) vs mean(baseline)."""
        if not recent or not baseline:
            return 0.0
        r = sum(recent) / len(recent)
        b = sum(baseline) / len(baseline)
        if b <= 1e-9:
            # Avoid division by zero; caller picks a sensible default
            # (e.g. "treat zero edits as a tiny baseline so a single
            # edit registers as a large positive delta").
            if r > 1e-9:
                return float(zero_baseline_default)
            return 0.0
        pct = 100.0 * (r - b) / b
        if not math.isfinite(pct):
            return 0.0
        return float(pct)

    def _infer_direction(
        self, iki_delta_pct: float, edit_delta_pct: float
    ) -> str:
        """Map keystroke deltas to a direction label.

        * **rising_load**: IKI ≥ +20% AND/OR edits ≥ +50%.
        * **falling_load**: IKI ≤ -15% AND edits flat-or-falling
          (≤ +5%).
        * **neutral**: anything else.
        """
        if (
            iki_delta_pct >= self._RISING_IKI_PCT
            or edit_delta_pct >= self._RISING_EDIT_PCT
        ):
            return "rising_load"
        if (
            iki_delta_pct <= self._FALLING_IKI_PCT
            and edit_delta_pct <= self._FALLING_EDIT_MAX_PCT
        ):
            return "falling_load"
        return "neutral"

    def _keystroke_fired(
        self,
        direction: str,
        iki_delta_pct: float,
        edit_delta_pct: float,
    ) -> bool:
        """Pure-keystroke fallback trigger.

        Fires when both conditions hold for the implied direction:

        * rising_load: IKI ≥ +20% AND edits ≥ +50% (the conjunction
          documented in the brief: "shift = IKI > 1.2× baseline AND
          edits > 1.5× baseline").
        * falling_load: IKI ≤ -15% AND edits flat-or-falling.
        """
        if direction == "rising_load":
            return (
                iki_delta_pct >= self._KS_RISING_IKI_PCT
                and edit_delta_pct >= self._KS_RISING_EDIT_PCT
            )
        if direction == "falling_load":
            return (
                iki_delta_pct <= self._FALLING_IKI_PCT
                and edit_delta_pct <= self._FALLING_EDIT_MAX_PCT
            )
        return False

    @staticmethod
    def _pick_suggestion(
        user_id: str,
        session_id: str,
        turn_index: int,
        direction: str,
    ) -> str:
        """Deterministically pick a suggestion from the bank for *direction*.

        Same ``(user_id, session_id, turn_index)`` always produces the
        same suggestion — important for the demo so the trace doesn't
        drift between runs.
        """
        if direction == "rising_load":
            bank = RISING_LOAD_SUGGESTIONS
        elif direction == "falling_load":
            bank = FALLING_LOAD_SUGGESTIONS
        else:
            bank = NEUTRAL_SHIFT_SUGGESTIONS
        # SHA-1 over the tuple keeps the hash stable and is ample for
        # picking 1-of-3.  ``hash(tuple)`` is salted per-process, which
        # would break determinism across server restarts.
        h = hashlib.sha1(
            f"{user_id}::{session_id}::{turn_index}".encode("utf-8")
        ).digest()
        idx = int.from_bytes(h[:4], "big") % len(bank)
        return bank[idx]


__all__ = [
    "AffectShift",
    "AffectShiftDetector",
    "RISING_LOAD_SUGGESTIONS",
    "FALLING_LOAD_SUGGESTIONS",
    "NEUTRAL_SHIFT_SUGGESTIONS",
]
