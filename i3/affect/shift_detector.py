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
        confidence: Normalised confidence in ``[0.0, 1.0]``.  ``0.0``
            when not detected; in ``[0.5, 1.0]`` when detected, where
            0.5 means a tier just crossed and 1.0 means strong
            multi-tier corroboration.  Iter 9 — surfaces calibrated
            trust to the UI chip without requiring the consumer to
            re-derive it from raw deltas.
    """

    detected: bool
    direction: str
    magnitude: float
    iki_delta_pct: float
    edit_delta_pct: float
    suggestion: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the WebSocket layer."""
        return {
            "detected": bool(self.detected),
            "direction": str(self.direction),
            "magnitude": float(self.magnitude),
            "iki_delta_pct": float(self.iki_delta_pct),
            "edit_delta_pct": float(self.edit_delta_pct),
            "suggestion": str(self.suggestion),
            "confidence": float(self.confidence),
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
    #
    # Two tiers (iter 1 precision improvement):
    #
    # * STRONG: BOTH IKI ≥ +20% AND edits ≥ +50%.  Documented brief
    #   trigger; symmetric with `_infer_direction`'s primary rule.
    # * MODERATE: a SINGLE signal that is large on its own.  Catches
    #   the case where one channel dominates (e.g. user types markedly
    #   slower without making more edits, or pastes a chunk and then
    #   edits heavily without pausing).  Before iter 1, those cases
    #   were classified as `rising_load` by `_infer_direction` (which
    #   uses OR) but failed `_keystroke_fired` (which used AND), so
    #   the shift dropped silently unless the embedding magnitude
    #   independently crossed.
    _KS_RISING_IKI_PCT: float = 20.0           # strong: 1.2× ratio
    _KS_RISING_EDIT_PCT: float = 50.0          # strong: 1.5× ratio
    _KS_RISING_IKI_PCT_MODERATE: float = 35.0  # moderate single-signal: 1.35× ratio
    _KS_RISING_EDIT_PCT_MODERATE: float = 120.0  # moderate single-signal: 2.2× ratio

    # Debounce: don't append a suggestion more than once every
    # ``_DEBOUNCE_TURNS`` turns of the same session.
    _DEBOUNCE_TURNS: int = 4

    def __init__(
        self,
        window_size: int = 8,
        recent_size: int = 3,
        magnitude_threshold: float = 1.4,
        max_sessions: int = 1000,
        baseline_size: int | None = None,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if recent_size < 1 or recent_size >= window_size:
            raise ValueError("recent_size must satisfy 1 <= recent_size < window_size")
        self.window_size = int(window_size)
        self.recent_size = int(recent_size)
        # Iter 7: baseline_size = number of observations the *fixed*
        # baseline anchor holds.  Defaults to (window_size - recent_size)
        # to match the historical effective baseline length so the
        # change is invisible to short-session callers.
        self.baseline_size = int(
            baseline_size if baseline_size is not None
            else max(2, window_size - recent_size)
        )
        self.magnitude_threshold = float(magnitude_threshold)
        self.max_sessions = int(max_sessions)

        # Per-session ring buffers (rolling; holds the recent window).
        self._buffers: OrderedDict[tuple[str, str], Deque[_Observation]] = OrderedDict()
        # Iter 7: per-session FIXED baseline anchored to the first
        # ``baseline_size`` observations of the session.  Persists
        # across the rolling buffer's eviction so a sustained shift
        # is still measured against the user's original normal —
        # not a tail that drifts toward the new normal.
        self._baselines: dict[tuple[str, str], list[_Observation]] = {}
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
        # Iter 7: populate the fixed baseline with the FIRST
        # ``baseline_size`` observations of this session.  Once full,
        # it never changes for the lifetime of the session — sustained
        # shifts are measured against the original anchor, not a
        # rolling tail that drifts toward the new normal.
        baseline_anchor = self._baselines.setdefault(key, [])
        if len(baseline_anchor) < self.baseline_size:
            baseline_anchor.append(obs)
        # Increment debounce counter every turn.
        self._turns_since_suggestion[key] = self._turns_since_suggestion.get(key, 99) + 1
        self._turn_index[key] = self._turn_index.get(key, -1) + 1

        # ---- Need at least baseline + recent observations ------------
        # Warm-up: until BOTH the fixed baseline is populated AND the
        # rolling recent buffer holds enough entries we cannot
        # meaningfully separate baseline from recent.
        if (
            len(baseline_anchor) < self.baseline_size
            or len(buf) < self.recent_size + 2
        ):
            return AffectShift(
                detected=False,
                direction="neutral",
                magnitude=0.0,
                iki_delta_pct=0.0,
                edit_delta_pct=0.0,
                suggestion="",
            )

        # ---- Split buffer into baseline + recent windows ------------
        # Baseline = the FIXED first-N observations of the session.
        # Recent   = the last ``recent_size`` from the rolling buffer.
        baseline_window = list(baseline_anchor)
        recent_window = list(buf)[-self.recent_size :]

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
                confidence=0.0,
            )

        # ---- Iter 9: compute calibrated confidence for this shift -----
        confidence = self._compute_confidence(
            detected=True,
            magnitude=magnitude,
            iki_delta_pct=iki_delta_pct,
            edit_delta_pct=edit_delta_pct,
            direction=direction,
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
                confidence=confidence,
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
            confidence=confidence,
        )

    def end_session(self, user_id: str, session_id: str) -> None:
        """Drop all rolling state for a session.

        Idempotent — calling on an unknown session is a no-op.
        Iter 7: also wipes the fixed baseline anchor.
        """
        key = (str(user_id), str(session_id))
        self._buffers.pop(key, None)
        self._baselines.pop(key, None)
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
            self._baselines.pop(evicted_key, None)
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

    # The canonical embedding dimension produced by the TCN encoder.
    # Iter 6: every input embedding is canonicalised to this shape
    # before being stored in the ring buffer so torch.stack inside
    # _embedding_magnitude never raises on shape mismatch.
    _CANONICAL_DIM: int = 64

    @staticmethod
    def _safe_embedding(embedding: torch.Tensor | None) -> torch.Tensor:
        """Coerce arbitrary embedding inputs to a 64-dim float32 1-D tensor.

        Iter 6: every input is canonicalised to the canonical embedding
        dimension (64).  Undersized inputs zero-pad on the right;
        oversized inputs truncate on the right; multi-dim inputs flatten
        first.  This guarantees that the per-session ring buffer holds
        a stack-compatible set of tensors no matter how the embedding
        source's output shape evolves over time (e.g. an encoder that
        warms up at a partial dim, a downstream consumer that resizes
        the latent, etc.) — the detector is decorative and must never
        silently drop detection due to a shape mismatch.

        Falls back to a zero 64-dim tensor on bad input rather than
        raising — the detector must never break the pipeline.
        """
        target_dim = AffectShiftDetector._CANONICAL_DIM
        if embedding is None:
            return torch.zeros(target_dim, dtype=torch.float32)
        try:
            t = embedding.detach().to(
                device="cpu", dtype=torch.float32, copy=False
            )
        except Exception:
            return torch.zeros(target_dim, dtype=torch.float32)
        # Iter 25: treat any non-1D tensor (including scalar 0-dim
        # tensors) by flattening to 1D first.  ``torch.cat`` rejects
        # 0-dim tensors and ``ndim > 1`` missed scalars.
        if t.ndim != 1:
            t = t.flatten()
        if t.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32)
        # Replace NaN / inf with zeros so the L2 distance stays finite.
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        # Canonicalise to target_dim: zero-pad short, truncate long.
        n = t.numel()
        if n == target_dim:
            return t
        if n < target_dim:
            pad = torch.zeros(target_dim - n, dtype=torch.float32)
            return torch.cat([t, pad])
        # n > target_dim: truncate.
        return t[:target_dim].contiguous()

    # Iter 14: minimum sigma_baseline below which the embedding-
    # magnitude trigger is *not trusted*.  Below this floor the
    # baseline is too consistent to derive a meaningful sigma
    # estimate — dividing by an artificially-floored sigma blows
    # up the magnitude on any tiny perturbation, producing false
    # positives.  At sigma below this floor we fall back to the
    # keystroke channel only.
    _MIN_SIGMA_FOR_EMBEDDING_TRIGGER: float = 1e-2

    @staticmethod
    def _embedding_magnitude(
        baseline: list[_Observation],
        recent: list[_Observation],
    ) -> float:
        """``L2(recent_mean - baseline_mean) / max(σ_baseline, floor)``.

        Iter 14: returns 0.0 when ``σ_baseline`` is below the
        :attr:`_MIN_SIGMA_FOR_EMBEDDING_TRIGGER` floor.  A baseline
        with effectively-zero variance carries no information about
        what \"normal\" embedding noise looks like — dividing by an
        artificially-floored sigma produced multi-thousand-σ
        magnitudes on tiny embedding perturbations and triggered
        spurious shifts.  By returning 0.0 in that regime we let
        the keystroke channel be the sole detector, which is the
        documented fallback path anyway.
        """
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
        # window.  When the window has only one entry we have no
        # variance information at all.
        if b_stack.shape[0] >= 2:
            sigma = float(b_stack.std(dim=0, unbiased=False).mean().item())
        else:
            sigma = 0.0
        # Iter 14: don't trust the embedding-magnitude trigger below
        # the sigma floor — return 0 and let the keystroke channel
        # decide.
        if sigma < AffectShiftDetector._MIN_SIGMA_FOR_EMBEDDING_TRIGGER:
            return 0.0
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

    def _compute_confidence(
        self,
        *,
        detected: bool,
        magnitude: float,
        iki_delta_pct: float,
        edit_delta_pct: float,
        direction: str,
    ) -> float:
        """Normalised confidence score in ``[0.0, 1.0]`` for an
        :class:`AffectShift`.

        Iter 9 — gives downstream UI a calibrated number for the
        routing chip without the consumer re-deriving it from raw
        deltas.

        Convention:

        * ``0.0`` — not detected.
        * ``[0.5, 1.0]`` — detected.  ``0.5`` means a tier just
          crossed; ``1.0`` means strong multi-tier corroboration.

        Components:

        * Embedding evidence: ramp from 0 at ``magnitude_threshold``
          to 1 at ``3 × magnitude_threshold``.
        * Keystroke evidence (rising_load): max of
            * IKI ramp 20% -> 100% in [0, 1]
            * edit ramp 50% -> 300% in [0, 1]
        * Keystroke evidence (falling_load): IKI ramp -15% -> -60%.

        Confidence is the maximum of the two components, mapped from
        ``[0, 1]`` to ``[0.5, 1.0]``.
        """
        if not detected:
            return 0.0

        threshold = self.magnitude_threshold
        denom = max(1e-3, 2.0 * threshold)
        emb_evidence = max(0.0, min(1.0, (magnitude - threshold) / denom))

        if direction == "rising_load":
            iki_score = max(0.0, min(1.0, (iki_delta_pct - 20.0) / 80.0))
            edit_score = max(0.0, min(1.0, (edit_delta_pct - 50.0) / 250.0))
            ks_evidence = max(iki_score, edit_score)
        elif direction == "falling_load":
            # iki_delta_pct is negative; the more negative, the stronger.
            ks_evidence = max(0.0, min(1.0, (-iki_delta_pct - 15.0) / 45.0))
        else:
            ks_evidence = 0.0

        raw = max(emb_evidence, ks_evidence)
        # Map to [0.5, 1.0] when detected — ensures the chip never
        # shows < 0.5 on a real shift, while still distinguishing
        # weak and strong evidence.
        confidence = 0.5 + 0.5 * raw
        if not math.isfinite(confidence):
            return 0.5
        return max(0.5, min(1.0, confidence))

    def _keystroke_fired(
        self,
        direction: str,
        iki_delta_pct: float,
        edit_delta_pct: float,
    ) -> bool:
        """Pure-keystroke fallback trigger.

        Two tiers for the rising-load case (iter 1 precision improvement):

        * **Strong tier** — BOTH IKI ≥ +20% AND edits ≥ +50%.  The
          documented brief trigger; symmetric with the primary rule
          in :meth:`_infer_direction`.
        * **Moderate tier** — a SINGLE signal that is large on its
          own: IKI ≥ +35% (so a clearly slower typist fires even
          without an edit spike) OR edits ≥ +120% (so a large edit
          spike fires even without a slowdown).

        Falling-load follows the same documented AND rule (no moderate
        tier — falling_load by definition needs both signals to be
        consistent).

        Returns:
            ``True`` if a tier fires, ``False`` otherwise.
        """
        if direction == "rising_load":
            strong = (
                iki_delta_pct >= self._KS_RISING_IKI_PCT
                and edit_delta_pct >= self._KS_RISING_EDIT_PCT
            )
            moderate_iki = iki_delta_pct >= self._KS_RISING_IKI_PCT_MODERATE
            moderate_edit = edit_delta_pct >= self._KS_RISING_EDIT_PCT_MODERATE
            return strong or moderate_iki or moderate_edit
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
