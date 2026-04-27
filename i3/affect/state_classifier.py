"""Discrete user-state classifier (Live State Badge head).

This module is the second HMI showpiece for the Huawei R&D UK / HMI Lab
brief.  Where :mod:`i3.affect.shift_detector` watches the *change* in
typing dynamics across a window, this classifier produces a snap label
of the *current* state on every turn (and every state-update mid-turn)
so the UI can paint a live ``state-badge`` chip in the nav.

Affective-computing background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The discrete states ``calm``, ``focused``, ``stressed``, ``tired``,
``distracted``, ``warming up`` come straight from the affective-
computing canon:

* Picard's *Affective Computing* (1997) gives the appraisal-theory
  basis for treating arousal × valence as a 2-D state space and then
  binning that space into a small set of discrete labels for HCI use.
* Sweller's *Cognitive Load Theory* (1988, 1994) connects observable
  task-time / edit-rate / pause statistics back to intrinsic, extraneous
  and germane load — i.e. why a high cognitive_load + edit count + IKI
  variance combination is read as "stressed", and why a high
  composition_time + low engagement is read as "tired".

The classifier itself is intentionally *not* a neural net.  A small
rule-based fuzzy classifier is

    1. **deterministic** — same inputs always produce the same label,
       which matters for the demo trace not drifting between runs;
    2. **explainable** — we can hand back the top 1-3 contributing
       signals for the badge tooltip;
    3. **robust to encoder warm-up** — the rules act directly on raw
       keystroke metrics + the 8-d adaptation, never on the noisy
       64-d TCN embedding.

Design discipline
~~~~~~~~~~~~~~~~~
* Pure Python.  No torch.  CPU-only.
* No fabrication: when a signal cannot be inferred (warm-up, missing
  keystroke baseline) we emit ``warming up`` rather than guess.
* Confidence is the softmax probability over the candidate scores; if
  the gap between top and second-best is < 0.15 we surface the
  runner-up as ``secondary_state`` so the UI can show "calm/focused"
  rather than feign certainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class UserStateLabel:
    """The result of one :func:`classify_user_state` call.

    Attributes:
        state: One of ``"calm"``, ``"focused"``, ``"stressed"``,
            ``"tired"``, ``"distracted"``, ``"warming up"``.
        confidence: Softmax probability of the chosen state in
            ``[0, 1]``.
        secondary_state: Second-best label when the top - second gap
            is < 0.15, otherwise ``None``.
        contributing_signals: 1-3 short human-readable labels (e.g.
            ``"high cognitive load"``, ``"normal IKI"``,
            ``"rare edits"``) that most influenced the choice.
    """

    state: str
    confidence: float
    secondary_state: str | None = None
    contributing_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the WebSocket layer."""
        return {
            "state": str(self.state),
            "confidence": float(self.confidence),
            "secondary_state": (
                None
                if self.secondary_state is None
                else str(self.secondary_state)
            ),
            "contributing_signals": [str(s) for s in self.contributing_signals],
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


# The set of candidate states the classifier ranks each turn.  Order
# matters only for tie-break stability (Python's ``max`` keeps the
# first-encountered argmax on ties).
_CANDIDATES: tuple[str, ...] = (
    "calm",
    "focused",
    "stressed",
    "tired",
    "distracted",
    "warming up",
)


def _step(value: float, low: float, high: float) -> float:
    """Smooth ramp: 0 below ``low``, 1 above ``high``, linear in between.

    Used as the building block for every signal-to-score mapping.  The
    smooth ramp (rather than a hard threshold) means small fluctuations
    in a metric don't cause the badge to jitter between adjacent
    states — confidence drifts gradually instead.
    """
    if not math.isfinite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= low else 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low))


def _inv_step(value: float, low: float, high: float) -> float:
    """Inverted ramp: 1 below ``low``, 0 above ``high``."""
    return 1.0 - _step(value, low, high)


def _band(value: float, lo: float, hi: float, soft: float = 0.05) -> float:
    """Score peaks (=1) when ``value`` is inside ``[lo, hi]`` and ramps off.

    Slack of ``soft`` either side gives a smooth on-ramp so the
    transition between adjacent bands isn't a cliff.
    """
    if not math.isfinite(value):
        return 0.0
    if value < lo:
        return _step(value, lo - soft, lo)
    if value > hi:
        return _inv_step(value, hi, hi + soft)
    return 1.0


def _softmax(values: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Standard softmax over a name → score mapping.

    Temperature defaults to 1.0; lowering it sharpens (more confident
    argmax), raising it flattens.  We use 1.0 for the natural reading.
    """
    if not values:
        return {}
    # Numerical stability: subtract max before exponentiating.
    items = list(values.items())
    max_v = max(v for _, v in items)
    exp_vals = [(k, math.exp((v - max_v) / max(temperature, 1e-3))) for k, v in items]
    total = sum(v for _, v in exp_vals) or 1.0
    return {k: v / total for k, v in exp_vals}


def _adapt(adaptation: dict | None, key: str, default: float = 0.5) -> float:
    """Pull a scalar from the (possibly nested) adaptation dict.

    Handles the post-promotion *flat* shape used by the WS layer
    (``cognitive_load`` / ``formality`` / ``verbosity`` / ... at the
    top level) as well as the engine's nested
    ``style_mirror.formality`` form.  Returns ``default`` when missing
    or non-finite.
    """
    if not isinstance(adaptation, dict):
        return float(default)
    if key in adaptation:
        try:
            v = float(adaptation[key])
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return v
    style = adaptation.get("style_mirror")
    if isinstance(style, dict) and key in style:
        try:
            v = float(style[key])
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return v
    return float(default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_user_state(
    *,
    adaptation: dict,
    composition_time_ms: float,
    edit_count: int,
    iki_mean: float,
    iki_std: float,
    engagement_score: float,
    deviation_from_baseline: float,
    baseline_established: bool,
    messages_in_session: int,
) -> UserStateLabel:
    """Return a discrete user-state label from raw signals.

    The implementation is a fuzzy rule-based scorer.  Each candidate
    state receives a score in ``[0, 1]`` formed by averaging the
    signal-conformance scores listed below, then we softmax over the
    candidate scores to produce a probability distribution.  The
    argmax is the returned ``state``; its softmax probability is the
    ``confidence``; the runner-up (when the gap is < 0.15) is the
    ``secondary_state``.

    Signals consumed
    ~~~~~~~~~~~~~~~~
    * **cognitive_load** (from the 8-d adaptation): primary axis.
      <0.4 favours calm; 0.4-0.65 favours focused; >0.7 favours
      stressed.
    * **edit_count**: small (≤1) favours calm/focused; ≥2 favours
      stressed; intermittent (1-2 with high IKI σ) favours distracted.
    * **iki_mean**: elevated (>180 ms) favours tired; normal (~100 ms)
      with low σ favours focused.
    * **iki_std**: low favours focused; high (>40 ms) favours
      distracted or stressed.
    * **composition_time_ms**: long (>4 s) with low engagement
      favours tired.
    * **engagement_score**: low (<0.4) penalises calm/focused, raises
      tired.
    * **formality** (from adaptation.style_mirror): high favours
      focused as a tie-breaker.
    * **baseline_established / messages_in_session**: drives the
      dedicated ``warming up`` candidate so the first 1-2 turns of a
      fresh session always land there.

    The same numeric inputs always produce the same label.

    Args:
        adaptation: 8-d adaptation dict (flat or nested).
        composition_time_ms: Total composition window for the turn.
        edit_count: Backspaces + deletes during composition.
        iki_mean: Mean inter-keystroke interval in ms.
        iki_std: Std-dev of inter-keystroke intervals in ms.
        engagement_score: Composite engagement in ``[0, 1]``.
        deviation_from_baseline: Cosine deviation of the user-state
            embedding from the baseline; only used as a tie-break.
        baseline_established: Whether enough turns have elapsed for
            the encoder to have a stable per-user baseline.
        messages_in_session: Running message count for the session.

    Returns:
        A :class:`UserStateLabel`.
    """
    # ------------------------------------------------------------------
    # Defensive coercion of every signal to a finite float.  Nothing
    # downstream tolerates NaN / inf and the inputs come from a noisy
    # mix of WS frames, the user model, and the adaptation layer.
    # ------------------------------------------------------------------
    cl = _adapt(adaptation, "cognitive_load", 0.5)
    formality = _adapt(adaptation, "formality", 0.5)
    accessibility = _adapt(adaptation, "accessibility", 0.0)
    verbosity = _adapt(adaptation, "verbosity", 0.5)

    def _f(v: float | int) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(x):
            return 0.0
        return x

    composition_ms = max(0.0, _f(composition_time_ms))
    edits = max(0, int(_f(edit_count)))
    iki_m = max(0.0, _f(iki_mean))
    iki_s = max(0.0, _f(iki_std))
    eng = max(0.0, min(1.0, _f(engagement_score)))
    deviation = abs(_f(deviation_from_baseline))
    msg_count = max(0, int(_f(messages_in_session)))

    # ------------------------------------------------------------------
    # Warm-up short-circuit.  The first turn of a session is
    # definitionally `warming up` — there is no baseline to compare
    # against.  Two-message sessions that have not yet established a
    # baseline still default to warming up unless a strong contrary
    # signal appears (we still run the full classifier and let the
    # warm-up candidate score handle the tie-break).
    # ------------------------------------------------------------------
    warmup_strong = (not baseline_established) and msg_count <= 1

    # ------------------------------------------------------------------
    # Per-state scoring.  Each line is a signal → score-in-[0,1]
    # contribution; per-state score is the mean.  We label each
    # contributor so the contributing-signals list can be assembled
    # from the most-influential ones below.
    # ------------------------------------------------------------------
    # Calm: low cognitive load, normal IKI, few edits, mid engagement.
    calm_signals = {
        "low cognitive load": _inv_step(cl, 0.4, 0.6),
        "normal IKI": _band(iki_m, 70.0, 160.0, soft=20.0),
        "low IKI variance": _inv_step(iki_s, 25.0, 50.0),
        "rare edits": _inv_step(float(edits), 0.5, 2.5),
        "mid engagement": _band(eng, 0.4, 0.85, soft=0.1),
    }

    # Focused: cognitive load in the 0.4-0.65 sweet spot, formality
    # raised, low IKI variance, few edits.
    focused_signals = {
        "cognitive load in working band": _band(cl, 0.4, 0.65, soft=0.1),
        "raised formality": _step(formality, 0.55, 0.8),
        "low IKI variance": _inv_step(iki_s, 18.0, 40.0),
        "few edits": _inv_step(float(edits), 0.5, 2.5),
        "engaged": _step(eng, 0.4, 0.7),
    }

    # Stressed: high cognitive load, multiple edits, elevated IKI σ.
    stressed_signals = {
        "high cognitive load": _step(cl, 0.55, 0.85),
        "elevated edit count": _step(float(edits), 1.0, 4.0),
        "elevated IKI variance": _step(iki_s, 25.0, 60.0),
        "long composition": _step(composition_ms, 2500.0, 6000.0),
    }

    # Tired: long composition, slow IKI, low engagement.
    tired_signals = {
        "long composition": _step(composition_ms, 3500.0, 8000.0),
        "slow IKI": _step(iki_m, 160.0, 260.0),
        "low engagement": _inv_step(eng, 0.2, 0.5),
        "raised cognitive load": _step(cl, 0.4, 0.7),
    }

    # Distracted: high IKI variance with normal mean, intermittent
    # edits, longer pauses.  Distinguished from "stressed" by
    # specifically penalising sustained high cognitive_load.
    distracted_signals = {
        "high IKI variance": _step(iki_s, 30.0, 80.0),
        "normal IKI mean": _band(iki_m, 80.0, 200.0, soft=20.0),
        "intermittent edits": _band(float(edits), 1.0, 3.0, soft=0.5),
        "moderate cognitive load": _band(cl, 0.35, 0.7, soft=0.1),
    }

    # Warming up: messages_in_session ≤ 1 OR baseline not yet
    # established.  Score is dominant on the first 1-2 turns, then
    # decays so the other states can win.
    warmup_score = 0.0
    if not baseline_established:
        if msg_count <= 1:
            warmup_score = 0.95
        elif msg_count == 2:
            warmup_score = 0.65
        elif msg_count == 3:
            warmup_score = 0.35
        else:
            warmup_score = 0.15

    def _mean(d: dict[str, float]) -> float:
        if not d:
            return 0.0
        return sum(d.values()) / len(d)

    raw_scores: dict[str, float] = {
        "calm": _mean(calm_signals),
        "focused": _mean(focused_signals),
        "stressed": _mean(stressed_signals),
        "tired": _mean(tired_signals),
        "distracted": _mean(distracted_signals),
        "warming up": warmup_score,
    }

    # ------------------------------------------------------------------
    # If the warm-up flag is strongly set we make sure no other state
    # can beat it on the first turn.  This guarantees the badge says
    # "warming up" the very first time the user sends a message,
    # matching the brief.
    # ------------------------------------------------------------------
    if warmup_strong:
        # Bump well above the per-state ceiling (1.0) so the softmax
        # cleanly resolves to "warming up" even when the keystroke
        # metrics happen to look calm on turn 1.
        raw_scores["warming up"] = max(raw_scores["warming up"], 1.4)

    # Softmax over the candidate scores.  Lower temperature makes the
    # winner cleaner; we use 0.2 to keep the badge readable (e.g.
    # 0.7 - 0.9 confidence on a clean signal rather than 0.18 - 0.21
    # ties from a temperature-1 softmax over near-equal scores).
    probs = _softmax(raw_scores, temperature=0.2)

    # Argmax.  Stable across ties via insertion order in _CANDIDATES.
    state = max(_CANDIDATES, key=lambda s: probs.get(s, 0.0))
    top_p = float(probs.get(state, 0.0))

    # Runner-up if the gap is small.
    sorted_pairs = sorted(probs.items(), key=lambda kv: -kv[1])
    secondary: str | None = None
    if len(sorted_pairs) >= 2:
        second_state, second_p = sorted_pairs[1]
        if top_p - second_p < 0.15:
            secondary = second_state

    # Pick the most-influential 1-3 signals from the chosen state's
    # contribution map.  Filtered to score >= 0.4 so we never list a
    # "contributing" signal that didn't actually fire.
    state_to_signals = {
        "calm": calm_signals,
        "focused": focused_signals,
        "stressed": stressed_signals,
        "tired": tired_signals,
        "distracted": distracted_signals,
    }
    contributing: list[str] = []
    if state in state_to_signals:
        ranked = sorted(
            state_to_signals[state].items(), key=lambda kv: -kv[1]
        )
        for label, score in ranked:
            if score >= 0.4 and len(contributing) < 3:
                contributing.append(label)
    elif state == "warming up":
        if msg_count <= 1:
            contributing.append("first message of session")
        elif not baseline_established:
            contributing.append("baseline still warming up")
        else:
            contributing.append("session warm-up")

    # Defensive: if the chosen state somehow has no >= 0.4 signal
    # (very flat distributions), at least surface the strongest one.
    if state in state_to_signals and not contributing:
        ranked = sorted(
            state_to_signals[state].items(), key=lambda kv: -kv[1]
        )
        if ranked:
            contributing = [ranked[0][0]]

    return UserStateLabel(
        state=state,
        confidence=top_p,
        secondary_state=secondary,
        contributing_signals=contributing,
    )


__all__ = [
    "UserStateLabel",
    "classify_user_state",
]
