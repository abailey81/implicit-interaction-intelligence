"""AdaptationVector → TTS parameter mapping.

This module converts I³'s 8-dimensional :class:`AdaptationVector` into a
concrete set of text-to-speech parameters (rate, pitch, volume, inter-
sentence pause, enunciation mode) so that the system speaks *how* it has
read the user.  This is the symmetry principle of the TTS layer: if we
read a user's implicit signals (slow typing, elevated correction rate,
low engagement) we mirror them on the output side by slowing down,
lengthening pauses, and enunciating more carefully.

The mapping is intentionally transparent and tested linearly — no
black-box prosody model.  Every dimension's contribution can be read
off the source, and :func:`explain_params` produces a one-sentence
human-readable justification that the UI surfaces alongside the audio
(see :doc:`docs/research/adaptive_tts.md` for the design rationale and
the full derivation).

The module deliberately does NOT import any TTS backend — it is pure
math over the AdaptationVector.  The :class:`~i3.tts.engine.TTSEngine`
is the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector

EnunciationMode = Literal["natural", "clear", "maximum"]

# Clip bounds applied to every returned :class:`TTSParams` instance.
# These match the natural operating range of pyttsx3 / Piper / Kokoro
# and the Web Speech API.
RATE_WPM_MIN: int = 80
RATE_WPM_MAX: int = 220
PITCH_CENTS_MIN: float = -100.0
PITCH_CENTS_MAX: float = 100.0
PAUSE_MS_MIN: int = 100
PAUSE_MS_MAX: int = 1000
VOLUME_DB_MIN: float = -12.0
VOLUME_DB_MAX: float = 6.0


class TTSParams(BaseModel):
    """Concrete TTS parameters derived from an :class:`AdaptationVector`.

    Attributes:
        rate_wpm: Speech rate in words per minute.  Bounded to
            ``[RATE_WPM_MIN, RATE_WPM_MAX]`` = ``[80, 220]``.
        pitch_cents: Pitch shift in cents relative to the voice's
            neutral pitch.  Bounded to ``[-100, +100]``.
        volume_db: Volume trim in dB relative to unity.  Bounded to
            ``[-12, +6]``.
        pause_ms_between_sentences: Silence inserted between sentences
            in milliseconds.  Bounded to ``[100, 1000]``.
        enunciation: Consonant articulation profile.  ``"natural"`` is
            the default, ``"clear"`` is a mild enunciation boost, and
            ``"maximum"`` is the accessibility-dominant mode.
        voice_id: Optional backend-specific voice identifier.  When
            ``None`` the engine picks the default voice.
    """

    model_config = ConfigDict(extra="forbid")

    rate_wpm: int = Field(..., ge=RATE_WPM_MIN, le=RATE_WPM_MAX)
    pitch_cents: float = Field(..., ge=PITCH_CENTS_MIN, le=PITCH_CENTS_MAX)
    volume_db: float = Field(..., ge=VOLUME_DB_MIN, le=VOLUME_DB_MAX)
    pause_ms_between_sentences: int = Field(..., ge=PAUSE_MS_MIN, le=PAUSE_MS_MAX)
    enunciation: EnunciationMode
    voice_id: str | None = None


@dataclass(frozen=True)
class _DimContribution:
    """Tracks one dimension's absolute Δrate contribution.

    Used by :func:`explain_params` to identify the dimension that moved
    the output parameters the most.
    """

    name: str
    value: float
    delta_rate_wpm: float


def _clip_int(value: float, lo: int, hi: int) -> int:
    """Clip *value* to ``[lo, hi]`` and coerce to :class:`int`.

    Args:
        value: Input value (may be ``float``).
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.

    Returns:
        ``int(round(value))`` clipped to ``[lo, hi]``.
    """
    return int(round(max(float(lo), min(float(hi), float(value)))))


def _clip_float(value: float, lo: float, hi: float) -> float:
    """Clip *value* to ``[lo, hi]`` and coerce to :class:`float`.

    Args:
        value: Input value.
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.

    Returns:
        ``max(lo, min(hi, value))`` as :class:`float`.
    """
    return float(max(float(lo), min(float(hi), float(value))))


def _validate_adaptation(adaptation: AdaptationVector) -> None:
    """Raise :class:`ValueError` when the adaptation vector is out-of-range.

    :class:`AdaptationVector` normally clamps on construction, but this
    module is also called with user-supplied override vectors from the
    what-if panel, and a negative value slipping through a hand-built
    override would silently produce a faster-than-max rate.  We refuse
    the call early.

    Args:
        adaptation: The vector to validate.

    Raises:
        ValueError: If any scalar dimension is outside ``[0, 1]``.
    """
    checks = {
        "cognitive_load": adaptation.cognitive_load,
        "emotional_tone": adaptation.emotional_tone,
        "accessibility": adaptation.accessibility,
        "style_mirror.formality": adaptation.style_mirror.formality,
        "style_mirror.verbosity": adaptation.style_mirror.verbosity,
        "style_mirror.emotionality": adaptation.style_mirror.emotionality,
        "style_mirror.directness": adaptation.style_mirror.directness,
    }
    for name, val in checks.items():
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(
                f"AdaptationVector.{name}={val!r} is outside [0, 1] — "
                "refusing to derive TTS params."
            )


def derive_tts_params(
    adaptation: AdaptationVector,
    base_rate: int = 180,
) -> TTSParams:
    """Map an :class:`AdaptationVector` to a concrete :class:`TTSParams`.

    The conditioning equations (all inputs ∈ [0, 1]):

    * ``rate_wpm = base_rate - 70 * cognitive_load``
      (linear; reaches 110 wpm at load 1.0).
    * A formality tilt of ±5 % is applied as
      ``rate *= 1 - 0.10 * (formality - 0.5)``
      (high formality slows, low formality speeds up).
    * A warm-tone nudge of +5 % is added on the warm end:
      ``rate *= 1 + 0.05 * (1 - emotional_tone)``
      (``emotional_tone=0`` is the warmest).
    * Accessibility dominates: if ``accessibility > 0.6`` the rate is
      hard-capped at 120 wpm, enunciation is forced to ``"maximum"``,
      and ``pause_ms >= 600`` is enforced.
    * ``pause_ms = 120 + 380 * cognitive_load`` (linear; 120–500 ms).
    * ``pitch_cents = 40 * (emotional_tone * 2 - 1)`` (linear; -40..+40).
    * ``volume_db = 0 + 3 * accessibility`` (accessibility boosts volume).

    The ``style_mirror.verbosity`` dimension is **intentionally not**
    consumed here: verbosity is a content-length signal that the
    :class:`~i3.cloud.postprocess.ResponsePostProcessor` handles
    upstream of TTS.

    Args:
        adaptation: Current user-adapted state vector.
        base_rate: Baseline speech rate in wpm at cognitive_load=0.
            Defaults to 180 wpm (conversational English).

    Returns:
        A fully-populated :class:`TTSParams` instance, with every field
        clipped into its valid range.

    Raises:
        ValueError: If *adaptation* carries an out-of-range scalar.
    """
    _validate_adaptation(adaptation)

    load = float(adaptation.cognitive_load)
    access = float(adaptation.accessibility)
    tone = float(adaptation.emotional_tone)
    formality = float(adaptation.style_mirror.formality)

    # --- Rate -----------------------------------------------------------
    # Linear rate drop with cognitive load: 0.0 -> base_rate, 1.0 -> base_rate - 70.
    rate = float(base_rate) - 70.0 * load
    # Formality tilt: centered at 0.5, so 1.0 formality -> -5%, 0.0 -> +5%.
    rate *= 1.0 - 0.10 * (formality - 0.5)
    # Warm-tone nudge: up to +5% when emotional_tone is fully warm (0).
    rate *= 1.0 + 0.05 * (1.0 - tone)

    # --- Pauses ---------------------------------------------------------
    pause = 120.0 + 380.0 * load  # linear 120-500 ms

    # --- Pitch / volume -------------------------------------------------
    # Centered on neutral pitch; tone=0 (warm) => -40, tone=1 (neutral) => +40.
    # Emotionality amplifies the swing.
    emotional = float(adaptation.style_mirror.emotionality)
    pitch = 40.0 * (tone * 2.0 - 1.0) * (0.5 + 0.5 * emotional)
    volume = 3.0 * access  # 0 dB default; up to +3 dB when accessibility is on.

    # --- Enunciation ----------------------------------------------------
    if access > 0.6:
        enunciation: EnunciationMode = "maximum"
    elif access > 0.3 or load > 0.6:
        enunciation = "clear"
    else:
        enunciation = "natural"

    # --- Accessibility dominance (hard caps) ----------------------------
    # When accessibility > 0.6 the user has been observed struggling; we
    # enforce a ceiling rate and a floor on inter-sentence pause so the
    # derivation is robust against a mild warmth / formality signal that
    # would otherwise pull rate back up.
    if access > 0.6:
        rate = min(rate, 120.0)
        pause = max(pause, 600.0)

    return TTSParams(
        rate_wpm=_clip_int(rate, RATE_WPM_MIN, RATE_WPM_MAX),
        pitch_cents=_clip_float(pitch, PITCH_CENTS_MIN, PITCH_CENTS_MAX),
        volume_db=_clip_float(volume, VOLUME_DB_MIN, VOLUME_DB_MAX),
        pause_ms_between_sentences=_clip_int(pause, PAUSE_MS_MIN, PAUSE_MS_MAX),
        enunciation=enunciation,
        voice_id=None,
    )


def _dimension_contributions(
    adaptation: AdaptationVector,
    base_rate: int,
) -> list[_DimContribution]:
    """Estimate each dimension's absolute effect on the rate.

    Args:
        adaptation: The vector whose params were derived.
        base_rate: Baseline rate used by :func:`derive_tts_params`.

    Returns:
        A list of :class:`_DimContribution` sorted by absolute magnitude.
    """
    load = float(adaptation.cognitive_load)
    access = float(adaptation.accessibility)
    tone = float(adaptation.emotional_tone)
    formality = float(adaptation.style_mirror.formality)

    # Approximate per-dimension delta in wpm at the base rate.
    # Accessibility dominance is a cap, not a delta, so express it as
    # the wpm reduction caused by clamping to 120.
    load_delta = -70.0 * load
    tone_delta = base_rate * 0.05 * (1.0 - tone)
    formality_delta = -base_rate * 0.10 * (formality - 0.5)
    access_delta = 0.0
    if access > 0.6 and base_rate + load_delta > 120:
        access_delta = 120.0 - (base_rate + load_delta)

    contribs = [
        _DimContribution("cognitive_load", load, load_delta),
        _DimContribution("accessibility", access, access_delta),
        _DimContribution("emotional_tone", tone, tone_delta),
        _DimContribution("formality", formality, formality_delta),
    ]
    contribs.sort(key=lambda c: abs(c.delta_rate_wpm), reverse=True)
    return contribs


def explain_params(
    params: TTSParams,
    adaptation: AdaptationVector,
    base_rate: int = 180,
) -> str:
    """Return a one-sentence natural-language explanation of the mapping.

    The explanation names the dimension that moved the rate the most and
    echoes the numeric shift — this string is rendered under the "Speak
    response" button so users can see *why* the voice sounds different.

    Args:
        params: The :class:`TTSParams` instance to explain.
        adaptation: The vector those params were derived from.
        base_rate: Baseline rate used when the params were derived.

    Returns:
        A single-sentence human-readable string, always non-empty.
    """
    contribs = _dimension_contributions(adaptation, base_rate)
    # At least one contribution always exists.
    primary = contribs[0]

    if primary.name == "cognitive_load" and adaptation.cognitive_load > 0.05:
        reason = (
            f"cognitive_load is {adaptation.cognitive_load:.2f}, "
            "so the voice slows down and pauses are lengthened"
        )
    elif primary.name == "accessibility" and adaptation.accessibility > 0.6:
        reason = (
            f"accessibility is elevated ({adaptation.accessibility:.2f}), "
            f"so enunciation is maximum and the rate is capped at {params.rate_wpm} wpm"
        )
    elif primary.name == "emotional_tone":
        warm = 1.0 - float(adaptation.emotional_tone)
        reason = (
            f"emotional_tone warmth is {warm:.2f}, "
            f"so pitch is {params.pitch_cents:+.0f} cents"
        )
    elif primary.name == "formality":
        reason = (
            f"style formality is {adaptation.style_mirror.formality:.2f}, "
            f"which nudges rate toward {params.rate_wpm} wpm"
        )
    else:
        reason = (
            f"the adaptation vector is near neutral, "
            f"so rate stays near the baseline of {base_rate} wpm"
        )

    return (
        f"Speech rate set to {params.rate_wpm} wpm with "
        f"{params.pause_ms_between_sentences} ms inter-sentence pause and "
        f"{params.enunciation} enunciation — {reason}."
    )


__all__ = [
    "EnunciationMode",
    "PAUSE_MS_MAX",
    "PAUSE_MS_MIN",
    "PITCH_CENTS_MAX",
    "PITCH_CENTS_MIN",
    "RATE_WPM_MAX",
    "RATE_WPM_MIN",
    "TTSParams",
    "VOLUME_DB_MAX",
    "VOLUME_DB_MIN",
    "derive_tts_params",
    "explain_params",
]
