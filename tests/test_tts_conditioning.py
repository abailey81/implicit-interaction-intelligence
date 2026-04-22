"""Tests for :mod:`i3.tts.conditioning` and the engine error path.

Covers monotonicity, accessibility dominance, bounds, explanation
content, input validation, and the "no backend available" error that
:class:`i3.tts.engine.TTSEngine` produces when soft-imports fail and
the Web Speech API fallback has been disabled.
"""

from __future__ import annotations

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.tts import (
    TTSEngine,
    TTSParams,
    derive_tts_params,
    explain_params,
)
from i3.tts.conditioning import (
    PAUSE_MS_MAX,
    PAUSE_MS_MIN,
    PITCH_CENTS_MAX,
    PITCH_CENTS_MIN,
    RATE_WPM_MAX,
    RATE_WPM_MIN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec(
    cognitive_load: float = 0.5,
    accessibility: float = 0.0,
    emotional_tone: float = 0.5,
    formality: float = 0.5,
) -> AdaptationVector:
    """Build a simple AdaptationVector with the named axes set."""
    return AdaptationVector(
        cognitive_load=cognitive_load,
        style_mirror=StyleVector(
            formality=formality,
            verbosity=0.5,
            emotionality=0.5,
            directness=0.5,
        ),
        emotional_tone=emotional_tone,
        accessibility=accessibility,
    )


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_rate_decreases_with_cognitive_load() -> None:
    """Rate must be a non-increasing function of cognitive_load."""
    r0 = derive_tts_params(_vec(cognitive_load=0.0)).rate_wpm
    r1 = derive_tts_params(_vec(cognitive_load=0.5)).rate_wpm
    r2 = derive_tts_params(_vec(cognitive_load=1.0)).rate_wpm
    assert r0 > r1 > r2


def test_pause_increases_with_cognitive_load() -> None:
    """Inter-sentence pause must be non-decreasing in cognitive_load."""
    p0 = derive_tts_params(_vec(cognitive_load=0.0)).pause_ms_between_sentences
    p1 = derive_tts_params(_vec(cognitive_load=0.5)).pause_ms_between_sentences
    p2 = derive_tts_params(_vec(cognitive_load=1.0)).pause_ms_between_sentences
    assert p0 < p1 < p2


# ---------------------------------------------------------------------------
# Accessibility dominance
# ---------------------------------------------------------------------------


def test_accessibility_high_caps_rate_at_120() -> None:
    """accessibility > 0.6 caps rate at 120 wpm regardless of other dims."""
    # Even with warm tone + low formality (both push rate up), the cap wins.
    vec = _vec(
        cognitive_load=0.1,
        accessibility=0.9,
        emotional_tone=0.0,
        formality=0.0,
    )
    params = derive_tts_params(vec)
    assert params.rate_wpm <= 120


def test_accessibility_high_forces_maximum_enunciation() -> None:
    """Accessibility > 0.6 forces enunciation to 'maximum'."""
    params = derive_tts_params(_vec(accessibility=0.9))
    assert params.enunciation == "maximum"


def test_accessibility_high_enforces_pause_floor() -> None:
    """Accessibility > 0.6 enforces pause >= 600 ms."""
    params = derive_tts_params(_vec(cognitive_load=0.0, accessibility=0.9))
    assert params.pause_ms_between_sentences >= 600


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("load", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("acc", [0.0, 0.3, 0.6, 0.9])
def test_rate_always_within_bounds(load: float, acc: float) -> None:
    """rate_wpm always lies in [80, 220]."""
    params = derive_tts_params(_vec(cognitive_load=load, accessibility=acc))
    assert RATE_WPM_MIN <= params.rate_wpm <= RATE_WPM_MAX


@pytest.mark.parametrize("tone", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pitch_always_within_bounds(tone: float) -> None:
    """pitch_cents always lies in [-100, 100]."""
    params = derive_tts_params(_vec(emotional_tone=tone))
    assert PITCH_CENTS_MIN <= params.pitch_cents <= PITCH_CENTS_MAX


@pytest.mark.parametrize("load", [0.0, 0.5, 1.0])
def test_pause_always_within_bounds(load: float) -> None:
    """pause_ms always lies in [100, 1000]."""
    params = derive_tts_params(_vec(cognitive_load=load))
    assert (
        PAUSE_MS_MIN <= params.pause_ms_between_sentences <= PAUSE_MS_MAX
    )


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


def test_explain_params_mentions_moved_dimension() -> None:
    """explain_params returns a non-empty string naming the dominant dim."""
    vec = _vec(cognitive_load=0.9)
    params = derive_tts_params(vec)
    text = explain_params(params, vec)
    assert text
    assert "cognitive_load" in text.lower()
    # And mentions the actual rate number.
    assert str(params.rate_wpm) in text


def test_explain_params_mentions_accessibility_when_dominant() -> None:
    """A strongly accessibility-elevated vector names accessibility."""
    vec = _vec(cognitive_load=0.2, accessibility=0.85)
    params = derive_tts_params(vec)
    text = explain_params(params, vec)
    assert "accessibility" in text.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_cognitive_load_raises() -> None:
    """Hand-built out-of-range AdaptationVectors are rejected.

    AdaptationVector's ``__post_init__`` clamps to [0,1], so a negative
    value routed through it would never actually reach derive_tts_params.
    We exercise the guard by constructing a thin shim object that
    bypasses the clamping.
    """

    class _Loose:
        cognitive_load = -0.5
        emotional_tone = 0.5
        accessibility = 0.0
        style_mirror = StyleVector.default()

    with pytest.raises(ValueError, match="outside"):
        derive_tts_params(_Loose())  # type: ignore[arg-type]


def test_out_of_range_accessibility_raises() -> None:
    """accessibility > 1 (uncoerced) is refused."""

    class _Loose:
        cognitive_load = 0.5
        emotional_tone = 0.5
        accessibility = 1.5
        style_mirror = StyleVector.default()

    with pytest.raises(ValueError, match="outside"):
        derive_tts_params(_Loose())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TTSParams shape
# ---------------------------------------------------------------------------


def test_ttsparams_is_serialisable() -> None:
    """TTSParams is a Pydantic model and round-trips through JSON."""
    params = derive_tts_params(_vec())
    dumped = params.model_dump(mode="json")
    restored = TTSParams.model_validate(dumped)
    assert restored == params


# ---------------------------------------------------------------------------
# Engine error path
# ---------------------------------------------------------------------------


def test_engine_raises_runtimeerror_when_no_backend() -> None:
    """Disabling the Web Speech fallback AND hiding every other backend
    triggers a clear RuntimeError with install hints.

    This test patches the backend registry's ``is_available`` probe so
    the heavy backends appear missing without having to actually
    uninstall them.
    """
    import i3.tts.engine as engine_mod

    originals = [
        (cls, cls.is_available) for cls in engine_mod._BACKEND_REGISTRY
    ]
    try:
        for cls, _ in originals:
            cls.is_available = classmethod(lambda _c: False)  # type: ignore[method-assign]
        eng = TTSEngine(allow_web_speech=False)
        params = derive_tts_params(_vec())
        with pytest.raises(RuntimeError, match="No TTS backend"):
            eng.speak("Hello there", params)
    finally:
        for cls, original in originals:
            cls.is_available = original  # type: ignore[method-assign]


def test_engine_rejects_unknown_backend_hint() -> None:
    """A backend_hint not in the registry raises."""
    eng = TTSEngine()
    params = derive_tts_params(_vec())
    with pytest.raises(RuntimeError, match="Unknown TTS backend"):
        eng.speak("Hi", params, backend_hint="no-such-backend")


def test_engine_rejects_empty_text() -> None:
    """Empty text raises ValueError before any backend is touched."""
    eng = TTSEngine()
    params = derive_tts_params(_vec())
    with pytest.raises(ValueError):
        eng.speak("   ", params)


def test_web_speech_backend_returns_directive() -> None:
    """The Web Speech API backend always returns a directive, no WAV."""
    eng = TTSEngine()
    params = derive_tts_params(_vec(cognitive_load=0.8))
    out = eng.speak("Hello from the demo.", params, backend_hint="web_speech_api")
    assert out.audio_wav_base64 is None
    assert out.directive is not None
    assert out.directive["text"] == "Hello from the demo."
    # A slowed rate (cognitive_load=0.8) should yield rate < 1.0.
    assert out.directive["rate"] < 1.0
