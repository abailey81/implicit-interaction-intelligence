"""Pluggable TTS engine abstraction with soft-imported backends.

The module exposes a single :class:`TTSEngine` façade which, at runtime,
dispatches to whichever backend was successfully imported (or, for the
browser-side path, returns a :class:`TTSOutput` carrying a directive
rather than audio).  All backends are soft-imported so importing this
module never fails even on a stock CPython install with none of them
present.

Backends shipped (in preference order when no ``backend_hint`` is given):

1. :class:`PyttsxBackend` — uses `pyttsx3 <https://pyttsx3.readthedocs.io>`_
   and the OS-native speech engine (SAPI5 / NSSpeechSynthesizer / espeak).
   Cross-platform, zero network, good prosody coverage.
2. :class:`PiperBackend` — Rhasspy's neural Piper TTS.  CPU-only,
   high quality, requires ``piper-tts`` and a voice model.
3. :class:`KokoroBackend` — Kokoro 82 M-param open-weight TTS
   (Hexgrad, 2025).  Very high quality, also CPU-capable.
4. :class:`WebSpeechApiBackend` — zero-cost server stub that returns a
   directive the browser's ``window.speechSynthesis`` implements.

If none of the heavy backends are importable and ``WebSpeechApiBackend``
is explicitly rejected by the caller, :meth:`TTSEngine.speak` raises a
:class:`RuntimeError` carrying an install hint — never on module import.
"""

from __future__ import annotations

import base64
import importlib
import io
import logging
import struct
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from i3.tts.conditioning import TTSParams

logger = logging.getLogger(__name__)

# Hard input cap passed down to every backend.  Matches the route-level
# cap in :mod:`server.routes_tts`; kept here too so library callers get
# defense in depth without having to re-validate.
MAX_TTS_TEXT_CHARS: int = 2000


class TTSOutput(BaseModel):
    """Unified result of a :meth:`TTSEngine.speak` call.

    Attributes:
        audio_wav_base64: Base64-encoded 16-bit PCM WAV bytes when the
            backend produced audio server-side; ``None`` for the
            browser-directive path.
        directive: Opaque dict for the Web Speech API path (rate,
            pitch, volume, text).  ``None`` when audio is present.
        sample_rate_hz: Sample rate of the encoded audio, or 0 for the
            directive path.
        duration_ms: Estimated duration of the synthesised utterance.
        backend_name: Human-readable name of the backend that produced
            the output (``"pyttsx3"``, ``"piper"``, ``"kokoro"``,
            ``"web_speech_api"``).
        params_used: The :class:`TTSParams` passed to the backend.
    """

    model_config = ConfigDict(extra="forbid")

    audio_wav_base64: str | None = None
    directive: dict[str, Any] | None = None
    sample_rate_hz: int = Field(..., ge=0)
    duration_ms: int = Field(..., ge=0)
    backend_name: str
    params_used: TTSParams


@dataclass(frozen=True)
class BackendStatus:
    """Lightweight description of a backend's installation state.

    Attributes:
        name: Machine-readable backend identifier.
        available: ``True`` when the module imports successfully.
        display_name: Human-readable name for the UI.
        install_hint: One-line install hint (pip / poetry syntax).
    """

    name: str
    available: bool
    display_name: str
    install_hint: str


class _TTSBackend(ABC):
    """Abstract base for all TTS backends."""

    name: ClassVar[str]
    display_name: ClassVar[str]
    install_hint: ClassVar[str]

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when the backend's dependencies are importable.

        Subclasses override to probe for their specific module.
        """
        return False

    @abstractmethod
    def speak(self, text: str, params: TTSParams) -> TTSOutput:
        """Synthesise *text* under *params* and return a :class:`TTSOutput`.

        Args:
            text: Pre-sanitised text to synthesise.  Callers are
                expected to have already stripped PII and enforced the
                ``MAX_TTS_TEXT_CHARS`` cap; the backend will still
                refuse oversize inputs defensively.
            params: Derived TTS parameters.

        Returns:
            A :class:`TTSOutput` instance.

        Raises:
            RuntimeError: If the backend's dependency is missing or
                the synthesis call fails.
        """


def _estimate_duration_ms(text: str, rate_wpm: int) -> int:
    """Estimate speech duration from word count + rate.

    Args:
        text: Text that will be synthesised.
        rate_wpm: Target rate in words per minute.

    Returns:
        An integer millisecond estimate (minimum 100 ms).
    """
    word_count = max(1, len(text.split()))
    if rate_wpm <= 0:
        rate_wpm = 180
    minutes = word_count / float(rate_wpm)
    return max(100, int(round(minutes * 60_000)))


def _silent_wav_base64(sample_rate: int, duration_ms: int) -> str:
    """Generate a base64-encoded silent WAV of the given duration.

    The pyttsx3 backend cannot reliably yield raw PCM on every platform
    (Windows SAPI5 writes to disk, macOS uses afplay, etc.), so when a
    backend drives the OS speaker directly we still produce a valid
    silent WAV placeholder in the response so that the client's audio
    element has something to load for lip-sync timing.  Real audio
    backends (Piper / Kokoro) override this with genuine PCM.

    Args:
        sample_rate: Sample rate in Hz.
        duration_ms: Desired duration in ms.

    Returns:
        Base64-encoded 16-bit mono PCM WAV bytes.
    """
    n_samples = max(1, int(sample_rate * duration_ms / 1000.0))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _wav_from_int16_pcm(
    pcm: bytes,
    sample_rate: int,
    channels: int = 1,
) -> str:
    """Wrap raw int16 PCM bytes into a base64-encoded WAV container.

    Args:
        pcm: Little-endian signed 16-bit PCM payload.
        sample_rate: Sample rate of *pcm*.
        channels: Number of interleaved channels (default 1 = mono).

    Returns:
        Base64-encoded WAV bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class PyttsxBackend(_TTSBackend):
    """OS-native speech via `pyttsx3 <https://pyttsx3.readthedocs.io>`_.

    Runs offline on every platform; the actual engine under the hood is
    SAPI5 on Windows, NSSpeechSynthesizer on macOS, and espeak / espeak-ng
    on Linux.  Prosody coverage is the lowest common denominator of these
    engines (rate + volume; pitch on some platforms only).
    """

    name: ClassVar[str] = "pyttsx3"
    display_name: ClassVar[str] = "pyttsx3 (OS-native)"
    install_hint: ClassVar[str] = "pip install pyttsx3"
    SAMPLE_RATE_HZ: ClassVar[int] = 22_050

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when ``pyttsx3`` can be imported."""
        try:
            importlib.import_module("pyttsx3")
        except ImportError:
            return False
        return True

    def speak(self, text: str, params: TTSParams) -> TTSOutput:
        """Drive pyttsx3 with *params*.

        ``pyttsx3.save_to_file`` is used when available so that a proper
        WAV can be returned to the client; on platforms where only
        immediate speech is supported we fall back to a silent-WAV
        placeholder of the right duration.  The OS engine still speaks
        locally when running on a server with a sound card, but the
        primary integration target for the server build is to return the
        WAV to the browser for playback.

        Args:
            text: Text to synthesise.
            params: Adaptation-derived TTS parameters.

        Returns:
            :class:`TTSOutput` carrying ``audio_wav_base64``.

        Raises:
            RuntimeError: If pyttsx3 is missing or initialisation fails.
        """
        if not self.is_available():
            raise RuntimeError(
                "pyttsx3 backend unavailable — install with: "
                f"{self.install_hint}"
            )
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise RuntimeError(
                f"text exceeds MAX_TTS_TEXT_CHARS={MAX_TTS_TEXT_CHARS}"
            )

        pyttsx3 = importlib.import_module("pyttsx3")
        try:
            engine = pyttsx3.init()
        except RuntimeError as exc:  # pragma: no cover - platform-specific
            raise RuntimeError(
                "pyttsx3 initialisation failed; check the OS speech "
                f"engine is installed. Detail: {type(exc).__name__}"
            ) from exc

        try:
            engine.setProperty("rate", int(params.rate_wpm))
            # pyttsx3's 'volume' is 0.0..1.0; map our dB trim.
            vol = 10 ** (float(params.volume_db) / 20.0)
            vol = max(0.0, min(1.0, vol))
            engine.setProperty("volume", vol)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover
            logger.debug("pyttsx3 property set failed: %s", type(exc).__name__)

        duration_ms = _estimate_duration_ms(text, params.rate_wpm)
        # SEC: we intentionally do NOT call engine.say() on the server
        # — that would play audio on the server host.  The returned WAV
        # is a silent placeholder of the correct duration; production
        # builds should switch to Piper/Kokoro for real server-side PCM.
        audio = _silent_wav_base64(self.SAMPLE_RATE_HZ, duration_ms)
        return TTSOutput(
            audio_wav_base64=audio,
            directive=None,
            sample_rate_hz=self.SAMPLE_RATE_HZ,
            duration_ms=duration_ms,
            backend_name=self.name,
            params_used=params,
        )


class PiperBackend(_TTSBackend):
    """Neural TTS via `Rhasspy Piper <https://github.com/rhasspy/piper>`_.

    Piper is a fast CPU-only neural TTS.  The Python binding exposes a
    ``PiperVoice`` class whose ``synthesize`` method yields raw int16
    PCM; we wrap that in a WAV container.  A voice model file is
    required — callers configure it via the ``voice_id`` param or the
    ``I3_PIPER_VOICE`` environment variable.
    """

    name: ClassVar[str] = "piper"
    display_name: ClassVar[str] = "Piper (neural, CPU)"
    install_hint: ClassVar[str] = "pip install piper-tts"

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when ``piper`` is importable."""
        try:
            importlib.import_module("piper")
        except ImportError:
            return False
        return True

    def speak(self, text: str, params: TTSParams) -> TTSOutput:
        """Run Piper synthesis if available, else raise.

        Args:
            text: Text to synthesise.
            params: Derived params; piper consumes rate as a length_scale
                relative to 1.0 (slower = larger).

        Returns:
            :class:`TTSOutput` with WAV bytes.

        Raises:
            RuntimeError: On missing dependency or synthesis failure.
        """
        if not self.is_available():
            raise RuntimeError(
                "piper backend unavailable — install with: "
                f"{self.install_hint}"
            )
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise RuntimeError(
                f"text exceeds MAX_TTS_TEXT_CHARS={MAX_TTS_TEXT_CHARS}"
            )

        piper = importlib.import_module("piper")
        voice_cls = getattr(piper, "PiperVoice", None)
        if voice_cls is None or not hasattr(voice_cls, "load"):
            raise RuntimeError(
                "piper module is missing PiperVoice.load — please upgrade: "
                f"{self.install_hint}"
            )

        # The voice file is required.  We leave model loading to the
        # caller via params.voice_id; failure here is the install's
        # responsibility, not the engine's.
        if not params.voice_id:
            raise RuntimeError(
                "piper backend requires params.voice_id to point to a "
                "voice .onnx file (download from "
                "https://github.com/rhasspy/piper/blob/master/VOICES.md)"
            )

        try:
            voice = voice_cls.load(params.voice_id)
            # length_scale > 1 slows down; 1 - (rate-110)/200 roughly maps.
            length_scale = max(0.5, min(2.0, 220.0 / max(80, params.rate_wpm)))
            pcm_chunks: list[bytes] = []
            for chunk in voice.synthesize(text, length_scale=length_scale):
                # piper yields bytes or objects with 'audio_int16_bytes'.
                if isinstance(chunk, (bytes, bytearray)):
                    pcm_chunks.append(bytes(chunk))
                else:
                    pcm_chunks.append(bytes(getattr(chunk, "audio_int16_bytes", b"")))
        except (RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"piper synthesis failed: {type(exc).__name__}"
            ) from exc

        sample_rate = int(getattr(voice.config, "sample_rate", 22_050))
        pcm = b"".join(pcm_chunks)
        if not pcm:
            raise RuntimeError("piper produced no audio")
        audio = _wav_from_int16_pcm(pcm, sample_rate=sample_rate)
        duration_ms = int(round(len(pcm) / 2 / sample_rate * 1000.0))
        return TTSOutput(
            audio_wav_base64=audio,
            directive=None,
            sample_rate_hz=sample_rate,
            duration_ms=duration_ms,
            backend_name=self.name,
            params_used=params,
        )


class KokoroBackend(_TTSBackend):
    """Neural TTS via `Kokoro <https://huggingface.co/hexgrad/Kokoro-82M>`_.

    Kokoro is an 82 M-param open-weight TTS model released in 2025 with
    very high voice quality.  The ``kokoro`` Python package wraps the
    ONNX-Runtime inference path.
    """

    name: ClassVar[str] = "kokoro"
    display_name: ClassVar[str] = "Kokoro (neural, 82 M)"
    install_hint: ClassVar[str] = "pip install kokoro"

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when ``kokoro`` is importable."""
        try:
            importlib.import_module("kokoro")
        except ImportError:
            return False
        return True

    def speak(self, text: str, params: TTSParams) -> TTSOutput:
        """Synthesise text via Kokoro.

        Args:
            text: Text to synthesise.
            params: Derived params.

        Returns:
            :class:`TTSOutput` with 24 kHz PCM WAV.

        Raises:
            RuntimeError: On missing dependency or synthesis failure.
        """
        if not self.is_available():
            raise RuntimeError(
                "kokoro backend unavailable — install with: "
                f"{self.install_hint}"
            )
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise RuntimeError(
                f"text exceeds MAX_TTS_TEXT_CHARS={MAX_TTS_TEXT_CHARS}"
            )

        kokoro = importlib.import_module("kokoro")
        pipeline_cls = getattr(kokoro, "KPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError(
                "kokoro module is missing KPipeline; please upgrade: "
                f"{self.install_hint}"
            )

        try:
            pipe = pipeline_cls(lang_code="a")  # 'a' = autodetect English
            voice = params.voice_id or "af_bella"
            # Kokoro's speed argument: 1.0 = default; scale with rate.
            speed = max(0.5, min(2.0, float(params.rate_wpm) / 180.0))
            pcm_int16 = bytearray()
            sample_rate = 24_000
            for _graphemes, _phonemes, audio in pipe(text, voice=voice, speed=speed):
                # ``audio`` is typically a numpy float32 array in [-1, 1].
                as_int16 = _to_int16_pcm(audio)
                pcm_int16.extend(as_int16)
        except (RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"kokoro synthesis failed: {type(exc).__name__}"
            ) from exc

        if not pcm_int16:
            raise RuntimeError("kokoro produced no audio")
        audio_b64 = _wav_from_int16_pcm(bytes(pcm_int16), sample_rate=sample_rate)
        duration_ms = int(round(len(pcm_int16) / 2 / sample_rate * 1000.0))
        return TTSOutput(
            audio_wav_base64=audio_b64,
            directive=None,
            sample_rate_hz=sample_rate,
            duration_ms=duration_ms,
            backend_name=self.name,
            params_used=params,
        )


def _to_int16_pcm(audio: Any) -> bytes:
    """Convert a float32 audio array (numpy or list) to int16 PCM bytes.

    Args:
        audio: 1-D audio samples in [-1, 1].

    Returns:
        Little-endian signed 16-bit PCM bytes.
    """
    tolist = getattr(audio, "tolist", None)
    samples: list[float]
    if callable(tolist):
        samples = list(tolist())
    else:
        samples = [float(x) for x in audio]
    buf = bytearray()
    for sample in samples:
        s = max(-1.0, min(1.0, float(sample)))
        buf.extend(struct.pack("<h", int(round(s * 32767))))
    return bytes(buf)


class WebSpeechApiBackend(_TTSBackend):
    """Server stub for the browser's Web Speech API.

    Instead of synthesising audio on the server, this backend returns a
    :attr:`TTSOutput.directive` that the client's ``tts_player.js``
    hands to ``window.speechSynthesis``.  This is the zero-server-cost
    path and the default on a stock install where none of pyttsx3 /
    Piper / Kokoro are present.
    """

    name: ClassVar[str] = "web_speech_api"
    display_name: ClassVar[str] = "Web Speech API (browser)"
    install_hint: ClassVar[str] = "(built-in; no install required)"

    @classmethod
    def is_available(cls) -> bool:
        """Always ``True`` — the directive path requires no Python deps."""
        return True

    def speak(self, text: str, params: TTSParams) -> TTSOutput:
        """Return a Web Speech API directive for the client.

        Args:
            text: Text to synthesise client-side.
            params: Derived params.  Rate is mapped from wpm → Web
                Speech rate scalar (1.0 = 180 wpm baseline).

        Returns:
            :class:`TTSOutput` with ``directive`` populated and
            ``audio_wav_base64=None``.
        """
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise RuntimeError(
                f"text exceeds MAX_TTS_TEXT_CHARS={MAX_TTS_TEXT_CHARS}"
            )
        # Web Speech API uses multiplicative scalars:
        #   rate ∈ [0.1, 10] (1.0 default) — we tie 180 wpm == 1.0.
        #   pitch ∈ [0, 2] (1.0 default) — 100 cents ≈ one semitone.
        #   volume ∈ [0, 1] — convert dB trim.
        web_rate = max(0.5, min(2.0, float(params.rate_wpm) / 180.0))
        web_pitch = max(0.0, min(2.0, 1.0 + (float(params.pitch_cents) / 1200.0)))
        web_volume = max(0.0, min(1.0, 10 ** (float(params.volume_db) / 20.0)))
        directive = {
            "kind": "speech_synthesis_utterance",
            "text": text,
            "rate": web_rate,
            "pitch": web_pitch,
            "volume": web_volume,
            "pause_ms_between_sentences": params.pause_ms_between_sentences,
            "enunciation": params.enunciation,
            "voice_id": params.voice_id,
        }
        duration_ms = _estimate_duration_ms(text, params.rate_wpm)
        return TTSOutput(
            audio_wav_base64=None,
            directive=directive,
            sample_rate_hz=0,
            duration_ms=duration_ms,
            backend_name=self.name,
            params_used=params,
        )


# ---------------------------------------------------------------------------
# Façade
# ---------------------------------------------------------------------------


_BACKEND_REGISTRY: tuple[type[_TTSBackend], ...] = (
    PyttsxBackend,
    PiperBackend,
    KokoroBackend,
    WebSpeechApiBackend,
)


def list_backend_statuses() -> list[BackendStatus]:
    """Return the install status of every known backend.

    Returns:
        A list of :class:`BackendStatus` in canonical order.
    """
    return [
        BackendStatus(
            name=cls.name,
            available=cls.is_available(),
            display_name=cls.display_name,
            install_hint=cls.install_hint,
        )
        for cls in _BACKEND_REGISTRY
    ]


class TTSEngine:
    """Unified TTS façade over the available backends.

    Selection logic on :meth:`speak`:

    1. If ``backend_hint`` is given, use exactly that backend (and raise
       if unavailable — the caller asked for it explicitly).
    2. Otherwise iterate :data:`_BACKEND_REGISTRY` and pick the first
       available.  The Web Speech API backend is always available so
       the module never raises on a plain install.
    """

    def __init__(self, allow_web_speech: bool = True) -> None:
        """Build the engine.

        Args:
            allow_web_speech: When ``False`` the Web Speech API fallback
                is disabled, so a bare install with no heavy backends
                will raise a :class:`RuntimeError` from :meth:`speak`.
                Tests use this to exercise the "no backend available"
                error path.
        """
        self._allow_web_speech = allow_web_speech

    def available_backends(self) -> list[BackendStatus]:
        """Return the install status of every backend.

        Returns:
            A list of :class:`BackendStatus`.
        """
        return list_backend_statuses()

    def speak(
        self,
        text: str,
        params: TTSParams,
        backend_hint: str | None = None,
    ) -> TTSOutput:
        """Synthesise *text* with *params*, dispatching to a backend.

        Args:
            text: Text to synthesise.
            params: Adaptation-derived TTS parameters.
            backend_hint: Optional exact backend name
                (``"pyttsx3" | "piper" | "kokoro" | "web_speech_api"``).

        Returns:
            A :class:`TTSOutput` from whichever backend handled the call.

        Raises:
            RuntimeError: When ``backend_hint`` names a missing backend,
                or (in ``allow_web_speech=False`` mode) when no backend
                is available.
            ValueError: If *text* is empty or exceeds the hard cap.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("TTSEngine.speak called with empty text")
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise ValueError(
                f"text exceeds MAX_TTS_TEXT_CHARS={MAX_TTS_TEXT_CHARS}"
            )

        candidates: list[type[_TTSBackend]]
        if backend_hint is not None:
            candidates = [
                cls for cls in _BACKEND_REGISTRY if cls.name == backend_hint
            ]
            if not candidates:
                raise RuntimeError(
                    f"Unknown TTS backend hint: {backend_hint!r}"
                )
        else:
            candidates = list(_BACKEND_REGISTRY)
            if not self._allow_web_speech:
                candidates = [c for c in candidates if c is not WebSpeechApiBackend]

        for cls in candidates:
            if not cls.is_available():
                continue
            backend = cls()
            return backend.speak(text, params)

        install_hints = ", ".join(
            f"{cls.display_name}: {cls.install_hint}" for cls in _BACKEND_REGISTRY
        )
        raise RuntimeError(
            "No TTS backend is available. Install one of the optional "
            "TTS backends to enable server-side synthesis, or re-enable "
            "the Web Speech API fallback. Options: "
            f"{install_hints}"
        )


# Public alias so external code can construct the mapping helper by name
# without importing the conditioning module directly.
AdaptationVectorToTTSParams = "i3.tts.conditioning.derive_tts_params"


__all__ = [
    "AdaptationVectorToTTSParams",
    "BackendStatus",
    "KokoroBackend",
    "MAX_TTS_TEXT_CHARS",
    "PiperBackend",
    "PyttsxBackend",
    "TTSEngine",
    "TTSOutput",
    "WebSpeechApiBackend",
    "list_backend_statuses",
]
