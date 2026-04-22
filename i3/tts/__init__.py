"""Adaptation-conditioned text-to-speech for Implicit Interaction Intelligence.

The :mod:`i3.tts` package extends the adaptation layer to the output
modality: the same 8-dim :class:`~i3.adaptation.types.AdaptationVector`
that shapes *what* the system says also shapes *how* it says it.

Public surface:

* :class:`TTSEngine` — pluggable façade over pyttsx3, Piper, Kokoro, and
  the Web Speech API directive path.
* :class:`TTSOutput` — unified response (WAV base64 OR browser
  directive).
* :func:`derive_tts_params` (re-exported as
  :data:`AdaptationVectorToTTSParams` for call-site clarity) — maps an
  :class:`AdaptationVector` to concrete prosody parameters.

The submodules are loaded eagerly; heavy TTS backends are soft-imported
inside :mod:`i3.tts.engine` so a stock install never fails here.

See :doc:`docs/research/adaptive_tts.md` for the design rationale.
"""

from __future__ import annotations

from i3.tts.conditioning import (
    EnunciationMode,
    TTSParams,
    derive_tts_params,
    explain_params,
)
from i3.tts.engine import (
    BackendStatus,
    KokoroBackend,
    PiperBackend,
    PyttsxBackend,
    TTSEngine,
    TTSOutput,
    WebSpeechApiBackend,
    list_backend_statuses,
)

# Public alias matching the spec: the function IS the mapping from
# AdaptationVector to TTSParams.
AdaptationVectorToTTSParams = derive_tts_params

__all__ = [
    "AdaptationVectorToTTSParams",
    "BackendStatus",
    "EnunciationMode",
    "KokoroBackend",
    "PiperBackend",
    "PyttsxBackend",
    "TTSEngine",
    "TTSOutput",
    "TTSParams",
    "WebSpeechApiBackend",
    "derive_tts_params",
    "explain_params",
    "list_backend_statuses",
]
