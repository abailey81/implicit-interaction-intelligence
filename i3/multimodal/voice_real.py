"""Runnable voice-prosody feature extractor (Batch F-1).

This module is the audio analogue of the keystroke-dynamics group used by the
I3 TCN encoder.  It extracts eight classical prosodic descriptors — pitch
mean/std, speech rate, voiced-frame ratio, pause rate, cycle-to-cycle jitter
and shimmer, and a harmonics-to-noise approximation — from a raw mono
waveform.  All heavy signal-processing work is delegated to ``librosa``, which
is **soft-imported**: importing this module never fails, but calling
:meth:`VoiceProsodyExtractor.extract` without librosa installed raises a clear
:class:`RuntimeError` and the module-level helpers return zeros so the
pipeline degrades gracefully.

The eight-feature contract matches :class:`VoiceFeatureVector` in this module
(Pydantic v2) and the 8-dim shape of the keystroke-dynamics group in
:class:`i3.interaction.types.InteractionFeatureVector`, so the same
TemporalConvNet encoder can consume a voice-driven stream unchanged.

References
----------
* Boersma, P. (2001). *Praat, a system for doing phonetics by computer.*
  Glot International 5(9/10), 341-345.  (Classical jitter/shimmer definitions.)
* McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg,
  E., Nieto, O. (2015). *librosa: Audio and Music Signal Analysis in Python.*
  Proc. 14th Python in Science Conference.
* Sagisaka, Y., Campbell, N. (2004). *Prosody in Speech Synthesis.*  Springer.
* Ververidis, D., Kotropoulos, C. (2006). *Emotional speech recognition:
  Resources, features, and methods.*  Speech Communication 48(9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import librosa.  We also soft-import soundfile for ``from_wav_file``.
# ---------------------------------------------------------------------------

try:
    import librosa  # type: ignore[import-not-found]

    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    librosa = None  # type: ignore[assignment]
    _LIBROSA_AVAILABLE = False

try:
    import soundfile as _soundfile  # type: ignore[import-not-found]

    _SOUNDFILE_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    _soundfile = None  # type: ignore[assignment]
    _SOUNDFILE_AVAILABLE = False


if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import numpy.typing as npt

    NDArrayF32 = npt.NDArray[np.float32]
else:
    NDArrayF32 = np.ndarray


_INSTALL_HINT = (
    "librosa is not installed. Install the optional multimodal group with "
    "`poetry install --with multimodal` to enable voice feature extraction."
)


# ---------------------------------------------------------------------------
# Pydantic feature vector
# ---------------------------------------------------------------------------

class VoiceFeatureVector(BaseModel):
    """Eight-dimensional prosody feature group.

    The attribute ordering is significant — :meth:`to_array` relies on it —
    and it mirrors the shape of the keystroke-dynamics group so the same
    encoder can ingest voice.  All fields carry physical units: they are
    **not** pre-normalised here; downstream fusion layers handle scaling.

    Attributes:
        pitch_mean_hz: Mean fundamental frequency F0 (Hz) computed via
            ``librosa.yin``.
        pitch_std_hz: Standard deviation of F0 across voiced frames (Hz).
        speech_rate_syllables_per_s: Estimated syllable rate derived from
            onset-envelope peaks per second.
        voiced_ratio: Fraction of analysis frames classified as voiced
            (non-NaN F0 from ``pyin``).
        pause_rate_per_s: Number of silence regions per second of audio.
        jitter_percent: Mean absolute relative frame-to-frame F0 variation,
            expressed as a percentage (Boersma 2001).
        shimmer_percent: Mean absolute relative frame-to-frame RMS
            amplitude variation, expressed as a percentage.
        harmonics_to_noise_ratio_db: Harmonic-to-noise ratio approximated via
            spectral flatness (dB).  Higher values indicate more tonal /
            harmonic signals.
    """

    pitch_mean_hz: float = Field(default=0.0)
    pitch_std_hz: float = Field(default=0.0)
    speech_rate_syllables_per_s: float = Field(default=0.0)
    voiced_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    pause_rate_per_s: float = Field(default=0.0, ge=0.0)
    jitter_percent: float = Field(default=0.0, ge=0.0)
    shimmer_percent: float = Field(default=0.0, ge=0.0)
    harmonics_to_noise_ratio_db: float = Field(default=0.0)

    def to_array(self) -> NDArrayF32:
        """Return the feature vector as an 8-element ``float32`` numpy array.

        Returns:
            1-D ``np.ndarray`` of shape ``(8,)`` in declaration order.
        """
        return np.asarray(
            [
                self.pitch_mean_hz,
                self.pitch_std_hz,
                self.speech_rate_syllables_per_s,
                self.voiced_ratio,
                self.pause_rate_per_s,
                self.jitter_percent,
                self.shimmer_percent,
                self.harmonics_to_noise_ratio_db,
            ],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> VoiceFeatureVector:
        """Return a zero-valued vector (used on missing input or fallback)."""
        return cls()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class VoiceProsodyExtractor:
    """Compute :class:`VoiceFeatureVector` from a mono waveform.

    Unlike the stub :class:`i3.multimodal.voice.VoiceFeatureExtractor`, this
    extractor performs real signal processing via ``librosa``.  It is still a
    pure-Python class with no learned parameters — the same instance can be
    reused across audio clips.

    Args:
        frame_length: Number of samples per analysis frame (default 2048).
        hop_length: Hop between frames, in samples (default 512).
        fmin_hz: Lower bound for the F0 search (default 65 Hz).
        fmax_hz: Upper bound for the F0 search (default 500 Hz).
        silence_top_db: Threshold (dB below peak) under which a region is
            classified as silence for pause detection.
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        fmin_hz: float = 65.0,
        fmax_hz: float = 500.0,
        silence_top_db: float = 30.0,
    ) -> None:
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.silence_top_db = float(silence_top_db)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def extract(self, waveform: np.ndarray, sample_rate: int) -> VoiceFeatureVector:
        """Extract the eight-dim prosody vector.

        Args:
            waveform: 1-D numpy array of PCM samples.  Multi-channel inputs
                are averaged to mono.
            sample_rate: Sampling rate in Hz.  Must be positive.

        Returns:
            A :class:`VoiceFeatureVector`.  Returns zeros (with a log warning)
            when the waveform is empty.

        Raises:
            RuntimeError: If ``librosa`` is not installed.
            ValueError: If ``sample_rate`` is non-positive or ``waveform`` is
                not a numpy array.
        """
        if not _LIBROSA_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)

        if not isinstance(waveform, np.ndarray):
            raise ValueError(
                f"waveform must be numpy.ndarray, got {type(waveform).__name__}"
            )
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        if waveform.size == 0:
            logger.warning("VoiceProsodyExtractor: empty waveform")
            return VoiceFeatureVector.zeros()

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1).astype(np.float32)

        duration_s = max(float(wav.size) / float(sample_rate), 1e-6)

        # -- F0 via YIN (librosa.yin is deterministic and dependency-light) --
        f0 = librosa.yin(  # type: ignore[union-attr]
            wav,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            sr=sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        # yin returns positive Hz for every frame; we derive a voicing mask by
        # thresholding the spectral flatness (a proxy for tonality).
        flatness = librosa.feature.spectral_flatness(  # type: ignore[union-attr]
            y=wav, n_fft=self.frame_length, hop_length=self.hop_length
        ).flatten()
        # Align lengths (flatness may be ±1 frame off depending on librosa version).
        n = int(min(len(f0), len(flatness)))
        f0 = np.asarray(f0[:n], dtype=np.float32)
        flatness = np.asarray(flatness[:n], dtype=np.float32)

        voiced_mask = (
            np.isfinite(f0) & (f0 > self.fmin_hz) & (flatness < 0.5)
        )
        voiced_ratio = float(voiced_mask.mean()) if n > 0 else 0.0
        f0_voiced = f0[voiced_mask] if voiced_mask.any() else np.asarray([], dtype=np.float32)

        pitch_mean = float(np.mean(f0_voiced)) if f0_voiced.size > 0 else 0.0
        pitch_std = float(np.std(f0_voiced)) if f0_voiced.size > 0 else 0.0

        # -- Speech rate via onset detection ---------------------------------
        onset_env = librosa.onset.onset_strength(  # type: ignore[union-attr]
            y=wav, sr=sample_rate, hop_length=self.hop_length
        )
        onsets = librosa.onset.onset_detect(  # type: ignore[union-attr]
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length,
            units="frames",
        )
        speech_rate = float(len(onsets)) / duration_s

        # -- Pause detection via non-silent interval complement --------------
        non_silent = librosa.effects.split(  # type: ignore[union-attr]
            wav, top_db=self.silence_top_db,
            frame_length=self.frame_length, hop_length=self.hop_length,
        )
        # Pauses = gaps between non-silent segments, plus leading/trailing silence.
        if non_silent.size == 0:
            # Whole clip is silence.
            n_pauses = 1
        else:
            n_pauses = max(int(non_silent.shape[0]) - 1, 0)
            if non_silent[0, 0] > 0:
                n_pauses += 1
            if non_silent[-1, 1] < wav.size:
                n_pauses += 1
        pause_rate = float(n_pauses) / duration_s

        # -- Jitter (F0 cycle-to-cycle variation, Boersma 2001) -------------
        if f0_voiced.size > 4:
            jitter = float(
                np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-9)
            ) * 100.0
        else:
            jitter = 0.0

        # -- Shimmer (amplitude cycle-to-cycle variation) -------------------
        rms = librosa.feature.rms(  # type: ignore[union-attr]
            y=wav, frame_length=self.frame_length, hop_length=self.hop_length
        ).flatten()
        if rms.size > 4:
            shimmer = float(
                np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9)
            ) * 100.0
        else:
            shimmer = 0.0

        # -- HNR approximation via spectral flatness -------------------------
        # Flatness in [0, 1]; 0 = pure tone (harmonic), 1 = white noise.
        # We approximate HNR_dB = 10*log10((1 - flatness) / flatness).
        # Mean flatness over voiced frames for stability.
        flat_mean = float(np.mean(flatness[voiced_mask])) if voiced_mask.any() else float(
            np.mean(flatness)
        ) if flatness.size > 0 else 0.5
        flat_mean = float(np.clip(flat_mean, 1e-6, 1.0 - 1e-6))
        hnr_db = float(10.0 * np.log10((1.0 - flat_mean) / flat_mean))

        return VoiceFeatureVector(
            pitch_mean_hz=float(pitch_mean),
            pitch_std_hz=float(pitch_std),
            speech_rate_syllables_per_s=float(speech_rate),
            voiced_ratio=float(max(0.0, min(1.0, voiced_ratio))),
            pause_rate_per_s=float(max(0.0, pause_rate)),
            jitter_percent=float(max(0.0, jitter)),
            shimmer_percent=float(max(0.0, shimmer)),
            harmonics_to_noise_ratio_db=float(hnr_db),
        )

    # ------------------------------------------------------------------
    # Convenience: load from a WAV file
    # ------------------------------------------------------------------
    @classmethod
    def from_wav_file(
        cls, path: str | Path, *, target_sr: int | None = None
    ) -> VoiceFeatureVector:
        """Load a WAV file, extract features, and return the result.

        Args:
            path: Filesystem path to the audio file.
            target_sr: Optional resampling target.  If ``None``, the audio is
                kept at its native rate.

        Returns:
            A :class:`VoiceFeatureVector`.

        Raises:
            RuntimeError: If neither ``soundfile`` nor ``librosa`` is
                available (the module cannot read the file).
        """
        wav_path = Path(path)
        if _LIBROSA_AVAILABLE:
            wav, sr = librosa.load(  # type: ignore[union-attr]
                str(wav_path), sr=target_sr, mono=True
            )
        elif _SOUNDFILE_AVAILABLE:
            wav, sr = _soundfile.read(str(wav_path), dtype="float32")  # type: ignore[union-attr]
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
        else:
            raise RuntimeError(_INSTALL_HINT)

        extractor = cls()
        return extractor.extract(np.asarray(wav, dtype=np.float32), int(sr))


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------

__all__ = [
    "VoiceFeatureVector",
    "VoiceProsodyExtractor",
]
