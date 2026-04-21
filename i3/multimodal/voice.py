"""Voice-prosody feature extractor.

Computes a fixed-length 8-dim prosody feature group from a raw audio waveform.
The output shape matches the keystroke-dynamics group in
:class:`i3.interaction.types.InteractionFeatureVector` so the same TCN encoder
can consume either (or both, after fusion — see :mod:`i3.multimodal.fusion`).

All signal-processing work is delegated to ``librosa`` (soft-imported).  When
librosa is not installed, the extractor returns a zero vector and emits a
``logger.warning``; it never raises.  This mirrors the brief's soft-import
convention (§11 *Future Work*).

References
----------
* Eyben, F., Weninger, F., Schuller, B. (2013). *Recent developments in
  openSMILE, the Munich open-source multimedia feature extractor.*  ACM MM.
* Titze, I. R. (1994). *Principles of Voice Production.*  Prentice-Hall.
* McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in
  Python.*  SciPy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import guard
# ---------------------------------------------------------------------------

try:
    import librosa  # type: ignore[import-not-found]

    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    librosa = None  # type: ignore[assignment]
    _LIBROSA_AVAILABLE = False


_INSTALL_HINT = (
    "librosa is not installed. Install the future-work group with "
    "`poetry install --with future-work` to enable voice feature extraction."
)


# ---------------------------------------------------------------------------
# 8-dim voice feature group
# ---------------------------------------------------------------------------

@dataclass
class VoiceFeatureVector:
    """8-dim prosody feature group.

    The group mirrors the shape of the keystroke-dynamics group so the
    existing TCN encoder can consume a voice-driven stream without any
    architectural change.

    Attributes:
        pitch_mean_hz: Mean fundamental frequency F₀ (Hz), normalised to
            ``[0, 1]`` by a 500 Hz ceiling (typical adult speaker upper bound).
        pitch_std_hz: Standard deviation of F₀, normalised by 200 Hz.
        speaking_rate_sps: Estimated syllables per second, normalised by 10.
        filled_pause_ratio: Fraction of frames classified as non-speech
            voiced material (umms/uhhs surrogate via voiced-frame density).
        speech_intensity_db: Root-mean-square energy on a normalised dB scale.
        voicing_ratio: Fraction of frames with a detected F₀.
        jitter_local: Cycle-to-cycle F₀ perturbation, proxy for vocal effort.
        shimmer_local: Cycle-to-cycle amplitude perturbation.

    All fields are clamped to ``[0, 1]`` before being returned.
    """

    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    speaking_rate_sps: float = 0.0
    filled_pause_ratio: float = 0.0
    speech_intensity_db: float = 0.0
    voicing_ratio: float = 0.0
    jitter_local: float = 0.0
    shimmer_local: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return an 8-element ``float32`` numpy array in declaration order."""
        return np.array(
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> VoiceFeatureVector:
        """Return the zero vector (used on missing / failed inputs)."""
        return cls()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    """Clamp a float to the closed interval ``[0, 1]`` with NaN guard."""
    if x != x:  # NaN check without math import
        return 0.0
    return float(max(0.0, min(1.0, x)))


class VoiceFeatureExtractor:
    """Compute :class:`VoiceFeatureVector` from a waveform.

    Parameters:
        frame_length: STFT frame length used for F₀ / energy estimation.
        hop_length: STFT hop length.
        pitch_ceiling_hz: Normalisation ceiling for the mean-pitch feature.
        pitch_std_ceiling_hz: Normalisation ceiling for the pitch std feature.
        rate_ceiling_sps: Syllables-per-second ceiling for normalisation.
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        pitch_ceiling_hz: float = 500.0,
        pitch_std_ceiling_hz: float = 200.0,
        rate_ceiling_sps: float = 10.0,
    ) -> None:
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pitch_ceiling_hz = pitch_ceiling_hz
        self.pitch_std_ceiling_hz = pitch_std_ceiling_hz
        self.rate_ceiling_sps = rate_ceiling_sps

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def extract(self, waveform: np.ndarray, sample_rate: int) -> VoiceFeatureVector:
        """Extract the 8-dim prosody vector from *waveform*.

        Args:
            waveform: 1-D numpy array of PCM samples (float32 or int16).
            sample_rate: Sampling rate in Hz.

        Returns:
            A :class:`VoiceFeatureVector`.  Returns zeros with a warning when
            ``librosa`` is unavailable or the waveform is empty.
        """
        if not _LIBROSA_AVAILABLE:
            logger.warning("VoiceFeatureExtractor: %s", _INSTALL_HINT)
            return VoiceFeatureVector.zeros()

        if waveform.size == 0:
            logger.warning("VoiceFeatureExtractor: empty waveform")
            return VoiceFeatureVector.zeros()

        wav = waveform.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        # -- F0 track ------------------------------------------------------
        # pyin returns (f0, voiced_flag, voiced_prob); ignore everything but f0.
        f0, voiced_flag, _prob = librosa.pyin(  # type: ignore[union-attr]
            wav,
            fmin=float(librosa.note_to_hz("C2")),  # type: ignore[union-attr]
            fmax=float(librosa.note_to_hz("C7")),  # type: ignore[union-attr]
            sr=sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        voicing_ratio = (
            float(np.mean(voiced_flag.astype(np.float32)))
            if voiced_flag is not None and voiced_flag.size > 0
            else 0.0
        )

        pitch_mean = float(np.mean(f0_valid)) if f0_valid.size > 0 else 0.0
        pitch_std = float(np.std(f0_valid)) if f0_valid.size > 0 else 0.0

        # -- Intensity (RMS energy in dB) ---------------------------------
        rms = librosa.feature.rms(  # type: ignore[union-attr]
            y=wav, frame_length=self.frame_length, hop_length=self.hop_length
        )
        rms_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
        # Map 0 .. 1 RMS into a compressive dB space, then clamp.
        db_norm = (20.0 * np.log10(rms_mean + 1e-9) + 60.0) / 60.0

        # -- Speaking rate (crude: onsets per second) ---------------------
        onset_env = librosa.onset.onset_strength(  # type: ignore[union-attr]
            y=wav, sr=sample_rate, hop_length=self.hop_length
        )
        duration_s = max(len(wav) / float(sample_rate), 1e-6)
        rate_sps = float(np.sum(onset_env > onset_env.mean())) / duration_s

        # -- Filled-pause surrogate ---------------------------------------
        # Frames that are voiced but have near-zero F0 variance locally
        # approximate a held-vowel filler ("umm"/"uhh").
        if f0_valid.size > 4:
            diffs = np.abs(np.diff(f0_valid))
            filled = float(np.mean(diffs < 2.0))
        else:
            filled = 0.0

        # -- Jitter / shimmer ---------------------------------------------
        jitter = (
            float(np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid) + 1e-6))
            if f0_valid.size > 4
            else 0.0
        )
        if rms.size > 4:
            rms_1d = rms.flatten()
            shimmer = float(np.mean(np.abs(np.diff(rms_1d))) / (np.mean(rms_1d) + 1e-6))
        else:
            shimmer = 0.0

        return VoiceFeatureVector(
            pitch_mean_hz=_clip01(pitch_mean / self.pitch_ceiling_hz),
            pitch_std_hz=_clip01(pitch_std / self.pitch_std_ceiling_hz),
            speaking_rate_sps=_clip01(rate_sps / self.rate_ceiling_sps),
            filled_pause_ratio=_clip01(filled),
            speech_intensity_db=_clip01(db_norm),
            voicing_ratio=_clip01(voicing_ratio),
            jitter_local=_clip01(jitter),
            shimmer_local=_clip01(shimmer),
        )
