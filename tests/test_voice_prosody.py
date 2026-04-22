"""Tests for :mod:`i3.multimodal.voice_real` (Batch F-1).

The suite uses deterministic synthetic waveforms — a pure sine, silence, a
noisy sine, and a modulated sine — so it runs with no recorded-audio
assets.  When ``librosa`` is missing, only the fallback / shape tests run;
every other test is skipped with a clear message.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from i3.multimodal import voice_real
from i3.multimodal.voice_real import (
    VoiceFeatureVector,
    VoiceProsodyExtractor,
    _LIBROSA_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_rate() -> int:
    """Fixed 16 kHz sample rate — enough for speech-band features."""
    return 16_000


@pytest.fixture(scope="module")
def sine_440(sample_rate: int) -> np.ndarray:
    """A 1-second, 440 Hz sine wave (above fmin, below fmax)."""
    t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    return 0.5 * np.sin(2 * np.pi * 150.0 * t).astype(np.float32)


@pytest.fixture(scope="module")
def silence(sample_rate: int) -> np.ndarray:
    """1 second of exact digital silence."""
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture(scope="module")
def noisy_sine(sample_rate: int) -> np.ndarray:
    """Sine wave with additive Gaussian noise (SNR ~ 0 dB)."""
    rng = np.random.default_rng(seed=42)
    t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    sine = 0.3 * np.sin(2 * np.pi * 180.0 * t).astype(np.float32)
    noise = rng.normal(0.0, 0.3, size=sample_rate).astype(np.float32)
    return sine + noise


# ---------------------------------------------------------------------------
# Feature vector contract
# ---------------------------------------------------------------------------

class TestVoiceFeatureVector:
    """Tests for the Pydantic feature-vector holder."""

    def test_zeros_is_eight_dim(self) -> None:
        arr = VoiceFeatureVector.zeros().to_array()
        assert arr.shape == (8,)
        assert arr.dtype == np.float32
        assert np.allclose(arr, 0.0)

    def test_named_fields_round_trip(self) -> None:
        vec = VoiceFeatureVector(
            pitch_mean_hz=150.0,
            pitch_std_hz=20.0,
            speech_rate_syllables_per_s=3.5,
            voiced_ratio=0.9,
            pause_rate_per_s=0.5,
            jitter_percent=1.0,
            shimmer_percent=2.0,
            harmonics_to_noise_ratio_db=12.0,
        )
        arr = vec.to_array()
        assert arr.shape == (8,)
        assert arr[0] == pytest.approx(150.0)
        assert arr[-1] == pytest.approx(12.0)

    def test_pydantic_validation_rejects_out_of_range_voiced_ratio(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VoiceFeatureVector(voiced_ratio=2.0)


# ---------------------------------------------------------------------------
# Module-level / fallback behaviour
# ---------------------------------------------------------------------------

class TestModuleImportFallback:
    """The module must import cleanly when librosa is absent."""

    def test_module_has_expected_public_api(self) -> None:
        assert "VoiceProsodyExtractor" in voice_real.__all__
        assert "VoiceFeatureVector" in voice_real.__all__

    def test_extract_raises_runtime_error_when_librosa_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reload the module with librosa masked; ``.extract`` must raise.

        We set ``sys.modules["librosa"] = None`` before the reload; Python's
        import machinery interprets a ``None`` slot as "import failed" and
        raises ``ImportError`` automatically, which is what the module's
        soft-import guard catches.
        """
        monkeypatch.setitem(sys.modules, "librosa", None)
        reloaded = importlib.reload(voice_real)
        try:
            assert reloaded._LIBROSA_AVAILABLE is False
            extractor = reloaded.VoiceProsodyExtractor()
            with pytest.raises(RuntimeError, match="librosa"):
                extractor.extract(
                    np.zeros(1000, dtype=np.float32), sample_rate=16_000
                )
        finally:
            # Restore the real librosa (if installed) so later tests still work.
            sys.modules.pop("librosa", None)
            importlib.reload(voice_real)


# ---------------------------------------------------------------------------
# Extraction behaviour (requires librosa)
# ---------------------------------------------------------------------------

pytestmark_requires_librosa = pytest.mark.skipif(
    not _LIBROSA_AVAILABLE, reason="librosa is not installed"
)


@pytestmark_requires_librosa
class TestSineExtraction:
    """End-to-end tests on deterministic synthetic waveforms."""

    def test_produces_eight_dim_vector(
        self, sine_440: np.ndarray, sample_rate: int
    ) -> None:
        vec = VoiceProsodyExtractor().extract(sine_440, sample_rate)
        arr = vec.to_array()
        assert arr.shape == (8,)
        assert arr.dtype == np.float32

    def test_sine_has_high_voiced_ratio(
        self, sine_440: np.ndarray, sample_rate: int
    ) -> None:
        vec = VoiceProsodyExtractor().extract(sine_440, sample_rate)
        assert vec.voiced_ratio >= 0.8

    def test_sine_pitch_mean_near_expected(
        self, sine_440: np.ndarray, sample_rate: int
    ) -> None:
        vec = VoiceProsodyExtractor().extract(sine_440, sample_rate)
        # The fixture is 150 Hz; librosa.yin is accurate to ~5 Hz on clean sines.
        assert 120.0 <= vec.pitch_mean_hz <= 200.0

    def test_silence_has_low_voiced_ratio(
        self, silence: np.ndarray, sample_rate: int
    ) -> None:
        vec = VoiceProsodyExtractor().extract(silence, sample_rate)
        assert vec.voiced_ratio == pytest.approx(0.0, abs=0.05)

    def test_silence_high_pause_rate(
        self, silence: np.ndarray, sample_rate: int
    ) -> None:
        vec = VoiceProsodyExtractor().extract(silence, sample_rate)
        # Pure silence produces at least one pause region per second.
        assert vec.pause_rate_per_s > 0.0

    def test_empty_waveform_returns_zero_vector(self, sample_rate: int) -> None:
        vec = VoiceProsodyExtractor().extract(np.asarray([], dtype=np.float32), sample_rate)
        assert np.allclose(vec.to_array(), 0.0)


@pytestmark_requires_librosa
class TestDeterminism:
    """Extraction must be deterministic under fixed RNG seeds."""

    def test_same_input_same_output(
        self, noisy_sine: np.ndarray, sample_rate: int
    ) -> None:
        e = VoiceProsodyExtractor()
        a = e.extract(noisy_sine, sample_rate).to_array()
        b = e.extract(noisy_sine.copy(), sample_rate).to_array()
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_stereo_input_averaged_to_mono(
        self, sine_440: np.ndarray, sample_rate: int
    ) -> None:
        stereo = np.stack([sine_440, sine_440], axis=-1)
        vec_mono = VoiceProsodyExtractor().extract(sine_440, sample_rate)
        vec_stereo = VoiceProsodyExtractor().extract(stereo, sample_rate)
        np.testing.assert_allclose(
            vec_mono.to_array(), vec_stereo.to_array(), rtol=1e-5, atol=1e-5
        )


@pytestmark_requires_librosa
class TestInputValidation:
    """The extractor must reject bad inputs with clear errors."""

    def test_negative_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            VoiceProsodyExtractor().extract(np.zeros(100, dtype=np.float32), -1)

    def test_non_ndarray_input_raises(self) -> None:
        with pytest.raises(ValueError, match="ndarray"):
            VoiceProsodyExtractor().extract([0.0, 0.1, 0.2], 16_000)  # type: ignore[arg-type]


@pytestmark_requires_librosa
class TestHarmonicsRatio:
    """HNR should distinguish tonal vs noisy signals."""

    def test_sine_has_higher_hnr_than_noise(
        self, sine_440: np.ndarray, sample_rate: int
    ) -> None:
        rng = np.random.default_rng(seed=1234)
        noise = rng.normal(0.0, 0.3, size=sine_440.size).astype(np.float32)
        sine_hnr = VoiceProsodyExtractor().extract(sine_440, sample_rate).harmonics_to_noise_ratio_db
        noise_hnr = VoiceProsodyExtractor().extract(noise, sample_rate).harmonics_to_noise_ratio_db
        assert sine_hnr > noise_hnr
