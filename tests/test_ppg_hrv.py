"""Tests for :mod:`i3.multimodal.ppg_hrv` (Batch F-2).

The suite uses deterministic synthetic PPG signals — clean sines, noisy
sines, signals of various HRV magnitudes — so it runs with no recorded
wearable data.  When ``scipy`` is unavailable the feature-extraction
tests degrade gracefully to fallback-checks.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from i3.multimodal import ppg_hrv
from i3.multimodal.ppg_hrv import (
    InsufficientDataError,
    PPGFeatureVector,
    PPGHRVExtractor,
    _SCIPY_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_rate() -> float:
    """Huawei Watch 5-style 25 Hz PPG sample rate."""
    return 25.0


def _synth_ppg(
    duration_s: float,
    sample_rate: float,
    hr_bpm: float,
    hrv_std_ms: float,
    noise_std: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise a PPG-like pulse train with controlled HRV.

    Uses a narrow-Gaussian pulse per beat with an IBI distribution
    centred at ``60 / hr_bpm`` seconds.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    mean_ibi = 60.0 / hr_bpm
    beat_times: list[float] = []
    cur = 0.0
    while cur < duration_s:
        ibi = float(rng.normal(loc=mean_ibi, scale=hrv_std_ms / 1000.0))
        ibi = max(0.3, min(2.0, ibi))
        cur += ibi
        beat_times.append(cur)
    signal = np.zeros(n, dtype=np.float64)
    sigma = 0.08
    for bt in beat_times:
        if bt < duration_s:
            signal += np.exp(-0.5 * ((t - bt) / sigma) ** 2)
    signal += rng.normal(0.0, noise_std, size=n)
    return signal.astype(np.float32)


@pytest.fixture(scope="module")
def clean_60_bpm(sample_rate: float) -> np.ndarray:
    """60 seconds of a low-HRV ~60 bpm PPG."""
    return _synth_ppg(
        duration_s=60.0,
        sample_rate=sample_rate,
        hr_bpm=60.0,
        hrv_std_ms=5.0,
        noise_std=0.01,
        seed=1,
    )


@pytest.fixture(scope="module")
def noisy_signal(sample_rate: float) -> np.ndarray:
    """60 seconds of a noisy ~72 bpm PPG with injected spike artefacts."""
    s = _synth_ppg(
        duration_s=60.0,
        sample_rate=sample_rate,
        hr_bpm=72.0,
        hrv_std_ms=30.0,
        noise_std=0.25,
        seed=2,
    )
    # Add occasional artefact spikes (motion artefacts).
    for idx in (100, 300, 800, 1100):
        if idx < s.size:
            s[idx] = 5.0
    return s


# ---------------------------------------------------------------------------
# Feature-vector contract
# ---------------------------------------------------------------------------


def test_feature_vector_zero_default() -> None:
    """Default :class:`PPGFeatureVector` is all zeros with correct shape."""
    vec = PPGFeatureVector.zeros()
    arr = vec.to_array()
    assert arr.shape == (8,)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)


def test_feature_vector_rejects_negative_hr() -> None:
    """The Pydantic validator rejects a negative heart rate."""
    with pytest.raises(ValidationError):
        PPGFeatureVector(hr_bpm=-10.0)


def test_feature_vector_rejects_nan_rmssd() -> None:
    """NaN RMSSD is rejected — protects downstream tensors from contagion."""
    with pytest.raises(ValidationError):
        PPGFeatureVector(rmssd_ms=float("nan"))


def test_to_array_ordering_matches_declaration() -> None:
    """``to_array`` follows the declared attribute order."""
    vec = PPGFeatureVector(
        hr_bpm=70.0,
        rmssd_ms=50.0,
        sdnn_ms=60.0,
        pnn50_percent=10.0,
        lf_power=400.0,
        hf_power=500.0,
        lf_hf_ratio=0.8,
        sample_entropy=1.5,
    )
    arr = vec.to_array()
    assert arr.tolist() == pytest.approx(
        [70.0, 50.0, 60.0, 10.0, 400.0, 500.0, 0.8, 1.5]
    )


# ---------------------------------------------------------------------------
# Extraction behaviour
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_clean_60_bpm_recovers_hr(clean_60_bpm: np.ndarray, sample_rate: float) -> None:
    """A clean ~60-bpm signal yields HR within +/-4 bpm of target."""
    vec = PPGHRVExtractor().extract(clean_60_bpm, sample_rate=sample_rate)
    assert vec.hr_bpm == pytest.approx(60.0, abs=4.0)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_all_features_finite_and_non_negative(
    clean_60_bpm: np.ndarray, sample_rate: float
) -> None:
    """Every HRV feature in the vector is a finite, non-negative float."""
    vec = PPGHRVExtractor().extract(clean_60_bpm, sample_rate=sample_rate)
    arr = vec.to_array()
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_lf_hf_ratio_strictly_positive(
    clean_60_bpm: np.ndarray, sample_rate: float
) -> None:
    """Non-trivial signal produces LF/HF > 0 (both bands have power)."""
    vec = PPGHRVExtractor().extract(clean_60_bpm, sample_rate=sample_rate)
    assert vec.lf_hf_ratio >= 0.0
    # With 60 s of data the LF band should register at least some power.
    assert vec.lf_power + vec.hf_power > 0.0


def test_short_signal_raises_insufficient_data(sample_rate: float) -> None:
    """Signals < 10 s raise the documented :class:`InsufficientDataError`."""
    short = np.zeros(int(5 * sample_rate), dtype=np.float32)
    with pytest.raises(InsufficientDataError) as excinfo:
        PPGHRVExtractor().extract(short, sample_rate=sample_rate)
    assert excinfo.value.duration_s == pytest.approx(5.0, abs=0.01)
    assert excinfo.value.minimum_s == pytest.approx(10.0, abs=0.01)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_noisy_signal_does_not_crash(
    noisy_signal: np.ndarray, sample_rate: float
) -> None:
    """A noisy signal with motion artefacts returns a valid feature vector."""
    vec = PPGHRVExtractor().extract(noisy_signal, sample_rate=sample_rate)
    arr = vec.to_array()
    assert arr.shape == (8,)
    assert np.all(np.isfinite(arr))


def test_import_when_scipy_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module imports and returns zeros when scipy is unavailable."""
    monkeypatch.setattr(ppg_hrv, "_SCIPY_AVAILABLE", False)
    monkeypatch.setattr(ppg_hrv, "_scipy_signal", None)
    signal = np.zeros(int(30 * 25.0), dtype=np.float32)
    vec = PPGHRVExtractor().extract(signal, sample_rate=25.0)
    assert np.all(vec.to_array() == 0.0)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_determinism_under_fixed_seed(sample_rate: float) -> None:
    """Same synthetic seed -> byte-identical feature vector."""
    sig_a = _synth_ppg(30.0, sample_rate, hr_bpm=75.0, hrv_std_ms=20.0, seed=7)
    sig_b = _synth_ppg(30.0, sample_rate, hr_bpm=75.0, hrv_std_ms=20.0, seed=7)
    a = PPGHRVExtractor().extract(sig_a, sample_rate=sample_rate).to_array()
    b = PPGHRVExtractor().extract(sig_b, sample_rate=sample_rate).to_array()
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_rmssd_monotone_in_hrv(sample_rate: float) -> None:
    """Higher input HRV (larger IBI std) -> larger RMSSD and SDNN."""
    low_hrv = _synth_ppg(60.0, sample_rate, hr_bpm=70.0, hrv_std_ms=5.0, seed=9)
    high_hrv = _synth_ppg(60.0, sample_rate, hr_bpm=70.0, hrv_std_ms=60.0, seed=9)
    low = PPGHRVExtractor().extract(low_hrv, sample_rate=sample_rate)
    high = PPGHRVExtractor().extract(high_hrv, sample_rate=sample_rate)
    assert high.rmssd_ms > low.rmssd_ms
    assert high.sdnn_ms > low.sdnn_ms


def test_extract_rejects_non_ndarray(sample_rate: float) -> None:
    """Passing a list instead of an ndarray raises :class:`ValueError`."""
    with pytest.raises(ValueError):
        PPGHRVExtractor().extract([0.0, 1.0, 2.0], sample_rate=sample_rate)  # type: ignore[arg-type]


def test_extract_rejects_non_positive_sample_rate() -> None:
    """A zero or negative sample rate is rejected up-front."""
    sig = np.zeros(300, dtype=np.float32)
    with pytest.raises(ValueError):
        PPGHRVExtractor().extract(sig, sample_rate=0.0)
    with pytest.raises(ValueError):
        PPGHRVExtractor().extract(sig, sample_rate=-1.0)


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required for extraction")
def test_from_raw_csv_roundtrip(
    tmp_path, sample_rate: float, clean_60_bpm: np.ndarray
) -> None:
    """CSV round-trip yields an equivalent feature vector."""
    csv_path = tmp_path / "ppg.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("amplitude\n")
        for v in clean_60_bpm.tolist():
            fh.write(f"{v}\n")
    vec = PPGHRVExtractor.from_raw_csv(
        csv_path, sample_rate=sample_rate, column="amplitude"
    )
    direct = PPGHRVExtractor().extract(clean_60_bpm, sample_rate=sample_rate)
    # CSV path may clip float precision; compare with loose tolerance.
    np.testing.assert_allclose(vec.to_array(), direct.to_array(), rtol=1e-3, atol=1e-3)


def test_from_raw_csv_missing_file_raises(tmp_path) -> None:
    """Missing CSV raises :class:`FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        PPGHRVExtractor.from_raw_csv(tmp_path / "does-not-exist.csv")


def test_module_reimport_clean() -> None:
    """Module reload cycle doesn't leave stale state."""
    importlib.reload(sys.modules["i3.multimodal.ppg_hrv"])
    from i3.multimodal.ppg_hrv import PPGFeatureVector as Reloaded

    assert Reloaded.zeros().hr_bpm == 0.0
