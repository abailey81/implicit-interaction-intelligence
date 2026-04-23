"""Runnable PPG / HRV feature extractor (Batch F-2).

Photoplethysmography (PPG) is the optical technique used by every modern
wrist-worn wearable — including the **Huawei Watch 5** three-in-one
ECG + PPG + pressure sensor — to recover a beat-to-beat pulse waveform from
light absorption at the skin.  The inter-beat intervals (IBIs) derived from
successive systolic peaks are the raw material for heart-rate variability
(HRV) analysis: the oldest and most evidence-backed non-invasive biomarker
for autonomic-nervous-system balance, cognitive load, and stress.

This module is the wearable-signal analogue of
:class:`i3.multimodal.voice_real.VoiceProsodyExtractor`.  It consumes a raw
PPG array + sample rate and emits an eight-dimensional feature vector that
plugs directly into the I³ TCN encoder — the same 8-dim per-modality
contract used by keystroke-dynamics, voice-prosody, accelerometer, touch,
and gaze extractors.

The eight features (time-domain + frequency-domain + non-linear):

1. ``hr_bpm`` — mean heart rate derived from IBIs.
2. ``rmssd_ms`` — root mean square of successive IBI differences.
3. ``sdnn_ms`` — standard deviation of NN intervals.
4. ``pnn50_percent`` — percentage of successive IBIs differing by > 50 ms.
5. ``lf_power`` — low-frequency (0.04–0.15 Hz) spectral power.
6. ``hf_power`` — high-frequency (0.15–0.4 Hz) spectral power.
7. ``lf_hf_ratio`` — sympathovagal balance (LF/HF).
8. ``sample_entropy`` — short-sample approximate entropy.

All signal-processing work is delegated to ``scipy.signal`` (already a
transitive dependency via ``scikit-learn``) and is **soft-imported**:
importing this module never fails.  When scipy is absent, the extractor
returns a zero vector and emits a ``logger.warning`` so the pipeline
degrades gracefully.

References
----------
* Task Force of the European Society of Cardiology and the North American
  Society of Pacing and Electrophysiology (1996).  *Heart rate variability:
  standards of measurement, physiological interpretation, and clinical use.*
  Circulation 93(5), 1043–1065.  (Canonical HRV measurement standard.)
* Shaffer, F., Ginsberg, J. P. (2017).  *An overview of heart rate
  variability metrics and norms.*  Frontiers in Public Health 5:258.
* Makivic, B., Djordjevic Nikic, M., Willis, M. S. (2013).  *Heart rate
  variability (HRV) as a tool for diagnostic and monitoring performance in
  sport and physical activities.*  Journal of Exercise Physiology Online
  16(3), 103–131.
* Allen, J. (2007).  *Photoplethysmography and its application in clinical
  physiological measurement.*  Physiological Measurement 28(3), R1–R39.
* Lu, G., Yang, F., Taylor, J. A., Stein, J. F. (2009).  *A comparison of
  photoplethysmography and ECG recording to analyse heart rate variability
  in healthy subjects.*  Journal of Medical Engineering & Technology
  33(8), 634–641.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import scipy.signal.  scipy ships transitively with scikit-learn, but
# we never want an import-time crash in environments where it is absent.
# ---------------------------------------------------------------------------

try:
    from scipy import signal as _scipy_signal  # type: ignore[import-not-found]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    _scipy_signal = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import numpy.typing as npt

    NDArrayF32 = npt.NDArray[np.float32]
else:
    NDArrayF32 = np.ndarray


_INSTALL_HINT = (
    "scipy is not installed. Install scipy (it ships transitively with "
    "scikit-learn) to enable PPG / HRV feature extraction."
)

# Physiological bounds used to sanity-check peak detection.  Below ~30 bpm
# we assume the wearable has lost contact; above ~220 bpm we assume noise.
_MIN_HR_BPM: float = 30.0
_MAX_HR_BPM: float = 220.0
_MIN_IBI_S: float = 60.0 / _MAX_HR_BPM
_MAX_IBI_S: float = 60.0 / _MIN_HR_BPM

# Frequency-domain HRV band edges from the 1996 Task Force standard.
_LF_BAND_HZ: tuple[float, float] = (0.04, 0.15)
_HF_BAND_HZ: tuple[float, float] = (0.15, 0.40)

# Minimum recording duration for meaningful HRV estimation.  The 1996 Task
# Force recommends at least 2 minutes for short-term HRV, but we allow
# 10 s as an absolute floor for the time-domain features and warn below
# that.  Anything < 10 s is outright rejected.
_MIN_SIGNAL_DURATION_S: float = 10.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InsufficientDataError(ValueError):
    """Raised when the PPG signal is too short for meaningful HRV analysis.

    HRV features require at least a few cardiac cycles; by convention this
    extractor rejects signals shorter than ``_MIN_SIGNAL_DURATION_S`` seconds
    (10 s).  Attempting to analyse shorter windows silently would produce
    HRV values with nonsense confidence intervals.

    Attributes:
        duration_s: Actual duration of the offered signal, in seconds.
        minimum_s: The minimum duration this extractor requires.
    """

    def __init__(self, duration_s: float, minimum_s: float = _MIN_SIGNAL_DURATION_S) -> None:
        self.duration_s = float(duration_s)
        self.minimum_s = float(minimum_s)
        super().__init__(
            f"PPG signal too short: {duration_s:.2f}s < minimum {minimum_s:.2f}s "
            "required for HRV feature extraction (Task Force 1996)."
        )


# ---------------------------------------------------------------------------
# Pydantic feature vector
# ---------------------------------------------------------------------------


class PPGFeatureVector(BaseModel):
    """Eight-dimensional HRV feature group extracted from a PPG waveform.

    The attribute ordering is significant — :meth:`to_array` relies on it —
    and mirrors the keystroke-dynamics group so the same TCN encoder can
    ingest wearable HRV unchanged.

    Attributes:
        hr_bpm: Mean heart rate from inter-beat intervals (beats / minute).
            Physiological range roughly 30–220 bpm.
        rmssd_ms: Root mean square of successive IBI differences
            (milliseconds).  A canonical parasympathetic / vagal-tone index.
        sdnn_ms: Standard deviation of NN intervals (milliseconds).  Captures
            overall HRV.
        pnn50_percent: Percentage of successive IBIs differing by more than
            50 ms.  A second parasympathetic index.
        lf_power: Spectral power in the 0.04–0.15 Hz band (ms²).  Primarily
            reflects baroreflex modulation.
        hf_power: Spectral power in the 0.15–0.40 Hz band (ms²).  Dominantly
            parasympathetic (respiratory sinus arrhythmia).
        lf_hf_ratio: LF / HF ratio.  Approximates sympathovagal balance.
        sample_entropy: Sample entropy of the IBI series (dimensionless).
            Captures signal regularity / complexity.
    """

    hr_bpm: float = Field(default=0.0, ge=0.0, le=300.0)
    rmssd_ms: float = Field(default=0.0, ge=0.0)
    sdnn_ms: float = Field(default=0.0, ge=0.0)
    pnn50_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    lf_power: float = Field(default=0.0, ge=0.0)
    hf_power: float = Field(default=0.0, ge=0.0)
    lf_hf_ratio: float = Field(default=0.0, ge=0.0)
    sample_entropy: float = Field(default=0.0, ge=0.0)

    @field_validator("hr_bpm", "rmssd_ms", "sdnn_ms", "lf_power", "hf_power")
    @classmethod
    def _reject_nan(cls, v: float) -> float:
        """Reject NaN / inf so downstream tensors stay sane."""
        if not np.isfinite(v):
            raise ValueError(f"HRV feature must be finite, got {v!r}")
        return float(v)

    def to_array(self) -> NDArrayF32:
        """Return the feature vector as an 8-element ``float32`` array.

        Returns:
            1-D ``np.ndarray`` of shape ``(8,)`` in declaration order.
        """
        return np.asarray(
            [
                self.hr_bpm,
                self.rmssd_ms,
                self.sdnn_ms,
                self.pnn50_percent,
                self.lf_power,
                self.hf_power,
                self.lf_hf_ratio,
                self.sample_entropy,
            ],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> PPGFeatureVector:
        """Return a zero-valued vector (used on missing input or fallback)."""
        return cls()


# ---------------------------------------------------------------------------
# Helper signal-processing primitives (pure numpy fallbacks + scipy paths)
# ---------------------------------------------------------------------------


def _bandpass_ppg(
    signal: np.ndarray,
    sample_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 5.0,
) -> np.ndarray:
    """Band-pass filter the PPG waveform around plausible pulse frequencies.

    A 4-th order Butterworth filter is the de-facto standard for PPG
    pre-processing (Allen 2007).  We pass 0.5–5 Hz which covers 30–300 bpm.
    When scipy is unavailable or the signal is too short for ``filtfilt``,
    we fall back to a mean-centred signal.

    Args:
        signal: 1-D numpy array.
        sample_rate: Sampling rate in Hz.
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.

    Returns:
        A 1-D numpy array (same shape as ``signal``), detrended and filtered.
    """
    x = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    if not _SCIPY_AVAILABLE or _scipy_signal is None:
        return x
    nyquist = 0.5 * float(sample_rate)
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if not low < high:  # bogus rate
        return x
    try:
        b, a = _scipy_signal.butter(N=4, Wn=[low, high], btype="bandpass")
        # filtfilt needs a minimum length — fall back to raw detrend below.
        pad = 3 * (max(len(a), len(b)) - 1)
        if x.size <= pad:
            return x
        return _scipy_signal.filtfilt(b, a, x)
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
        logger.debug("scipy filtfilt failed (%s); falling back to raw signal", exc)
        return x


def _detect_peaks(
    filtered: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Detect systolic PPG peaks.

    Uses ``scipy.signal.find_peaks`` with an adaptive amplitude threshold
    (0.3 * std deviation above the mean) and an enforced minimum inter-peak
    distance derived from the physiological maximum heart rate.  Falls
    back to a simple numpy diff-sign algorithm when scipy is absent.

    Args:
        filtered: Band-passed PPG signal.
        sample_rate: Sampling rate in Hz.

    Returns:
        A 1-D numpy array of integer peak indices.
    """
    if filtered.size == 0:
        return np.asarray([], dtype=np.int64)
    # Minimum samples between peaks -> maximum physiological HR.
    min_distance = max(int(round(sample_rate * _MIN_IBI_S)), 1)
    threshold = float(np.mean(filtered) + 0.3 * np.std(filtered))

    if _SCIPY_AVAILABLE and _scipy_signal is not None:
        peaks, _ = _scipy_signal.find_peaks(
            filtered, distance=min_distance, height=threshold
        )
        return np.asarray(peaks, dtype=np.int64)

    # Pure-numpy fallback: look for points higher than both neighbours and
    # above threshold, enforce min-distance greedily.
    arr = np.asarray(filtered, dtype=np.float64)
    candidates = np.where(
        (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]) & (arr[1:-1] > threshold)
    )[0] + 1
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)
    selected: list[int] = [int(candidates[0])]
    for idx in candidates[1:]:
        if int(idx) - selected[-1] >= min_distance:
            selected.append(int(idx))
    return np.asarray(selected, dtype=np.int64)


def _ibi_series_from_peaks(peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Convert peak indices to an IBI series in seconds.

    Args:
        peaks: 1-D numpy array of peak indices.
        sample_rate: Sampling rate in Hz.

    Returns:
        1-D numpy array of IBIs (seconds).  Outliers outside the
        physiological range are dropped.
    """
    if peaks.size < 2:
        return np.asarray([], dtype=np.float64)
    ibis = np.diff(peaks.astype(np.float64)) / float(sample_rate)
    mask = (ibis >= _MIN_IBI_S) & (ibis <= _MAX_IBI_S)
    return ibis[mask]


def _sample_entropy(series: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Approximate sample entropy of a 1-D series.

    This is the Richman & Moorman (2000) definition used throughout the
    HRV literature.  For efficiency we use a coarse O(N²) nested-loop
    implementation — N is typically < 300 IBIs.

    Args:
        series: 1-D IBI array (seconds).
        m: Template length (default 2).
        r: Tolerance.  Defaults to 0.2 * std(series), the standard choice.

    Returns:
        A non-negative float.  Returns ``0.0`` when the series is too
        short to be meaningful (< ``m + 1`` samples).
    """
    n = int(series.size)
    if n < m + 2:
        return 0.0
    x = np.asarray(series, dtype=np.float64)
    if r is None:
        r = 0.2 * float(np.std(x))
    if r <= 0:
        return 0.0

    def _phi(length: int) -> int:
        count = 0
        templates = np.stack([x[i : i + length] for i in range(n - length)], axis=0)
        for i in range(templates.shape[0]):
            # Chebyshev distance
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            # Exclude self-match at position i (Richman-Moorman convention).
            count += int(np.sum(dist <= r) - 1)
        return count

    b = _phi(m)
    a = _phi(m + 1)
    if a <= 0 or b <= 0:
        return 0.0
    return float(-np.log(a / b))


def _welch_bandpower(
    ibis_s: np.ndarray, band: tuple[float, float], fs: float = 4.0
) -> float:
    """Compute spectral power of the IBI series within a frequency band.

    Interpolates the unevenly-sampled IBI series to a 4 Hz uniform grid
    (the standard rate for short-term HRV PSD), then applies Welch's
    method.  When scipy is absent we use a numpy FFT fallback.

    Args:
        ibis_s: IBI series in seconds.
        band: ``(low_hz, high_hz)`` tuple defining the band of interest.
        fs: Target uniform sampling rate for the interpolated series.

    Returns:
        Power (ms²) within the band.  ``0.0`` when the IBI series is too
        short (< 4 samples) or the band is empty.
    """
    if ibis_s.size < 4:
        return 0.0
    times = np.cumsum(ibis_s)
    if times[-1] <= 0:
        return 0.0
    ibis_ms = ibis_s * 1000.0
    uniform_t = np.arange(0.0, times[-1], 1.0 / fs)
    if uniform_t.size < 8:
        return 0.0
    uniform_x = np.interp(uniform_t, times, ibis_ms)

    low, high = band
    if _SCIPY_AVAILABLE and _scipy_signal is not None:
        # nperseg must not exceed the signal length.
        nperseg = int(min(256, uniform_x.size))
        freqs, psd = _scipy_signal.welch(uniform_x, fs=fs, nperseg=nperseg)
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))

    # Numpy FFT fallback.
    x = uniform_x - np.mean(uniform_x)
    n = int(x.size)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x)) ** 2 / max(n, 1)
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return 0.0
    return float(np.trapz(spectrum[mask], freqs[mask]))


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class PPGHRVExtractor:
    """Compute :class:`PPGFeatureVector` from a raw PPG waveform.

    The extractor is stateless — a single instance may be reused across
    sessions — and deterministic for a given input.

    Args:
        bandpass_low_hz: Lower cutoff for the PPG pre-filter (default 0.5 Hz).
        bandpass_high_hz: Upper cutoff (default 5.0 Hz, covers up to ~300 bpm).
        sample_entropy_m: Template length for :func:`_sample_entropy`.
    """

    def __init__(
        self,
        bandpass_low_hz: float = 0.5,
        bandpass_high_hz: float = 5.0,
        sample_entropy_m: int = 2,
    ) -> None:
        self.bandpass_low_hz = float(bandpass_low_hz)
        self.bandpass_high_hz = float(bandpass_high_hz)
        self.sample_entropy_m = int(sample_entropy_m)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def extract(self, ppg: np.ndarray, sample_rate: float) -> PPGFeatureVector:
        """Extract an 8-dim HRV feature vector from a raw PPG signal.

        Args:
            ppg: 1-D numpy array of PPG samples.  Multi-channel inputs are
                averaged to mono.
            sample_rate: Sampling rate in Hz.  Must be positive.  Typical
                wrist wearables run 25 Hz (Huawei Watch 5 PPG), 50 Hz, or
                100 Hz.

        Returns:
            A :class:`PPGFeatureVector`.

        Raises:
            ValueError: If ``sample_rate`` is non-positive or ``ppg`` is not
                a numpy array.
            InsufficientDataError: If the signal is shorter than 10 s.
        """
        if not isinstance(ppg, np.ndarray):
            raise ValueError(
                f"ppg must be numpy.ndarray, got {type(ppg).__name__}"
            )
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        if not _SCIPY_AVAILABLE:
            logger.warning(
                "scipy unavailable: PPGHRVExtractor returning zero feature "
                "vector. %s",
                _INSTALL_HINT,
            )
            return PPGFeatureVector.zeros()

        sig = np.asarray(ppg, dtype=np.float32)
        if sig.ndim > 1:
            sig = sig.mean(axis=-1).astype(np.float32)

        duration_s = float(sig.size) / float(sample_rate)
        if duration_s < _MIN_SIGNAL_DURATION_S:
            raise InsufficientDataError(duration_s=duration_s)

        # Stage 1: band-pass + adaptive-threshold peak detection.
        filtered = _bandpass_ppg(
            sig,
            sample_rate=float(sample_rate),
            low_hz=self.bandpass_low_hz,
            high_hz=self.bandpass_high_hz,
        )
        peaks = _detect_peaks(filtered, sample_rate=float(sample_rate))
        ibis_s = _ibi_series_from_peaks(peaks, sample_rate=float(sample_rate))

        if ibis_s.size < 3:
            logger.warning(
                "PPGHRVExtractor: only %d plausible IBIs detected; returning "
                "zero vector.",
                int(ibis_s.size),
            )
            return PPGFeatureVector.zeros()

        # Stage 2: HRV features (time-domain).
        ibis_ms = ibis_s * 1000.0
        hr_bpm = float(60.0 / np.mean(ibis_s))
        diff_ms = np.diff(ibis_ms)
        rmssd_ms = float(np.sqrt(np.mean(diff_ms ** 2))) if diff_ms.size > 0 else 0.0
        sdnn_ms = float(np.std(ibis_ms)) if ibis_ms.size > 1 else 0.0
        pnn50 = (
            float(np.sum(np.abs(diff_ms) > 50.0)) / float(diff_ms.size) * 100.0
            if diff_ms.size > 0
            else 0.0
        )

        # Frequency-domain.
        lf = _welch_bandpower(ibis_s, _LF_BAND_HZ)
        hf = _welch_bandpower(ibis_s, _HF_BAND_HZ)
        lf_hf = float(lf / hf) if hf > 1e-9 else 0.0

        # Non-linear.
        sampen = _sample_entropy(ibis_s, m=self.sample_entropy_m)

        # Clip to physiological bounds before passing to the Pydantic model
        # so the validator does not raise on near-misses caused by a single
        # spurious peak.
        hr_bpm = float(np.clip(hr_bpm, 0.0, 300.0))

        return PPGFeatureVector(
            hr_bpm=hr_bpm,
            rmssd_ms=max(0.0, rmssd_ms),
            sdnn_ms=max(0.0, sdnn_ms),
            pnn50_percent=max(0.0, min(100.0, pnn50)),
            lf_power=max(0.0, lf),
            hf_power=max(0.0, hf),
            lf_hf_ratio=max(0.0, lf_hf),
            sample_entropy=max(0.0, sampen),
        )

    # ------------------------------------------------------------------
    # Convenience: load from a CSV file
    # ------------------------------------------------------------------
    @classmethod
    def from_raw_csv(
        cls,
        path: str | Path,
        *,
        sample_rate: float = 25.0,
        column: str | int = 0,
    ) -> PPGFeatureVector:
        """Load a CSV file of PPG samples, extract features, return vector.

        The CSV may have either a single column of amplitudes (no header)
        or multiple columns with a header; in the latter case ``column`` may
        be a string column name or integer index.

        Args:
            path: Filesystem path to the CSV file.
            sample_rate: Sampling rate in Hz (default 25 Hz — the Huawei
                Watch 5 PPG rate).
            column: Column to read, by index or name.

        Returns:
            A :class:`PPGFeatureVector`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``column`` cannot be located or parsing fails.
            InsufficientDataError: If the file contains < 10 s of data.
        """
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"PPG CSV not found: {csv_path!s}")

        values: list[float] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if not rows:
            raise ValueError(f"PPG CSV is empty: {csv_path!s}")

        # Detect header: if the first row has non-numeric cells, treat as header.
        header: list[str] | None
        try:
            [float(cell) for cell in rows[0]]
            header = None
            data_rows = rows
        except ValueError:
            header = rows[0]
            data_rows = rows[1:]

        if isinstance(column, str):
            if header is None:
                raise ValueError(
                    f"Column name {column!r} requested but CSV has no header."
                )
            if column not in header:
                raise ValueError(
                    f"Column {column!r} not found in header {header!r}"
                )
            col_idx = header.index(column)
        else:
            col_idx = int(column)

        for row in data_rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx].strip()
            if not cell:
                continue
            try:
                values.append(float(cell))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric PPG value in {csv_path!s}: {cell!r}"
                ) from exc

        arr = np.asarray(values, dtype=np.float32)
        return cls().extract(arr, sample_rate=float(sample_rate))


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------

__all__ = [
    "InsufficientDataError",
    "PPGFeatureVector",
    "PPGHRVExtractor",
]
