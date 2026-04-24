"""Unified multi-stream wearable signal ingestor (Batch F-2).

This module sits one layer above :class:`PPGHRVExtractor` and is the
canonical adapter between raw vendor-specific wearable exports (Huawei
Health, Fitbit, Apple Health, Garmin, Polar, generic IBI text) and the
I³ TCN encoder's 8-dim-per-modality convention.

Concretely it emits a **16-dim wearable feature vector** per 60-second
window:

* 8 dims from PPG / HRV (see :mod:`i3.multimodal.ppg_hrv`).
* 8 dims from a lightweight accelerometer summary — gravity-corrected
  mean-magnitude, step cadence, jerk, orientation variance, tremor
  amplitude, activity intensity, rest fraction, gait regularity.

Two design decisions are worth highlighting up-front:

1.  The 8-dim accelerometer block is **re-implemented locally** rather
    than importing :mod:`i3.multimodal.accelerometer`, because the
    published feature contract of that module is slightly different
    (it focuses on short raw-IMU windows), and Batch F-2 explicitly
    requires a wearable-aggregation variant.  The local implementation
    is additive — it does not modify any existing file.
2.  Vendor parsers are registered through the :class:`WearableFormat`
    enum, so extending support for a new wearable is a single-file
    change inside this module.

References
----------
* Huawei Health Kit / HarmonyOS DDM documentation for Huawei Watch 5
  (2025).  Three-in-one ECG + PPG + pressure sensor and 30-minute stress
  monitoring via HRV at rest.
* Fitbit (Google): Intraday heart-rate / IBI CSV export specification.
* Apple HealthKit XML export format for ``HKQuantityTypeIdentifierHeartRate``
  and IBI streams.
* Garmin FIT file specification (Heart Rate + IBI messages).
* Polar H10 strap: per-RR-interval Bluetooth characteristic & text export.
* Nelson, B. W., Low, C. A. et al. (2020).  *Guidelines for wrist-worn
  consumer wearable assessment of heart rate in cardiology: just because
  you can, doesn't mean you should.*  npj Digital Medicine 3, 90.
"""

from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from i3.multimodal.ppg_hrv import (
    InsufficientDataError,
    PPGFeatureVector,
    PPGHRVExtractor,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import numpy.typing as npt

    NDArrayF32: TypeAlias = npt.NDArray[np.float32]
else:
    NDArrayF32 = np.ndarray


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WearableFormatError(ValueError):
    """Raised when an unknown / invalid vendor format is requested."""


class WearableParseError(ValueError):
    """Raised when a vendor file is malformed or cannot be parsed."""


# ---------------------------------------------------------------------------
# Format enum + samples
# ---------------------------------------------------------------------------


class WearableFormat(str, Enum):
    """Supported wearable export formats.

    Exactly six entries.  Order is semantically significant only insofar
    as unit tests assert on the count.
    """

    HUAWEI_WATCH = "huawei_watch"
    FITBIT_CSV = "fitbit_csv"
    APPLE_HEALTH_XML = "apple_health_xml"
    GARMIN_FIT = "garmin_fit"
    POLAR_H10 = "polar_h10"
    GENERIC_IBI_TXT = "generic_ibi_txt"


@dataclass(frozen=True)
class WearableSample:
    """A single time-stamped wearable reading.

    Missing channels are represented as ``None``; downstream aggregation
    masks them accordingly.

    Attributes:
        timestamp: Unix epoch seconds.
        ppg_amplitude: Raw PPG sample, or ``None`` for vendors that only
            export beat-to-beat IBIs.
        ibi_ms: Inter-beat interval (milliseconds), when the vendor
            pre-computed beats.
        accelerometer: ``(x, y, z)`` tuple in units of ``g`` (gravity)
            when available, otherwise ``None``.
        skin_temperature_c: Wrist skin temperature (Celsius) if reported.
        activity_stage: Vendor-specific label such as ``"rest"``,
            ``"walking"``, ``"running"``, ``"sleep"``.
    """

    timestamp: float
    ppg_amplitude: float | None = None
    ibi_ms: float | None = None
    accelerometer: tuple[float, float, float] | None = None
    skin_temperature_c: float | None = None
    activity_stage: str | None = None


# ---------------------------------------------------------------------------
# Accelerometer feature block
# ---------------------------------------------------------------------------


_ACCEL_FEATURE_NAMES: tuple[str, ...] = (
    "gravity_corrected_mag",
    "step_cadence_hz",
    "jerk",
    "orientation_variance",
    "tremor_amplitude",
    "activity_intensity",
    "rest_fraction",
    "gait_regularity",
)


def _extract_accel_features(samples: list[WearableSample]) -> NDArrayF32:
    """Return an 8-dim float32 vector summarising wrist accelerometry.

    The feature order matches :data:`_ACCEL_FEATURE_NAMES`.

    Args:
        samples: Sequence of :class:`WearableSample` within the aggregation
            window.

    Returns:
        1-D ``np.ndarray`` of shape ``(8,)``; all zeros when no
        accelerometer channel is present.
    """
    triaxial: list[tuple[float, float, float]] = [
        s.accelerometer for s in samples if s.accelerometer is not None
    ]
    timestamps: list[float] = [
        s.timestamp for s in samples if s.accelerometer is not None
    ]
    if len(triaxial) < 2:
        return np.zeros(8, dtype=np.float32)

    accel = np.asarray(triaxial, dtype=np.float64)  # (N, 3)
    ts = np.asarray(timestamps, dtype=np.float64)

    # Gravity-corrected magnitude: subtract the mean (the ~1 g DC bias on a
    # stationary wrist) and take the Euclidean norm.
    centred = accel - np.mean(accel, axis=0, keepdims=True)
    mag = np.linalg.norm(centred, axis=1)
    gravity_corrected_mag = float(np.mean(mag))

    # Step cadence: approximate dominant frequency in 0.5–3 Hz (walk/run).
    dt_seq = np.diff(ts)
    fs = 1.0 / float(np.median(dt_seq)) if dt_seq.size > 0 and np.median(dt_seq) > 0 else 1.0
    if mag.size >= 8 and fs > 0:
        spectrum = np.abs(np.fft.rfft(mag - np.mean(mag))) ** 2
        freqs = np.fft.rfftfreq(mag.size, d=1.0 / fs)
        band = (freqs >= 0.5) & (freqs <= 3.0)
        if band.any() and spectrum[band].sum() > 0:
            step_cadence_hz = float(freqs[band][int(np.argmax(spectrum[band]))])
        else:
            step_cadence_hz = 0.0
    else:
        step_cadence_hz = 0.0

    # Jerk: RMS of first time-derivative of magnitude.
    if mag.size > 1:
        dmag = np.diff(mag) / np.maximum(dt_seq, 1e-6)
        jerk = float(np.sqrt(np.mean(dmag ** 2)))
    else:
        jerk = 0.0

    # Orientation variance: sum of per-axis variance.
    orientation_variance = float(np.sum(np.var(accel, axis=0)))

    # Tremor amplitude: band-passed 4–12 Hz energy proxy via high-pass diff.
    tremor_amplitude = float(np.std(np.diff(mag))) if mag.size > 1 else 0.0

    # Activity intensity: mean absolute magnitude above a rest threshold.
    activity_intensity = float(np.mean(np.abs(mag)))

    # Rest fraction: fraction of samples below the rest threshold.
    rest_threshold = 0.05  # 0.05 g (≈ wrist at rest)
    rest_fraction = float(np.mean(np.abs(mag) < rest_threshold))

    # Gait regularity: normalised autocorrelation at lag ≈ 1 step.
    if mag.size >= 4:
        mag_centred = mag - np.mean(mag)
        denom = float(np.dot(mag_centred, mag_centred))
        if denom > 1e-9:
            lag = max(1, int(round(fs / max(step_cadence_hz, 0.5))))
            lag = min(lag, mag.size - 1)
            num = float(np.dot(mag_centred[:-lag], mag_centred[lag:]))
            gait_regularity = float(max(0.0, min(1.0, num / denom)))
        else:
            gait_regularity = 0.0
    else:
        gait_regularity = 0.0

    return np.asarray(
        [
            gravity_corrected_mag,
            step_cadence_hz,
            jerk,
            orientation_variance,
            tremor_amplitude,
            activity_intensity,
            rest_fraction,
            gait_regularity,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vendor parsers
# ---------------------------------------------------------------------------


def _parse_huawei_watch(path: Path) -> list[WearableSample]:
    """Parse a JSON payload exported by Huawei Health (DDM mock).

    The expected schema is an array of objects with ``timestamp`` (seconds
    since epoch or ISO-8601 string) and any of ``ppg``, ``ibi_ms``,
    ``accelerometer``, ``skin_temperature_c``, ``activity_stage``.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        A list of :class:`WearableSample`.

    Raises:
        WearableParseError: If the JSON is malformed or a sample lacks a
            timestamp.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise WearableParseError(f"Invalid Huawei Watch JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise WearableParseError(
            "Huawei Watch JSON must be a top-level array of sample objects."
        )

    samples: list[WearableSample] = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            raise WearableParseError(
                f"Huawei Watch JSON entry {i} is not an object."
            )
        ts_raw = obj.get("timestamp")
        if ts_raw is None:
            raise WearableParseError(f"Huawei Watch JSON entry {i}: missing timestamp.")
        ts = _coerce_timestamp(ts_raw)
        accel_raw = obj.get("accelerometer")
        accel_tuple: tuple[float, float, float] | None = None
        if isinstance(accel_raw, (list, tuple)) and len(accel_raw) == 3:
            accel_tuple = (float(accel_raw[0]), float(accel_raw[1]), float(accel_raw[2]))

        samples.append(
            WearableSample(
                timestamp=ts,
                ppg_amplitude=_optional_float(obj.get("ppg")),
                ibi_ms=_optional_float(obj.get("ibi_ms")),
                accelerometer=accel_tuple,
                skin_temperature_c=_optional_float(obj.get("skin_temperature_c")),
                activity_stage=_optional_str(obj.get("activity_stage")),
            )
        )
    return samples


def _parse_fitbit_csv(path: Path) -> list[WearableSample]:
    """Parse a Fitbit intraday CSV export.

    Expected columns (header-driven): ``timestamp``, ``heart_rate`` (bpm).
    Fitbit exposes intraday HR; we convert each HR reading to a synthetic
    IBI = 60000 / HR ms.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        A list of :class:`WearableSample` with ``ibi_ms`` populated.

    Raises:
        WearableParseError: If required columns are missing.
    """
    samples: list[WearableSample] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise WearableParseError("Fitbit CSV missing 'timestamp' column.")
        if "heart_rate" not in reader.fieldnames:
            raise WearableParseError("Fitbit CSV missing 'heart_rate' column.")
        for row in reader:
            try:
                ts = _coerce_timestamp(row["timestamp"])
                hr = float(row["heart_rate"])
            except (KeyError, ValueError) as exc:
                raise WearableParseError(f"Bad Fitbit row: {row}") from exc
            ibi = 60_000.0 / hr if hr > 0 else None
            samples.append(WearableSample(timestamp=ts, ibi_ms=ibi))
    return samples


def _parse_apple_health_xml(path: Path) -> list[WearableSample]:
    """Parse an Apple Health XML export for heart-rate records.

    Reads ``<Record type="HKQuantityTypeIdentifierHeartRate">`` entries
    (value in bpm) and ``<Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN">``
    entries (IBI proxy).

    Args:
        path: Filesystem path to the XML file.

    Returns:
        A list of :class:`WearableSample`.

    Raises:
        WearableParseError: If the XML is malformed.
    """
    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        raise WearableParseError(f"Invalid Apple Health XML: {exc}") from exc

    root = tree.getroot()
    samples: list[WearableSample] = []
    for rec in root.iter("Record"):
        rec_type = rec.get("type", "")
        start_date = rec.get("startDate") or rec.get("creationDate")
        value_raw = rec.get("value")
        if start_date is None or value_raw is None:
            continue
        try:
            ts = _coerce_timestamp(start_date)
            value = float(value_raw)
        except ValueError:
            continue
        if rec_type.endswith("HeartRate"):
            ibi = 60_000.0 / value if value > 0 else None
            samples.append(WearableSample(timestamp=ts, ibi_ms=ibi))
        elif rec_type.endswith("HeartRateVariabilitySDNN"):
            # HKQuantityTypeIdentifierHeartRateVariabilitySDNN is reported in
            # milliseconds; treat it as a direct IBI proxy for aggregation.
            samples.append(WearableSample(timestamp=ts, ibi_ms=value))
    return samples


def _parse_garmin_fit(path: Path) -> list[WearableSample]:
    """Parse a Garmin FIT-exported file.

    Full FIT parsing requires the ``fitparse`` package which is **not** a
    project dependency.  For tests and the reference pipeline we accept a
    JSON-lines sidecar with the same records; the production pipeline would
    soft-import ``fitparse`` and walk its message iterator.

    Args:
        path: Filesystem path.  May be a ``.fit.json`` JSONL file (one
            message per line) or a plain JSON array.

    Returns:
        A list of :class:`WearableSample`.

    Raises:
        WearableParseError: If the payload cannot be interpreted.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError as exc:
        raise WearableParseError(f"Cannot read Garmin FIT sidecar: {exc}") from exc
    if not text:
        raise WearableParseError("Garmin FIT sidecar is empty.")

    rows: list[dict]
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WearableParseError(f"Invalid Garmin FIT JSON: {exc}") from exc
    else:
        rows = []
        for i, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise WearableParseError(
                    f"Invalid Garmin FIT JSONL line {i}: {exc}"
                ) from exc

    samples: list[WearableSample] = []
    for obj in rows:
        if not isinstance(obj, dict):
            continue
        ts_raw = obj.get("timestamp")
        if ts_raw is None:
            continue
        ts = _coerce_timestamp(ts_raw)
        hr = _optional_float(obj.get("heart_rate"))
        ibi_val = _optional_float(obj.get("ibi_ms"))
        if ibi_val is None and hr is not None and hr > 0:
            ibi_val = 60_000.0 / hr
        samples.append(
            WearableSample(
                timestamp=ts,
                ibi_ms=ibi_val,
                activity_stage=_optional_str(obj.get("activity")),
            )
        )
    return samples


def _parse_polar_h10(path: Path) -> list[WearableSample]:
    """Parse a Polar H10 chest-strap RR export.

    The exported text file is usually three columns: ``timestamp_s``,
    ``rr_ms``, and optionally ``hr_bpm``.  Whitespace- or comma-separated.

    Args:
        path: Filesystem path.

    Returns:
        A list of :class:`WearableSample` with ``ibi_ms`` set.

    Raises:
        WearableParseError: If no parseable rows are found.
    """
    samples: list[WearableSample] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p for p in stripped.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])
                rr = float(parts[1])
            except ValueError:
                continue
            samples.append(WearableSample(timestamp=ts, ibi_ms=rr))
    if not samples:
        raise WearableParseError(f"Polar H10 file contained no parseable rows: {path!s}")
    return samples


def _parse_generic_ibi_txt(path: Path) -> list[WearableSample]:
    """Parse a generic newline-delimited IBI text file.

    Each non-blank line is one IBI in milliseconds.  Timestamps are
    synthesised by cumulative sum starting at ``t=0``.  This is the
    lowest-common-denominator format accepted by most HRV research tools
    (Kubios, RHRV).

    Args:
        path: Filesystem path.

    Returns:
        A list of :class:`WearableSample`.

    Raises:
        WearableParseError: If no IBIs are found.
    """
    ibis: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ibis.append(float(line))
            except ValueError:
                continue
    if not ibis:
        raise WearableParseError(f"Generic IBI file empty: {path!s}")
    samples: list[WearableSample] = []
    t = 0.0
    for ibi in ibis:
        t += ibi / 1000.0
        samples.append(WearableSample(timestamp=t, ibi_ms=ibi))
    return samples


_PARSERS: dict[WearableFormat, Callable[[Path], list[WearableSample]]] = {
    WearableFormat.HUAWEI_WATCH: _parse_huawei_watch,
    WearableFormat.FITBIT_CSV: _parse_fitbit_csv,
    WearableFormat.APPLE_HEALTH_XML: _parse_apple_health_xml,
    WearableFormat.GARMIN_FIT: _parse_garmin_fit,
    WearableFormat.POLAR_H10: _parse_polar_h10,
    WearableFormat.GENERIC_IBI_TXT: _parse_generic_ibi_txt,
}


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_timestamp(raw: float | int | str) -> float:
    """Coerce a timestamp field to Unix epoch seconds.

    Args:
        raw: A float / int (epoch seconds) or ISO-8601 string.

    Returns:
        Epoch seconds as a float.

    Raises:
        WearableParseError: If the value cannot be parsed.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        try:
            return float(stripped)
        except ValueError:
            pass
        # Accept ISO 8601 with an optional trailing Z.
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WearableParseError(f"Unparseable timestamp: {raw!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise WearableParseError(f"Unsupported timestamp type: {type(raw).__name__}")


def _optional_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_str(v: object) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


@dataclass
class WearableFeatureVector:
    """16-dim feature vector: 8 HRV dims concatenated with 8 accel dims.

    Attributes:
        hrv: The 8-dim :class:`PPGFeatureVector` component.
        accel: 8-dim float32 numpy array (see :data:`_ACCEL_FEATURE_NAMES`).
    """

    hrv: PPGFeatureVector
    accel: NDArrayF32 = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )

    def to_array(self) -> NDArrayF32:
        """Return the concatenated 16-dim ``float32`` vector.

        Returns:
            1-D ``np.ndarray`` of shape ``(16,)``.
        """
        return np.concatenate([self.hrv.to_array(), self.accel.astype(np.float32)])


class WearableSignalIngestor:
    """Multi-vendor wearable ingestor producing 16-dim feature windows.

    The ingestor is stateless: one instance can be reused across sessions.

    Args:
        ppg_sample_rate: Assumed PPG sample rate for vendors that export
            raw waveforms (Huawei Watch 5 → 25 Hz, some clinical devices
            → 100+ Hz).  Defaults to 25 Hz.
        extractor: Optional custom :class:`PPGHRVExtractor` (useful for
            tests and to inject deterministic parameter overrides).
    """

    def __init__(
        self,
        ppg_sample_rate: float = 25.0,
        extractor: PPGHRVExtractor | None = None,
    ) -> None:
        self.ppg_sample_rate = float(ppg_sample_rate)
        self.extractor = extractor if extractor is not None else PPGHRVExtractor()

    # ------------------------------------------------------------------
    def parse(
        self, path: str | Path, format: WearableFormat
    ) -> list[WearableSample]:
        """Parse a session file in the given vendor format.

        Args:
            path: Filesystem path.
            format: Member of :class:`WearableFormat` identifying the
                vendor layout.

        Returns:
            A list of time-stamped :class:`WearableSample` objects.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            WearableFormatError: If ``format`` is not a valid enum member.
            WearableParseError: If the file content is malformed.
        """
        fs_path = Path(path)
        if not fs_path.is_file():
            raise FileNotFoundError(f"Wearable file not found: {fs_path!s}")
        if not isinstance(format, WearableFormat):
            raise WearableFormatError(
                f"format must be a WearableFormat, got {type(format).__name__}"
            )
        parser = _PARSERS.get(format)
        if parser is None:  # pragma: no cover - defensive
            raise WearableFormatError(f"Unsupported WearableFormat: {format}")
        samples = parser(fs_path)
        logger.info(
            "WearableSignalIngestor: parsed %d samples from %s (%s)",
            len(samples),
            fs_path.name,
            format.value,
        )
        return samples

    # ------------------------------------------------------------------
    def aggregate_to_feature_vector(
        self,
        samples: list[WearableSample],
        window_s: int = 60,
    ) -> NDArrayF32:
        """Aggregate samples into the canonical 16-dim feature vector.

        The aggregation logic:

        * Select samples whose timestamps fall within the last
          ``window_s`` seconds of the session.
        * Reconstruct a PPG / IBI-derived 8-dim :class:`PPGFeatureVector`.
          If the window contains only pre-computed IBIs (Fitbit, Garmin,
          Polar, generic IBI), we build a synthetic PPG signal from
          impulses at the IBI-implied beat times and run the standard
          extractor so the whole pipeline stays uniform.
        * Compute the 8-dim accelerometer block from the same window.

        Args:
            samples: The full session; the function selects the final
                ``window_s``-seconds.
            window_s: Window size in seconds (default 60 s, the I³ canonical
                feature-window).

        Returns:
            A 16-dim ``float32`` numpy array.  Returns zeros if no samples
            are available or HRV cannot be computed (warning is logged).
        """
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if not samples:
            return np.zeros(16, dtype=np.float32)

        last_ts = max(s.timestamp for s in samples)
        cutoff = last_ts - float(window_s)
        window_samples = [s for s in samples if s.timestamp >= cutoff]
        if not window_samples:
            return np.zeros(16, dtype=np.float32)

        hrv = self._extract_hrv(window_samples)
        accel = _extract_accel_features(window_samples)
        return WearableFeatureVector(hrv=hrv, accel=accel).to_array()

    # ------------------------------------------------------------------
    def _extract_hrv(self, window_samples: list[WearableSample]) -> PPGFeatureVector:
        """Derive a :class:`PPGFeatureVector` from mixed PPG / IBI streams.

        Preference order:

        1. If ≥ 10 s of raw PPG amplitudes are available, extract directly.
        2. Otherwise, if IBIs are available, synthesise impulses at
           beat-times (an analytically equivalent signal for peak detection).
        3. If neither is usable, return zeros and warn.

        Args:
            window_samples: Samples falling within the aggregation window.

        Returns:
            An 8-dim :class:`PPGFeatureVector`.
        """
        ppg_values = [
            s.ppg_amplitude for s in window_samples if s.ppg_amplitude is not None
        ]
        if len(ppg_values) >= int(self.ppg_sample_rate * 10):
            arr = np.asarray(ppg_values, dtype=np.float32)
            try:
                return self.extractor.extract(arr, sample_rate=self.ppg_sample_rate)
            except InsufficientDataError:
                logger.debug("HRV from raw PPG failed: insufficient data")

        ibis = [s.ibi_ms for s in window_samples if s.ibi_ms is not None]
        if len(ibis) >= 4:
            return self._hrv_from_ibis(np.asarray(ibis, dtype=np.float64))

        logger.warning(
            "WearableSignalIngestor: window has neither enough PPG nor IBIs; "
            "returning zero HRV vector."
        )
        return PPGFeatureVector.zeros()

    # ------------------------------------------------------------------
    def _hrv_from_ibis(self, ibis_ms: np.ndarray) -> PPGFeatureVector:
        """Compute HRV features directly from an IBI series.

        Rather than re-implementing the time-domain / frequency-domain
        pipeline we synthesise an impulse train whose peaks occur at the
        IBI-implied beat times and delegate to :class:`PPGHRVExtractor`.
        This keeps a single canonical code path for HRV maths.

        Args:
            ibis_ms: 1-D numpy array of IBIs in milliseconds.

        Returns:
            An 8-dim :class:`PPGFeatureVector`.
        """
        ibis_s = np.asarray(ibis_ms, dtype=np.float64) / 1000.0
        total_s = float(np.sum(ibis_s))
        if total_s <= 0:
            return PPGFeatureVector.zeros()
        fs = max(self.ppg_sample_rate, 25.0)
        n_samples = int(np.ceil(total_s * fs)) + 2
        signal = np.zeros(n_samples, dtype=np.float32)
        t = 0.0
        for ibi in ibis_s:
            t += float(ibi)
            idx = int(round(t * fs))
            if 0 <= idx < n_samples:
                # Triangular pulse so the band-pass preserves it.
                for offset, amp in ((-1, 0.5), (0, 1.0), (1, 0.5)):
                    j = idx + offset
                    if 0 <= j < n_samples:
                        signal[j] = float(amp)
        try:
            return self.extractor.extract(signal, sample_rate=fs)
        except InsufficientDataError:
            return PPGFeatureVector.zeros()


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------

__all__ = [
    "WearableFeatureVector",
    "WearableFormat",
    "WearableFormatError",
    "WearableParseError",
    "WearableSample",
    "WearableSignalIngestor",
]
