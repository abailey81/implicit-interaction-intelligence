"""Tests for :mod:`i3.multimodal.wearable_ingest` (Batch F-2).

Each vendor parser is exercised via a small hand-written fixture placed
in a ``tmp_path`` directory.  The suite uses no real wearable exports —
every format's minimum-viable example is synthesised inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from i3.multimodal.wearable_ingest import (
    WearableFeatureVector,
    WearableFormat,
    WearableFormatError,
    WearableParseError,
    WearableSample,
    WearableSignalIngestor,
)


# ---------------------------------------------------------------------------
# Enum surface
# ---------------------------------------------------------------------------


def test_wearable_format_has_six_entries() -> None:
    """Format enum exposes exactly the six vendors documented in F-2."""
    assert len(list(WearableFormat)) == 6
    expected = {
        WearableFormat.HUAWEI_WATCH,
        WearableFormat.FITBIT_CSV,
        WearableFormat.APPLE_HEALTH_XML,
        WearableFormat.GARMIN_FIT,
        WearableFormat.POLAR_H10,
        WearableFormat.GENERIC_IBI_TXT,
    }
    assert set(WearableFormat) == expected


# ---------------------------------------------------------------------------
# Per-vendor round-trips
# ---------------------------------------------------------------------------


def _make_huawei_payload(tmp_path: Path) -> Path:
    """Write a toy Huawei Watch DDM JSON and return the path."""
    records = []
    base_ts = 1_700_000_000.0
    for i in range(120):
        records.append(
            {
                "timestamp": base_ts + i * 0.5,
                "ppg": float(np.sin(i * 0.3)),
                "accelerometer": [0.0, 0.0, 1.0 + 0.01 * i],
                "activity_stage": "rest",
                "skin_temperature_c": 32.5,
            }
        )
    path = tmp_path / "huawei.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_parse_huawei_watch_roundtrip(tmp_path: Path) -> None:
    """Huawei Watch JSON parses to the expected number of samples."""
    path = _make_huawei_payload(tmp_path)
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.HUAWEI_WATCH)
    assert len(samples) == 120
    assert isinstance(samples[0], WearableSample)
    assert samples[0].accelerometer == pytest.approx((0.0, 0.0, 1.0))


def test_parse_fitbit_csv_roundtrip(tmp_path: Path) -> None:
    """Fitbit CSV produces IBIs from intraday HR."""
    path = tmp_path / "fitbit.csv"
    rows = ["timestamp,heart_rate"]
    for i in range(20):
        rows.append(f"{1_700_000_000 + i},{72 + (i % 3)}")
    path.write_text("\n".join(rows), encoding="utf-8")
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.FITBIT_CSV)
    assert len(samples) == 20
    assert all(s.ibi_ms is not None and s.ibi_ms > 0 for s in samples)


def test_parse_apple_health_xml_roundtrip(tmp_path: Path) -> None:
    """Apple Health XML reads HeartRate records."""
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<HealthData>
  <Record type=\"HKQuantityTypeIdentifierHeartRate\" startDate=\"2025-04-22T12:00:00Z\" value=\"72\"/>
  <Record type=\"HKQuantityTypeIdentifierHeartRate\" startDate=\"2025-04-22T12:00:10Z\" value=\"75\"/>
  <Record type=\"HKQuantityTypeIdentifierHeartRateVariabilitySDNN\" startDate=\"2025-04-22T12:00:20Z\" value=\"45\"/>
</HealthData>
"""
    path = tmp_path / "apple.xml"
    path.write_text(xml, encoding="utf-8")
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.APPLE_HEALTH_XML)
    assert len(samples) == 3


def test_parse_garmin_fit_jsonl_roundtrip(tmp_path: Path) -> None:
    """Garmin JSONL sidecar is parsed and heart_rate converted to IBI."""
    path = tmp_path / "garmin.fit.json"
    lines = []
    for i in range(10):
        lines.append(
            json.dumps(
                {
                    "timestamp": 1_700_000_000 + i,
                    "heart_rate": 70 + i,
                    "activity": "rest",
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.GARMIN_FIT)
    assert len(samples) == 10
    assert samples[0].activity_stage == "rest"
    assert samples[0].ibi_ms is not None


def test_parse_polar_h10_roundtrip(tmp_path: Path) -> None:
    """Polar H10 text export produces IBI samples."""
    path = tmp_path / "polar.txt"
    lines = ["# timestamp_s rr_ms hr_bpm"]
    t = 0.0
    for i in range(60):
        rr = 850 + (i % 10) * 5
        t += rr / 1000.0
        lines.append(f"{t:.3f} {rr} {60000/rr:.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.POLAR_H10)
    assert len(samples) == 60
    assert samples[0].ibi_ms == pytest.approx(850.0, abs=1.0)


def test_parse_generic_ibi_txt_roundtrip(tmp_path: Path) -> None:
    """Generic IBI text file reconstructs cumulative timestamps."""
    path = tmp_path / "ibi.txt"
    path.write_text("\n".join(str(800 + i) for i in range(30)), encoding="utf-8")
    ingestor = WearableSignalIngestor()
    samples = ingestor.parse(path, WearableFormat.GENERIC_IBI_TXT)
    assert len(samples) == 30
    assert samples[-1].timestamp > samples[0].timestamp


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    """Nonexistent files raise :class:`FileNotFoundError`."""
    ingestor = WearableSignalIngestor()
    with pytest.raises(FileNotFoundError):
        ingestor.parse(tmp_path / "nope.json", WearableFormat.HUAWEI_WATCH)


def test_parse_invalid_format_raises(tmp_path: Path) -> None:
    """Passing a non-enum to ``format`` raises :class:`WearableFormatError`."""
    path = tmp_path / "f.json"
    path.write_text("[]", encoding="utf-8")
    ingestor = WearableSignalIngestor()
    with pytest.raises(WearableFormatError):
        ingestor.parse(path, "huawei_watch")  # type: ignore[arg-type]


def test_parse_malformed_huawei_json_raises(tmp_path: Path) -> None:
    """A Huawei Watch entry missing required fields raises parse error."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    ingestor = WearableSignalIngestor()
    with pytest.raises(WearableParseError):
        ingestor.parse(path, WearableFormat.HUAWEI_WATCH)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_produces_16_dim_vector(tmp_path: Path) -> None:
    """Aggregating a Huawei session yields a 16-dim feature vector."""
    path = _make_huawei_payload(tmp_path)
    ingestor = WearableSignalIngestor(ppg_sample_rate=2.0)
    samples = ingestor.parse(path, WearableFormat.HUAWEI_WATCH)
    vec = ingestor.aggregate_to_feature_vector(samples, window_s=60)
    assert vec.shape == (16,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_aggregate_empty_samples_returns_zeros() -> None:
    """No samples in -> 16 zeros out."""
    ingestor = WearableSignalIngestor()
    out = ingestor.aggregate_to_feature_vector([], window_s=60)
    assert out.shape == (16,)
    assert np.all(out == 0.0)


def test_aggregate_rejects_non_positive_window() -> None:
    """Window size must be positive."""
    ingestor = WearableSignalIngestor()
    with pytest.raises(ValueError):
        ingestor.aggregate_to_feature_vector(
            [WearableSample(timestamp=0.0)], window_s=0
        )


def test_feature_vector_concatenation() -> None:
    """:class:`WearableFeatureVector.to_array` yields an 8+8 = 16 vector."""
    from i3.multimodal.ppg_hrv import PPGFeatureVector

    wv = WearableFeatureVector(
        hrv=PPGFeatureVector(hr_bpm=70.0),
        accel=np.arange(8, dtype=np.float32),
    )
    arr = wv.to_array()
    assert arr.shape == (16,)
    assert arr[0] == pytest.approx(70.0)
    assert arr[8:].tolist() == list(range(8))
