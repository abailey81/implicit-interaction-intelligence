# PPG / HRV Modality — Quickstart (Batch F-2)

This module adds photoplethysmography-derived heart-rate variability
(HRV) as the fifth TCN input modality alongside keystrokes, voice,
vision, and accelerometry. Target device: **Huawei Watch 5** (three-
in-one ECG + PPG + pressure sensor, 30-minute HRV loop).

## TL;DR

* `i3.multimodal.ppg_hrv.PPGHRVExtractor` turns a raw PPG waveform
  into an 8-dim `PPGFeatureVector` (HR, RMSSD, SDNN, pNN50, LF, HF,
  LF/HF, sample entropy).
* `i3.multimodal.wearable_ingest.WearableSignalIngestor` ingests six
  vendor formats (Huawei Watch JSON, Fitbit CSV, Apple Health XML,
  Garmin FIT sidecar, Polar H10 txt, generic IBI txt) and emits a
  16-dim wearable vector = 8 HRV + 8 accelerometer per 60-s window.
* `i3.huawei.watch_integration.HuaweiWatchHRVSource` models the
  HarmonyOS DDM-synced payload from the watch with
  `subscribe / unsubscribe / latest_hrv` surface matching what the
  production native binding would expose.

## Run the demo

```bash
python -m scripts.run_hrv_demo
python -m scripts.run_hrv_demo --stress-scenario
python -m scripts.run_hrv_demo --duration 120 --sample-rate 50
```

The demo synthesises a 60-s, 25-Hz wrist PPG, extracts the 8-dim
feature vector, and prints a Task-Force-1996-referenced interpretation
line. The `--stress-scenario` flag injects an elevated-HR / reduced-
HRV response and prints the delta.

## Minimal programmatic usage

```python
import numpy as np
from i3.multimodal.ppg_hrv import PPGHRVExtractor

ppg = np.load("ppg_25hz.npy")           # 1-D numpy array, 25 Hz
vec = PPGHRVExtractor().extract(ppg, sample_rate=25.0)
print(vec.hr_bpm, vec.rmssd_ms, vec.lf_hf_ratio)
print(vec.to_array().shape)              # (8,)
```

## Vendor ingest

```python
from i3.multimodal.wearable_ingest import (
    WearableFormat, WearableSignalIngestor,
)

ingestor = WearableSignalIngestor()
samples = ingestor.parse("huawei.json", WearableFormat.HUAWEI_WATCH)
vec16 = ingestor.aggregate_to_feature_vector(samples, window_s=60)
```

## Huawei Watch DDM integration (reference)

```python
from i3.huawei.watch_integration import HuaweiWatchHRVSource

src = HuaweiWatchHRVSource(ddm_payload_path="ddm_hrv.json")
src.subscribe(lambda fv: print("new HRV:", fv.rmssd_ms, "ms"))
src.refresh()
print(src.latest_hrv())
```

In a HarmonyOS build, swap the JSON-file reader for the native
`@ohos.health.hrv` subscription. The public API does not change.

## Soft dependencies

* `scipy.signal` — band-pass + peak detection + Welch PSD. Already a
  transitive dependency via scikit-learn. Absent → extractor returns
  zeros and logs a warning; the module still imports.
* `numpy` and `pydantic` (v2) — required.

## Tests

```bash
pytest tests/test_ppg_hrv.py tests/test_wearable_ingest.py -q
```

All tests run on synthetic signals and toy fixtures — no recorded
wearable data required.

## References (short list)

* Task Force 1996 HRV Standards (Circulation 93(5))
* Shaffer & Ginsberg 2017 HRV Overview (Front. Public Health 5:258)
* Makivic et al. 2013 HRV in Sports Medicine
* Allen 2007 PPG Review (Physiological Measurement 28(3))
* Lu et al. 2009 PPG vs ECG HRV Validation
* Richman & Moorman 2000 Sample Entropy
* Alshareef 2024 Real-time Stress from PPG (PMC 11970940)
* Al-Shehari 2025 HRV Stress Prediction Review (Wiley)

See `docs/research/ppg_hrv_integration.md` for the full research note.
