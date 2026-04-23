# PPG / HRV as a TCN Modality in Implicit Interaction Intelligence

**Scope:** Photoplethysmography-derived heart-rate variability as the
fifth first-class input modality of the I³ Temporal Convolutional Network
encoder, alongside keystroke dynamics, voice prosody, facial affect /
gaze, and accelerometry.
**Primary device target:** Huawei Watch 5 (2025) three-in-one
ECG + PPG + pressure sensor, 30-minute resting HRV loop.

## 1. Motivation

Implicit Interaction Intelligence infers a user's cognitive-load,
affective, and stress state from behavioural signals that the user is
already emitting in the course of normal interaction. The keystroke-
dynamics group has proven that inter-key timing carries sufficient
signal for non-invasive adaptation; Batch F-1 extended the TCN encoder
to voice prosody, facial affect, and wrist accelerometry. The open gap
— and the gap Batch F-2 closes — is a direct physiological readout of
autonomic balance. The 2024 *PMC 11970940* real-time stress-
from-PPG study demonstrated that commodity wrist PPG, after the usual
band-pass + peak-detection pipeline, supports second-scale stress
classification with accuracies competitive with chest-strap ECG
(Alshareef et al., 2024). A 2025 Wiley review (Al-Shehari et al.,
2025) further confirms HRV as the single most evidence-backed
non-invasive stress biomarker in consumer wearables.

The Huawei Watch 5 ships this capability today: the three-in-one
sensor module measures ECG, PPG, and skin pressure simultaneously and
runs a 30-minute sliding HRV estimation at rest, surfacing a four-level
stress label through Huawei Health and Xiaoyi's health summaries.
Aligning I³ with that native device capability means the HarmonyOS
device constellation — phone + AI Glasses + Smart Hanhan + Watch — now
contributes to the user-state embedding.

## 2. Background: HRV Feature Space

The Task Force of the European Society of Cardiology & the North
American Society of Pacing and Electrophysiology (1996) defined the
still-canonical taxonomy of HRV metrics used in this module:

### Time-domain metrics

* **HR (bpm).** Inverse of the mean inter-beat interval (IBI).
* **RMSSD (ms).** Root mean square of successive IBI differences.
  Tracks short-term (beat-to-beat) variability and is the most commonly
  reported parasympathetic / vagal-tone index (Shaffer & Ginsberg
  2017).
* **SDNN (ms).** Standard deviation of NN intervals. A global-
  variability summary statistic correlated with overall autonomic
  flexibility.
* **pNN50 (%).** Proportion of successive IBIs differing by more than
  50 ms. A second parasympathetic index, more sensitive than RMSSD at
  high variability and less sensitive at low variability.

### Frequency-domain metrics

* **LF power (0.04–0.15 Hz, ms²).** Reflects a mixture of sympathetic
  and baroreflex modulation.
* **HF power (0.15–0.40 Hz, ms²).** Dominated by respiratory sinus
  arrhythmia and therefore by parasympathetic outflow.
* **LF/HF ratio.** Controversial but popular: often interpreted as the
  sympatho-vagal balance (Makivic et al., 2013). We carry it forward
  for compatibility with the downstream stress literature while noting
  the interpretation caveats.

### Non-linear metrics

* **Sample entropy.** Richman & Moorman's (2000) regularity statistic
  over the IBI series, capturing short-term complexity unavailable to
  either the time-domain or the frequency-domain summaries.

These eight features form the 8-dim per-modality vector used by the I³
TCN, matching the keystroke, voice, touch, gaze, and accelerometer
groups.

## 3. Method: PPG → 8-dim HRV Vector

`i3/multimodal/ppg_hrv.py` implements a three-stage pipeline:

1. **Pre-filter.** A 4-th order Butterworth band-pass (0.5–5 Hz,
   covering ~30–300 bpm) is applied via `scipy.signal.filtfilt`.
   The filter cutoffs follow Allen (2007)'s PPG review. scipy is
   soft-imported — the module import never fails; when scipy is
   unavailable the extractor emits a warning and returns zeros, so
   the TCN sees a graceful-degradation signal rather than a crash.
2. **Peak detection.** `scipy.signal.find_peaks` with an adaptive
   amplitude threshold (0.3 × std above the mean) and a physiology-
   derived minimum inter-peak distance (`60 / MAX_HR` seconds) extracts
   systolic peaks. The resulting series is converted to IBIs and
   filtered against the physiological `[_MIN_IBI_S, _MAX_IBI_S]`
   bounds.
3. **Feature extraction.** Time-domain features are straight numpy
   reductions; frequency-domain features rely on Welch's PSD applied to
   the IBI series interpolated to a 4 Hz uniform grid (the standard
   rate for short-term HRV PSD, Task Force 1996); sample entropy uses a
   direct Richman-Moorman implementation with tolerance `0.2 × std`.

The class exposes `extract(ppg: np.ndarray, sample_rate: float)
-> PPGFeatureVector` and a convenience `from_raw_csv(path)`
constructor. Signals shorter than 10 s raise an explicit
`InsufficientDataError` — we refuse to fabricate HRV values from 2-s
windows.

## 4. Integration into the TCN Multimodal Encoder

Batch F-1 established two fusion strategies:

* `late_concat` — concatenate per-modality 8-dim vectors, pass through
  a learned linear projection, feed the resulting 32-dim vector into
  the TCN as a length-1 sequence.
* `late_gated` — identical concatenation but each modality is
  multiplied by a learned sigmoid gate computed from the concat, so
  missing / unreliable modalities can be down-weighted.

The HRV vector plugs in at the same boundary as the voice, gaze, and
accelerometer vectors. Concretely, the call-site at
`i3/multimodal/fusion_real.py::MultimodalFusion.fuse` accepts an
`extra_features` mapping; a HarmonyOS deployment would pass
`{"hrv": hrv_vector}` and extend `modality_dim_map` with `hrv -> 8`.
Because we do not modify any existing module in this batch, the
wiring is additive: the F-1 fusion head already supports additional
modalities declared at construction time.

The higher-level 16-dim wearable feature vector is produced by
`WearableSignalIngestor.aggregate_to_feature_vector`. Eight dims come
from the HRV extractor and eight from a lightweight accelerometer
summary (gravity-corrected magnitude, step cadence, jerk, orientation
variance, tremor amplitude, activity intensity, rest fraction, gait
regularity). The 16-dim vector is the canonical transport payload
from the watch to the phone over HarmonyOS DDM.

## 5. Huawei Alignment

The Huawei Watch 5 is the first mass-market consumer device to combine
continuous PPG with a pressure sensor — the pressure channel lets the
watch decide when the PPG estimate is trustworthy versus when motion
artefacts dominate. In production, the I³ runtime does not see the
raw PPG. It sees one of three surfaces:

* **Huawei Health Kit** `@ohos.health.hrv` native subscription. The
  Watch 5 pre-computes RMSSD, SDNN, and stress labels on-device and
  emits them over the HarmonyOS IPC bus. The I³ phone-side runtime
  subscribes and receives a `PPGFeatureVector`-shaped payload every 30
  minutes at rest (plus an on-demand reading when the user triggers
  one).
* **Distributed Data Manager (DDM).** HarmonyOS mirrors the watch's
  HRV table to the phone KV-store. The I³ process watches the store
  and ingests new rows — the `HuaweiWatchHRVSource` mock in
  `i3/huawei/watch_integration.py` models exactly this shape.
* **Huawei Health Cloud Kit** fallback. For multi-device households
  where the watch and phone don't share a network, the cloud provides
  a paged HRV endpoint.

The decoded 8-dim HRV vector feeds two downstream consumers inside
I³:

1. The TCN encoder, via the Batch F-1 multimodal fusion head.
2. The Xiaoyi personal-assistant's health summary surface; Xiaoyi's
   daily brief can mention HRV trends if the user has opted in, with
   the interpretation line produced by `scripts/demos/hrv.py`.

## 6. Threats to Validity

* **Wrist PPG is noisier than chest-strap ECG.** Lu et al. (2009) show
  agreement in RMSSD to within ~10 % between wrist PPG and ECG at rest,
  but the gap widens under motion. Nelson et al. (2020) argue that
  consumer wearables should not be used to *diagnose* arrhythmia —
  our use is trend-tracking, not diagnostic, which sidesteps the
  strongest objection.
* **Motion artefacts dominate during activity.** The
  `pressure` channel on the Watch 5 partially mitigates this at the
  source. At the I³ layer, the activity-intensity dim inside the 8-dim
  accelerometer block flags when HRV should be weighted down by the
  late-gated fusion head.
* **Population bias.** Almost all validation literature is on adult,
  healthy, mostly-male cohorts. Baseline HRV distributions are
  population-dependent and age-dependent; the TCN encoder sees a
  *trend* rather than an absolute value, but downstream stress-label
  thresholds need calibration.
* **Stress ≠ cognitive load.** HRV correlates with both but is not
  specific. The I³ multimodal story is protective here: the TCN sees
  HRV only in combination with keystrokes, voice, gaze, and
  accelerometry — no single modality drives the user-state embedding.

## 7. Future Work

* **On-device streaming via HarmonyOS native bindings.** Replace the
  JSON-file mock in `HuaweiWatchHRVSource` with a thin wrapper around
  the `@ohos.health.hrv` event channel. The public Python API stays
  unchanged; only the `refresh` internals need to move.
* **Multi-vendor calibration.** Fitbit, Apple, Garmin, and Polar all
  report slightly different normal ranges. A calibration layer inside
  `WearableSignalIngestor` should learn a per-device affine
  transformation on top of the raw 16-dim vector so that downstream
  encoders see a canonical distribution.
* **Cross-device sensor fusion.** The phone already carries an IMU.
  Fusing phone-IMU activity context with the watch HRV gives a
  confidence weight for every HRV reading — "was the user moving when
  this HRV was measured?" — beyond what the on-device pressure channel
  provides.
* **Federated HRV baselines.** Because HRV is individual and
  age-dependent, personalised baselines are critical. A small federated-
  averaging loop running over per-user RMSSD deltas would keep raw
  HRV off the server while giving the TCN a population-relative signal.

## 8. References

* Task Force of the European Society of Cardiology and the North
  American Society of Pacing and Electrophysiology (1996). *Heart rate
  variability: standards of measurement, physiological interpretation,
  and clinical use.* Circulation 93(5), 1043–1065.
* Shaffer, F., & Ginsberg, J. P. (2017). *An overview of heart rate
  variability metrics and norms.* Frontiers in Public Health 5:258.
* Makivic, B., Djordjevic Nikic, M., & Willis, M. S. (2013). *Heart
  rate variability (HRV) as a tool for diagnostic and monitoring
  performance in sport and physical activities.* Journal of Exercise
  Physiology Online 16(3), 103–131.
* Allen, J. (2007). *Photoplethysmography and its application in
  clinical physiological measurement.* Physiological Measurement
  28(3), R1–R39.
* Lu, G., Yang, F., Taylor, J. A., & Stein, J. F. (2009). *A
  comparison of photoplethysmography and ECG recording to analyse
  heart rate variability in healthy subjects.* Journal of Medical
  Engineering & Technology 33(8), 634–641.
* Richman, J. S., & Moorman, J. R. (2000). *Physiological time-series
  analysis using approximate entropy and sample entropy.* American
  Journal of Physiology — Heart and Circulatory Physiology 278(6).
* Nelson, B. W., Low, C. A., Jacobson, N., Areán, P., Torous, J., &
  Allen, N. B. (2020). *Guidelines for wrist-worn consumer wearable
  assessment of heart rate in cardiology: just because you can,
  doesn't mean you should.* npj Digital Medicine 3, 90.
* Alshareef et al. (2024). *Real-time stress detection from wearable
  PPG: a deep-learning pipeline.* PMC 11970940.
* Al-Shehari et al. (2025). *A systematic review of HRV-based stress
  prediction in consumer wearables.* Wiley Online Library (in press).
* Huawei Consumer Business (2025). *Huawei Watch 5 launch press
  release: three-in-one ECG + PPG + pressure sensor, 30-minute HRV-based
  stress monitoring.*
* Huawei (2024). *Health Kit developer documentation — HRV streaming
  API and HarmonyOS DDM synchronization of health records.*
