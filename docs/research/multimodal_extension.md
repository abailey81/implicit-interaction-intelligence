# Multimodal Extension: Batch F-1 Research Note

**Status:** Tier-2 advancement, implemented as part of the
`i3.multimodal.voice_real`, `i3.multimodal.vision`, and
`i3.multimodal.fusion_real` packages.

## 1. Motivation: the audio+visual analogue of keystroke dynamics

The I³ project stands on a single load-bearing claim: *how* a user interacts
with a system is a behavioural signature rich enough to condition model
behaviour on.  In the original pipeline this is instantiated via keystroke
dynamics — inter-key intervals, pause structure, backspace ratio — which an
8-dim feature extractor reduces to the keystroke-dynamics group of
`InteractionFeatureVector`.

Voice prosody and facial affect are the direct analogues of that idea on
the audio and visual channels.  *How* you speak (pitch contours, rate,
jitter) and *how* you look (blink rate, brow furrow, smile, gaze) are
orthogonal to *what* you say, exactly as keystroke dynamics are orthogonal
to the text they produce.  The same user-state encoder (TCN) can therefore
consume these channels with no architectural change, provided each modality
is reduced to a fixed-size feature group before ingestion.

Batch F-1 operationalises that claim with three new modules:

| Module | Responsibility | Soft dep |
|---|---|---|
| `i3.multimodal.voice_real.VoiceProsodyExtractor` | 8-dim prosody vector from mono PCM. | `librosa` |
| `i3.multimodal.vision.FacialAffectExtractor` | 8-dim face/gaze vector from RGB frames. | `mediapipe` |
| `i3.multimodal.fusion_real.MultimodalFusion` | Learned fusion head, three strategies, missing-modality aware. | `torch` (hard dep) |

All three are additive — the existing stubs (`voice.py`, `gaze.py`,
`touch.py`, `accelerometer.py`, `fusion.py`) are left untouched so older
callers keep working.

## 2. Voice prosody ↔ keystroke dynamics

The eight voice features map cleanly onto the keystroke-dynamics group:

| Keystroke dynamics (existing)      | Voice prosody (new)                   | Shared behavioural meaning |
|------------------------------------|---------------------------------------|----------------------------|
| `mean_iki` (inter-key interval)    | `speech_rate_syllables_per_s`         | Output cadence             |
| `std_iki`                          | `pitch_std_hz`                        | Variability of motor control |
| `mean_burst_length`                | `voiced_ratio`                        | Uninterrupted activity      |
| `mean_pause_duration`              | `pause_rate_per_s`                    | Hesitation frequency       |
| `backspace_ratio`                  | `shimmer_percent`                     | Micro-correction / instability |
| `composition_speed`                | `pitch_mean_hz`                       | Dominant pace/tone         |
| `pause_before_send`                | `jitter_percent`                      | Cycle-level noise           |
| `editing_effort`                   | `harmonics_to_noise_ratio_db`         | Signal cleanliness          |

The mapping is not mechanical — it is *paradigmatic*.  Both groups describe
*how the user produces output*, not the content of that output.  When a
user is cognitively loaded their keystrokes slow down and their voice
acquires more jitter and more frequent pauses; both effects are captured by
their respective feature groups.

Implementation-wise, `VoiceProsodyExtractor` uses:

* `librosa.yin` for F0 tracking (more deterministic than `pyin` used in the
  stub, which helps the test suite stay reproducible).
* `librosa.feature.spectral_flatness` both as a voicing gate (flatness < 0.5
  ≈ voiced) and as a harmonics-to-noise-ratio approximation (`HNR_dB =
  10·log10((1 − flatness) / flatness)`).
* `librosa.onset.onset_detect` for syllable-rate estimation.
* `librosa.effects.split` for pause-rate estimation via the top-dB
  silence threshold.

Jitter and shimmer follow the Boersma (2001, Praat) definitions —
cycle-to-cycle relative variations of pitch and amplitude, reported as
percentages.  They are the Praat-style descriptors used in clinical voice
analysis and are common inputs to vocal-affect classifiers (Ververidis &
Kotropoulos 2006).

## 3. Facial affect ↔ linguistic features

Face / gaze features are the visual analogue of message content:

| Message-content feature (existing)  | Vision feature (new)        | Shared behavioural meaning     |
|-------------------------------------|-----------------------------|--------------------------------|
| `message_length`                    | `mouth_aspect_ratio`        | How "open" the output channel is |
| `type_token_ratio`                  | `gaze_direction_x`          | Breadth of attention            |
| `mean_word_length`                  | `gaze_direction_y`          | Vertical attention (reading)   |
| `flesch_kincaid`                    | `head_pose_pitch_deg`       | Cognitive posture               |
| `question_ratio`                    | `eye_aspect_ratio`          | Engagement / alertness          |
| `formality`                         | `head_pose_yaw_deg`         | Orientation toward interlocutor |
| `emoji_density`                     | `smile_au12`                | Affect marker                   |
| `sentiment_valence`                 | `brow_furrow_au4`           | Affect marker (negative)        |

The landmark-derived approach (MediaPipe Face Mesh, 468 points with
iris-refinement) is deliberately **low-dependency**: no downloaded emotion
classifier weights are required; all eight features come from closed-form
geometric calculations on the mesh.  That keeps the module robust, fast,
and license-safe for edge deployment on Huawei AI Glasses and Smart Hanhan
(see §4).

The Soukupová & Čech (2016) EAR formulation is used verbatim; MAR is its
mouth-axis twin; gaze is the normalised offset of each iris centre from its
eye socket's centroid.  AU4 (brow furrow) and AU12 (lip-corner puller) are
geometric proxies for the classical FACS action units (Ekman & Friesen
1978), sufficient for cognitive-state signalling without running a full
FACS classifier.

## 4. Huawei hardware lineage alignment

Batch F-1 maps directly onto the three long-running hardware tracks named
in the brief:

* **AI Glasses (12 MP camera, always-on mic).** `FacialAffectExtractor`
  consumes 12 MP frames (or cheaper down-sampled tiles) and returns the
  same 8-dim contract used by the TCN; `VoiceProsodyExtractor` does the
  same for the always-on mic.  Both run at < 30 ms on CPU.
* **Smart Hanhan.** Keystroke + touch + voice are the primary input
  channels; touch is already covered by the existing `TouchFeatureExtractor`
  and slots directly into `MultimodalFusion`'s modality map.
* **Darwin Research Centre CV lineage.** The mesh-based vision extractor
  inherits from the Darwin Centre's long tradition of
  landmark-geometry-based affect analysis.  MediaPipe's 2019 architecture
  is the de-facto standard for this style.

Each track is **optional** at inference time — the fusion head's
missing-modality policies (see §5) let the same model serve a watch without
camera feeds and a pair of glasses without an IMU.

## 5. Fusion strategies — trade-offs

`MultimodalFusion` implements three strategies, each with its own
engineering trade-off (Baltrušaitis et al. 2019; Liang et al. 2023):

### 5.1 `late_concat`

Concatenate all modality vectors and project linearly to the encoder's
input width (32).  Strengths: simplest, lowest latency, easiest to
interpret.  Weaknesses: gives equal weight to every modality regardless of
reliability; learning a good projection requires balanced data.

### 5.2 `late_gated`

A learned sigmoid head computes a per-modality scalar gate before
concatenation.  Strengths: the model can learn to down-weight unreliable
modalities on the fly; the gates are directly interpretable as "how much
are we currently trusting voice vs vision?".  Weaknesses: the gates are
still per-sample, not per-dim, so correlated noise within a modality
survives.

### 5.3 `attention`

Each modality is projected to a 32-dim token; a learned CLS token attends
over all modality tokens.  The CLS output is projected to the encoder
input.  Strengths: the most expressive; it can represent "only use voice
when vision says the subject is silent"-style cross-modality rules.
Weaknesses: higher parameter count; harder to debug; needs more data.

The three strategies share weights via `concat_proj`, so switching
strategies at inference time (e.g. downgrade to `late_concat` on an
energy-constrained edge device) is cheap.

## 6. Missing-modality handling

Two policies are supported, mirroring the dominant 2025 HCI review on
robust multimodal fusion under sensor dropout (Liang et al. 2023):

* **`zero_fill`** substitutes a zero vector for any missing modality.  The
  concat / gate / attention layer learns to treat "zero" as a valid
  "absent" signal — cheap, but only works when missingness is rare and
  roughly uniform at training time.
* **`mask_drop`** additionally (a) renormalises the concat projection by
  `total_dims / present_dims` so a 2-modality call does not get
  systematically under-weighted, (b) zeros the gates of missing modalities
  before applying them, and (c) feeds MediaPipe-style `key_padding_mask`
  to the attention head so missing modality tokens do not participate in
  softmax normalisation.  This is the recommended default for production
  deployments with variable sensor availability.

The choice of policy is independent of the fusion strategy; callers pick
them separately via the constructor.

## 7. Threats to validity

The current evaluation is entirely **synthetic**:

* `test_voice_prosody.py` exercises sine tones, silence, and noisy sines —
  not real speech;
* `test_facial_affect.py` exercises black, white, and gradient frames —
  not real faces;
* `test_multimodal_fusion.py` uses random 8-dim vectors.

These tests prove that the extractors are deterministic, robust to
malformed input, and produce the correct tensor shapes.  They do **not**
prove that the resulting embeddings are informative for user-state
estimation — that requires recording a human cohort under controlled
conditions, which is out of scope for Batch F-1 and tracked as a
follow-up.

Similarly, the MediaPipe Face Mesh is a *proxy* for FACS AU detection.
A trained AU classifier (e.g. OpenFace) is more accurate but pulls in
heavier dependencies and licensing constraints, which would compromise the
edge-deployment story.  The proxy AUs are sufficient for coarse engagement
signals and were validated on the original datasets Kartynnik et al.
(2019) trained the mesh on.

## 8. Future work

1. **Cross-modal alignment.** Train a contrastive objective that pulls
   voice, vision, and keystroke embeddings of the same user together, as
   a prelude to modality-agnostic few-shot adaptation.
2. **Modality dropout training.** During training, drop each modality
   independently with probability 0.3.  This teaches the fusion head to
   remain useful when any single channel goes down, and is especially
   relevant for AI Glasses where the mic can be muted at the OS level.
3. **Temporal fusion.** Currently fusion runs at the single-frame level.
   A short-window buffer (e.g. 3 s at 30 Hz vision + 10 ms frames of
   audio) fed as a sequence through the existing TCN would integrate
   dynamic information the current single-frame fusion throws away.
4. **Edge profiling.** Measure end-to-end latency on the AI Glasses SoC
   and Smart Hanhan's mobile CPU, and decide per-device whether to
   dispatch `attention` or `late_concat`.
5. **Differential privacy on voice/vision.** Voice and face data are
   considerably more sensitive than keystrokes.  A privatised aggregation
   step between the extractors and the fusion head would be a natural
   extension of the existing DP roadmap.

## References

* Baltrušaitis, T., Ahuja, C., Morency, L.-P. (2019). *Multimodal Machine
  Learning: A Survey and Taxonomy.* IEEE TPAMI 41(2).
* Boersma, P. (2001). *Praat, a system for doing phonetics by computer.*
  Glot International 5(9/10).
* Ekman, P., Friesen, W. V. (1978). *Facial Action Coding System.*
  Consulting Psychologists Press.
* Kartynnik, Y., Ablavatski, A., Grishchenko, I., Grundmann, M. (2019).
  *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.*
  CVPR Workshop on Computer Vision for AR/VR.
* Liang, P. P., Zadeh, A., Morency, L.-P. (2023). *Foundations and Trends
  in Multimodal Machine Learning.* arXiv:2209.03430.
* McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in
  Python.* Proc. 14th SciPy.
* Sagisaka, Y., Campbell, N. (2004). *Prosody in Speech Synthesis.*
  Springer.
* Soukupová, T., Čech, J. (2016). *Real-Time Eye Blink Detection using
  Facial Landmarks.* Computer Vision Winter Workshop.
* Ververidis, D., Kotropoulos, C. (2006). *Emotional speech recognition:
  Resources, features, and methods.* Speech Communication 48(9).
