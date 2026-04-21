# Multi-modal Perception Port

> *"The TCN is modality-agnostic. Keystroke dynamics are just one of many
> possible 8-dim input groups."* — THE_COMPLETE_BRIEF §11

This package sketches the future-work extension of the I³ perception layer
from keystroke-only to multi-modal input: voice prosody, touchscreen
dynamics, gaze, and wearable accelerometry. It is a **code sketch** — the
signal-processing front-ends are intentionally minimal and rely on
soft-imported heavy libraries where appropriate (`librosa` for audio).

## Modality → feature-group mapping

Each modality produces an **8-dim feature group** matching the shape of the
existing `InteractionFeatureVector.keystroke_dynamics` block. A sequence of
fused 64-dim frames (8 groups × 8 dims) can feed the existing TCN encoder
after a light input-projection change.

| Modality | Extractor class | Feature-group index | 8 features |
|:---|:---|:---:|:---|
| Keystroke | (existing) `FeatureExtractor` | 0 | `mean_iki`, `std_iki`, `mean_burst_length`, `mean_pause_duration`, `backspace_ratio`, `composition_speed`, `pause_before_send`, `editing_effort` |
| Message content | (existing) `FeatureExtractor` | 1 | `message_length`, `type_token_ratio`, `mean_word_length`, `flesch_kincaid`, `question_ratio`, `formality`, `emoji_density`, `sentiment_valence` |
| Session dynamics | (existing) `FeatureExtractor` | 2 | `length_trend`, `latency_trend`, `vocab_trend`, `engagement_velocity`, `topic_coherence`, `session_progress`, `time_deviation`, `response_depth` |
| Deviation metrics | (existing) `DeviationMetrics` | 3 | `iki_deviation`, `length_deviation`, `vocab_deviation`, `formality_deviation`, `speed_deviation`, `engagement_deviation`, `complexity_deviation`, `pattern_deviation` |
| **Voice** | `VoiceFeatureExtractor` | 4 | `pitch_mean_hz`, `pitch_std_hz`, `speaking_rate_sps`, `filled_pause_ratio`, `speech_intensity_db`, `voicing_ratio`, `jitter_local`, `shimmer_local` |
| **Touch** | `TouchFeatureExtractor` | 5 | `pressure_mean`, `pressure_var`, `swipe_velocity`, `tap_duration`, `long_press_ratio`, `multi_touch_entropy`, `edge_proximity_ratio`, `path_curvature` |
| **Gaze** | `GazeFeatureExtractor` | 6 | `fixation_duration`, `saccade_rate`, `gaze_target_dwell`, `scanpath_length`, `fixation_variance`, `off_screen_ratio`, `blink_rate`, `smooth_pursuit_ratio` |
| **Accelerometer** | `AccelerometerFeatureExtractor` | 7 | `orientation_var`, `step_cadence_hz`, `jerk_magnitude`, `activity_intensity`, `stillness_ratio`, `tremor_energy`, `orientation_dominant_axis`, `rotation_rate` |

## Fusion and masking

`ModalityFusion.fuse(...)` concatenates present modalities into a 64-dim
tensor and returns a boolean mask indicating which modalities supplied real
data. Missing modalities contribute zeros.

A learned `ModalityEmbedding` (a `torch.nn.Embedding(num_modalities, d_model)`)
conditions the encoder on the *set* of present modalities:

```python
fused  = fusion.fuse(keystroke=kv, voice=vv)  # touch/gaze absent
conditioning = modality_embedding(fused.modality_mask.unsqueeze(0))
# tcn_input = input_projection(fused.features) + conditioning
```

## Soft-import policy

Heavy libraries are **soft-imported**; feature vectors fall back to zeros with
a `logger.warning` when the dependency is absent. Install the future-work
extras to enable all modalities:

```bash
poetry install --with future-work  # pulls librosa, opacus, flwr
```

## Huawei fit

- **AI Glasses** (launched April 21, 2026) — the natural production target for
  the gaze modality. The 30 g glasses offload heavy inference to the paired
  phone; the on-glasses encoder only needs the gaze 8-dim slice.
- **Smart Hanhan** — motion context from the toy's IMU via the accelerometer
  extractor.
- **HarmonyOS 6 Multimodal SDK** — the fused frame contract maps directly
  onto HMAF's multimodal-signal surface (§11 of
  `docs/huawei/harmony_hmaf_integration.md`).

## References

- Baltrušaitis, T., Ahuja, C., Morency, L.-P. (2019). *Multimodal machine
  learning: a survey and taxonomy.* IEEE TPAMI 41(2).
- Bai, S., Kolter, J. Z., Koltun, V. (2018). *An empirical evaluation of
  generic convolutional and recurrent networks for sequence modelling.*
- Salvucci, D. D., Goldberg, J. H. (2000). *Identifying fixations and
  saccades in eye-tracking protocols.* ETRA.
- Bao, L., Intille, S. S. (2004). *Activity recognition from user-annotated
  acceleration data.* Pervasive Computing.
- McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis.* SciPy.
