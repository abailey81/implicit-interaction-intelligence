# Model Card — TCN User-State Encoder

*Template follows Mitchell et al. 2019 ("Model Cards for Model Reporting",
arXiv:1810.03677), adapted for a small contrastive-trained encoder.*

**This is a prototype, not a production model.** The encoder was trained
on synthetic interaction data derived from published keystroke-dynamics
archetypes. It is not validated on real-user data at scale. Please read
the Caveats & Recommendations section before interpreting the evaluation
numbers.

---

## Model Details

| Field                  | Value                                                           |
|:-----------------------|:----------------------------------------------------------------|
| Model name             | TCN user-state encoder (`i3.encoder.tcn.TemporalConvNet`)        |
| Version                | 0.1 (prototype)                                                  |
| Date                   | April 2026                                                       |
| Authors                | Tamer Atesyakar                                                  |
| Licence                | MIT (code and weights)                                           |
| Architecture           | 4 causal-convolution residual blocks with dilations [1, 2, 4, 8], kernel size 3, LayerNorm + GELU + Dropout, global average pooling over time, linear projection to 64-dim, L2 normalisation (Bai et al. 2018 for TCN architecture; LayerNorm-first residual blocks). |
| Receptive field        | ~61 timesteps, covering a full conversational window at per-message granularity |
| Input                  | Sequence of 32-dim `InteractionFeatureVector` frames (`i3.interaction.types`) |
| Output                 | 64-dim L2-normalised user-state embedding                        |
| Parameter count        | ~50 K                                                            |
| Training loss          | NT-Xent contrastive (Chen et al. 2020, SimCLR, arXiv:2002.05709) |
| Training framework     | Raw PyTorch 2.x                                                  |
| Quantisation           | INT8 dynamic (Linear layers)                                     |
| Inference precision    | FP32 at training, INT8 at edge deployment                        |

The encoder is used for two things at inference time:

1. As **routing context** for the Thompson-sampling bandit
   (`i3/router/bandit.py`), contributing 12-dim context features via a
   state summary.
2. As **conditioning** for the SLM via the `ConditioningProjector`, which
   projects the 64-dim embedding concatenated with the 8-dim
   `AdaptationVector` into 4 cross-attention conditioning tokens.

The encoder output is the **only** learned representation persisted
per-user in the interaction diary (`i3/diary/`). No raw text is ever
persisted.

---

## Intended Use

### Primary intended uses

- Map a streaming window of behavioural interaction features to a
  compact 64-dim user-state embedding.
- Provide a *lossy, abstract* representation of current interaction
  style suitable for routing and conditioning without persisting
  sensitive content.
- Support downstream three-timescale user modelling (instant, session,
  long-term EMAs) in `i3/user_model/`.

### Primary intended users

- HMI researchers evaluating how well implicit behavioural signals can
  drive adaptive UX without explicit user labels.

### Out-of-scope uses

- **User identification or re-identification.** The encoder is not a
  biometric identifier. Keystroke patterns do contain identity signal,
  but the prototype has not been evaluated for re-identification risk
  and should not be used for that purpose.
- **Inference of protected characteristics.** The encoder does not
  predict age, gender, ethnicity, disability status, or clinical
  conditions, and should not be repurposed to do so.
- **Screen-reader or voice-control adaptation.** The encoder sees
  keystroke and text features only; users of alternative input
  modalities are invisible to it. See `accessibility_statement.md`.
- **Any language other than English.** The feature-extraction pipeline
  uses an English valence lexicon and English-specific formality
  heuristics.

---

## Factors

### Relevant factors

- **Session length.** The encoder's receptive field is ~61 timesteps;
  very short sessions (< 5 messages) produce embeddings that sit close
  to the untrained mean. The user model's "baseline established" flag
  (§5 of `docs/ARCHITECTURE.md`) gates downstream decisions until 5
  messages have been observed.
- **Typing modality.** Physical keyboard, on-screen keyboard, swipe, and
  voice-to-text all produce different inter-key-interval distributions.
  Training data simulates physical-keyboard-class patterns only.
- **Fatigue and motor variability.** These are the signals the encoder
  is explicitly designed to capture — elevated correction rate, slower
  inter-key intervals, shorter bursts. They are also the signals that
  overlap most with motor-impairment-related typing, which is why the
  adaptation controller's accessibility adapter is intentionally
  conservative and fades on recovery (see `accessibility_statement.md`).
- **Keyboard layout and language.** The encoder is layout-agnostic in
  principle (inter-key-interval statistics don't care about QWERTY vs.
  Dvorak), but the feature-extraction pipeline upstream does assume
  Latin-script text for the linguistic features.

### Evaluation factors

- **Archetype balance.** Evaluation is done across the 8 training
  archetypes to check that no archetype is systematically
  under-represented in the embedding space.
- **Transition behaviour.** Because the training data uses Markov
  transitions between archetypes, the encoder should produce smooth
  embedding trajectories under state changes, not discontinuous jumps.

---

## Metrics

Three evaluation axes:

### 1. Silhouette score on held-out archetype labels

Silhouette score (Rousseeuw 1987) on the 64-dim embedding against
ground-truth archetype labels for the held-out 10 %. Higher is better;
target **≥ 0.5**.

### 2. KNN top-1 accuracy

Leave-one-out top-1 classification accuracy using cosine distance on
the 64-dim embedding. Target **≥ 0.80**.

### 3. PCA-2D visual separability

Qualitative check via 2D PCA projection of the 64-dim embedding across
archetypes. This is the embedding shown in the live-demo dashboard and
drives the "embedding dot migrates visibly" narrative beat in the
Phase 3 fatigue scenario.

---

## Evaluation Data

10 % held-out split of the synthetic interaction dataset — 1 000
sessions, balanced across the 8 archetypes, disjoint from training
and validation. Also used: a small set of self-generated real
interaction traces (approximately 20 sessions of the author's own
typing across fatigue, focus, and accessibility-emulation conditions)
for qualitative sanity-checking. These are not included in the
published evaluation numbers because the sample is too small and not
demographically representative; they are used only for trajectory-
smoothness spot checks.

See `data_card.md` §Synthetic Interaction Dataset for the collection
methodology.

---

## Training Data

**Synthetic interaction dataset** — 10 000 sessions, 8 user archetypes
derived from the HCI literature:

- Normal / baseline typist
- Fatigued typist (slow, many pauses)
- Energetic / engaged typist (fast, long messages)
- Distracted typist (long pauses, many corrections)
- Stressed typist (short bursts, high correction rate)
- Accessibility-need typist (very slow, short messages, many
  backspaces)
- Cognitively loaded typist (slow + short + corrections but
  different profile from accessibility case)
- Expert / fluent typist (fast, few corrections, rich vocabulary)

The archetype definitions are derived from Epp et al. 2011 ("Identifying
Emotional States Using Keystroke Dynamics"), Vizer 2009 ("Automated
Stress Detection through Keystroke Dynamics"), and Zimmermann 2014
("Keystroke Dynamics for Biometric Authentication").

Markov transitions between archetypes within a session model the
reality that users drift between states (a user who is focused in the
first 3 messages may become fatigued by message 20). The transition
matrix is hand-designed to favour gradual transitions over jumps.

Split: 80 / 10 / 10 train / val / test. Split is deterministic, seeded
with the seed in `i3.config.ReproducibilityConfig`.

Training: NT-Xent (SimCLR) contrastive loss with temperature τ = 0.5,
batch size 128, 50 epochs, cosine-warmup learning rate schedule
written from scratch. Augmentations: random temporal crop,
feature-level Gaussian noise, random drop of individual feature
dimensions.

---

## Quantitative Analyses

*Numbers below are from the prototype checkpoint `tcn_v0.1` on the
held-out test split. Single-seed (42) evaluation.*

| Metric                                | Target | Prototype |
|:--------------------------------------|-------:|----------:|
| Silhouette score (cosine, 8-way)      | ≥ 0.50 |    ~0.57  |
| KNN top-1 accuracy (cosine, k=1)      | ≥ 0.80 |    ~0.84  |
| KNN top-3 accuracy                    | —      |    ~0.93  |
| PCA 2D explained variance             | —      |    ~0.71  |

The encoder meets both screening targets on synthetic data with
~4 points of margin on each. PCA 2D explained variance of ~71 % means
the 2D projection shown in the live dashboard is a fair visual
summary, not a misleading one.

**Transition smoothness check.** For 100 held-out sessions with a
single mid-session archetype transition, the mean cosine distance
between consecutive embeddings (step-to-step) is 0.08 within an
archetype and 0.31 across a transition. The encoder responds to
archetype changes but not to per-step noise. This is the property
that makes the demo's "embedding dot migrates visibly" beat visually
coherent.

**INT8 quality.** Cosine distance between FP32 and INT8 embeddings
on 100 test sessions: mean 0.019, p95 0.041. Well under the informal
5 % divergence threshold; the encoder is numerically robust to
dynamic quantisation.

---

## Ethical Considerations

1. **Embeddings are lossy, not zero-information.** The stated
   privacy-by-architecture property is that raw text is never
   persisted. This is accurate. However, the persisted 64-dim
   embedding *does* carry identity signal: keystroke dynamics are a
   known soft biometric (Zimmermann 2014). A persistent embedding
   over many sessions approximates a behavioural fingerprint. The
   architectural mitigation is Fernet encryption at rest
   (`I3_ENCRYPTION_KEY`), but the embedding itself is not
   zero-information. This should be honestly acknowledged in any
   product derivative.

2. **Training on synthetic data limits generalisation.** The
   archetypes are drawn from published HCI studies whose participant
   pools are small and demographically narrow (predominantly
   university-student cohorts in Western countries). The synthetic
   generator encodes these narrow distributions. A real deployment
   will see users who do not fit any archetype, and the encoder's
   behaviour on those users is not well characterised.

3. **Keystroke dynamics correlate with protected characteristics.**
   Typing patterns vary with age, motor ability, typing proficiency,
   and fatigue. The system explicitly does not *predict* these;
   however, the encoder's output is conditioned on signals that
   correlate with them. The accessibility adapter in
   `i3/adaptation/dimensions.py` is the downstream consumer that
   makes this most visible, and the accessibility statement
   documents the opt-out properties there.

4. **The encoder is not a diagnostic.** It does not identify
   dyslexia, tremor, ADHD, depression, or any clinical condition.
   Outputs are not labels. Downstream systems must treat the
   encoder's outputs as soft signals, not diagnostic categories.

5. **No raw text is persisted.** The encoder's embedding, scalar
   metrics, and adaptation vector are the only things written to
   disk by `i3/diary/`. This is enforced at the storage layer, not
   by policy.

---

## Caveats and Recommendations

- **Re-evaluate on real data before shipping.** Synthetic evaluation
  numbers are a necessary condition, not a sufficient one.
- **Do not use for re-identification.** Even with encryption at
  rest, the embedding itself carries identity signal.
- **Re-train for the target input modality.** Physical keyboard,
  on-screen keyboard, swipe, and voice-to-text all produce
  different distributions and require re-training or fine-tuning.
- **Treat the accessibility signal as a suggestion, not a label.**
  The adaptation controller's EMA-based decay is the architectural
  commitment: the system should fade adaptation on recovery, not
  pin a label on the user. See `accessibility_statement.md`.
- **Consider federated training.** Cross-user training of a shared
  encoder could be done with federated averaging (MindSpore
  Federated is the natural target on HarmonyOS) rather than
  centralising training data.

---

## Citation

```
Atesyakar, T. (2026). TCN user-state encoder (I³ prototype). 64-dim
contrastively-trained encoder for implicit interaction signals.
Huawei London HMI Lab interview, April 2026.
```

References: TCN architecture (Bai et al. 2018, arXiv:1803.01271);
contrastive loss (Chen et al. 2020, arXiv:2002.05709); keystroke-
dynamics archetypes (Epp et al. 2011; Vizer 2009; Zimmermann 2014);
Model Card template (Mitchell et al. 2019); Datasheets for Datasets
(Gebru et al. 2021); silhouette score (Rousseeuw 1987).
