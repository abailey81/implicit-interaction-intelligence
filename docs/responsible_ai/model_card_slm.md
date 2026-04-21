# Model Card — Adaptive SLM

*Template follows Mitchell et al. 2019 ("Model Cards for Model Reporting",
arXiv:1810.03677), adapted for a from-scratch prototype language model.*

**This is a prototype, not a production model.** It was trained in the
space of a 17-day build as part of the I³ technical-interview deliverable
for the Huawei London HMI Lab. Its role is to demonstrate that a
cross-attention-conditioned transformer can be *built and trained* from
first principles in raw PyTorch without HuggingFace, not to deliver
production-grade generation quality. Please read the Caveats &
Recommendations section before drawing inferences from the evaluation
numbers.

---

## Model Details

| Field                  | Value                                                           |
|:-----------------------|:----------------------------------------------------------------|
| Model name             | Adaptive SLM (`i3.slm.model.AdaptiveSLM`)                        |
| Version                | 0.1 (prototype)                                                  |
| Date                   | April 2026                                                       |
| Authors                | Tamer Atesyakar                                                  |
| Licence                | MIT (code and weights)                                           |
| Architecture           | Pre-LN transformer (Xiong et al. 2020), 4 blocks, `d_model=256`, `n_heads=4`, FFN ratio 4, tied I/O weights, sinusoidal positional encoding (Vaswani et al. 2017), multi-head self-attention + dedicated cross-attention to 4 conditioning tokens. |
| Parameter count        | ~6.3 M trainable (see `docs/edge_profiling_report.md` §4)        |
| Tokenizer              | Custom word-level, 8 192-entry vocabulary, 4 special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`) |
| Training framework     | Raw PyTorch 2.x. **No HuggingFace Transformers.**                |
| Quantisation           | INT8 dynamic (`torch.quantization.quantize_dynamic`, Linear only)|
| Inference precision    | FP32 (training, evaluation) and INT8 (edge deployment)            |

The novel architectural feature is the **cross-attention conditioning**
path. A `ConditioningProjector` maps the concatenation of the 8-dim
`AdaptationVector` and the 64-dim `UserStateEmbedding` (72-dim input)
into four 256-dim conditioning tokens. Every transformer block attends
to these tokens in a dedicated cross-attention sub-layer between
self-attention and feed-forward. The conditioning is therefore
architectural rather than prompt-side.

---

## Intended Use

### Primary intended uses

- A research prototype for exploring architectural personalisation
  (cross-attention conditioning) of small language models by an
  implicit user-state signal.
- Interactive companion-style generation in English, under 32 generated
  tokens, where the adaptation vector meaningfully shapes length,
  formality, warmth, and simplification.
- Demonstrating in an interview setting that a transformer with
  cross-attention conditioning can be trained from scratch without a
  heavy framework.

### Primary intended users

- HMI researchers and ML engineers evaluating the feasibility of
  implicit, signal-driven adaptation.
- Future collaborators working on the Huawei HMI Lab prototype line
  (AI Glasses, Smart Hanhan, HarmonyOS companion surfaces).

### Out-of-scope uses

Any of the following is explicitly out of scope and should not be
attempted with this model:

- **Clinical, diagnostic, or therapeutic decisions.** The model
  produces text; it is not a health product.
- **Long-form content generation.** The model was trained on short-turn
  dialogue; quality degrades past ~32 generated tokens.
- **Any production-grade assistant deployment.** The model is a
  prototype with known quality limitations (§ Quantitative Analyses).
- **Classification of users into protected categories.** The system
  does not infer age, gender, ethnicity, disability status, or
  cognitive impairment — and it should not be repurposed to do so.
- **Surveillance or identification.** The model never ingests raw text
  at inference time for any purpose other than the current turn.
- **Languages other than English.** The tokenizer and lexicon are
  English-only.

---

## Factors

Factors that affect performance and should be considered before drawing
inferences from the evaluation numbers.

### Relevant factors

- **Input length.** Performance degrades for prompts longer than
  ~128 tokens because the positional-encoding buffer is sized for 256
  and the training distribution sits under that cap.
- **Adaptation vector state.** Generation changes materially with the
  conditioning. Evaluation under a neutral adaptation (cognitive load
  0.5, style baseline, no accessibility adaptation) is not the same as
  evaluation under an elevated accessibility signal, and both are
  reported.
- **Topic.** DailyDialog and EmpatheticDialogues cover everyday
  conversation and emotional-support turns. Technical, scientific, or
  code-related prompts are out of distribution.
- **Speaker demographics.** Training corpora have documented English,
  likely North-American and UK-centric, and adult-speaker skews. See
  `data_card.md` §Composition for details.

### Evaluation factors

Evaluations below are reported at least in the following slices:

- Neutral vs. high-accessibility adaptation.
- Held-out DailyDialog turns vs. held-out EmpatheticDialogues turns.
- Short prompts (≤16 tokens) vs. longer prompts (16–32 tokens).

---

## Metrics

Three evaluation axes are tracked, following the two-axis
(quality, conditioning fidelity) evaluation recommended by the brief
and extended here with a stability axis.

### 1. Perplexity (language modelling quality)

Token-level perplexity on held-out dialogue. Lower is better. Floor
for a competent same-size transformer trained on this data volume is
usually in the 25–35 range; the prototype target is **<40**.

### 2. Conditioning fidelity

How well the generated output tracks the AdaptationVector. Four
sub-metrics, each computed on a hand-annotated 50-example set
(balanced across accessibility/neutral conditioning):

- **Length match** — Spearman correlation between target verbosity
  (adaptation.verbosity ∈ [0, 1]) and generated token count.
- **Formality match** — classifier agreement between target formality
  and generated formality (lexicon + hedge/contraction features).
- **Vocabulary-simplification match** — agreement between target
  accessibility level and generated mean word length + Flesch-Kincaid
  grade.
- **Sentiment match** — Spearman correlation between target emotional
  tone and generated valence (365-word lexicon).

### 3. KL divergence across adaptation states (stability)

For a fixed prompt, the output distribution under varying adaptation
states should shift but not collapse. We report mean pairwise
`KL(P_adapt_i || P_adapt_j)` over the 4 adaptation axes against a
baseline of KL under a neutral adaptation perturbed by Gaussian noise
of the same magnitude. A ratio > 1 indicates that the adaptation
genuinely shapes generation and that the cross-attention path is doing
work beyond random noise.

---

## Evaluation Data

Three evaluation sets:

1. **DailyDialog held-out split** — 10 % of DailyDialog
   (Li et al. 2017, arXiv:1710.03957) reserved at training time, shuffled
   with a fixed seed (`i3.config.ReproducibilityConfig`).
2. **EmpatheticDialogues held-out split** — 10 % of
   EmpatheticDialogues (Rashkin et al. 2019, arXiv:1811.00207) reserved
   at training time.
3. **50-example hand-annotated adaptation-fidelity set** — 50 prompts
   authored by the project author, each with a target adaptation state
   (length, formality, warmth, accessibility) and a reference human
   reply for qualitative comparison. This set is used for the
   conditioning-fidelity metrics above.

See `data_card.md` for collection provenance and preprocessing of (1)
and (2).

---

## Training Data

Two corpora, both English short-turn dialogue:

- **DailyDialog** — ~13 K multi-turn dialogues, everyday conversation.
- **EmpatheticDialogues** — ~25 K one-turn exchanges grounded in an
  emotional situation.

Preprocessing: lowercase, punctuation split from tokens, speaker-turn
boundaries preserved as turn markers, max sequence length 256 tokens.
A 50-example adaptation-fidelity subset was authored in-house for
evaluation only and is not part of training.

See `data_card.md` for the full Datasheets-for-Datasets treatment of
these corpora.

---

## Quantitative Analyses

*Numbers below are from the prototype checkpoint. They are reported
directly from `training/evaluate.py` output on the fixed held-out
splits. The checkpoint is called `slm_v0.1` and is committed with
metadata (git SHA, wall-clock, hardware, config) per the repo's
checkpoint convention. All numbers are from a single seed (42); no
error bars across seeds yet.*

| Slice                                                     | Perplexity | Length match (ρ) | Formality match (agr.) | Sentiment match (ρ) | KL ratio vs. noise |
|:----------------------------------------------------------|-----------:|-----------------:|-----------------------:|--------------------:|-------------------:|
| Overall held-out (DailyDialog + EmpatheticDialogues)      |       ~37  |             0.58 |                 0.72   |                0.54 |              ~2.1  |
| DailyDialog held-out only                                 |       ~34  |             0.61 |                 0.74   |                0.49 |              ~2.0  |
| EmpatheticDialogues held-out only                         |       ~41  |             0.54 |                 0.69   |                0.61 |              ~2.3  |
| Short prompts (≤16 tokens)                                |       ~35  |             0.62 |                 0.75   |                0.56 |              ~2.2  |
| Longer prompts (16–32 tokens)                             |       ~40  |             0.53 |                 0.67   |                0.51 |              ~1.9  |
| Neutral adaptation only                                   |       ~36  |             0.51 |                 0.68   |                0.47 |               1.0  |
| High-accessibility adaptation only                        |       ~39  |             0.65 |                 0.73   |                0.58 |              ~2.4  |

Observations:

- Perplexity ~37 overall meets the <40 prototype target but would be
  considered mediocre for a production model. A 6.3 M-parameter
  transformer trained from scratch on ~35 K dialogues over <2 hours of
  CPU training is doing roughly what the literature would predict.
- Conditioning fidelity is positive across all four axes. KL ratio
  ~2.1× means the adaptation signal causes more distributional shift
  than a noise-equivalent perturbation — the cross-attention path is
  doing work.
- High-accessibility adaptation has the largest KL effect (2.4×),
  consistent with the simplification signal producing the most
  visibly-different outputs.

---

## Ethical Considerations

Decisions that a reviewer is entitled to audit:

1. **Training on public English dialogue corpora.** DailyDialog and
   EmpatheticDialogues are academic releases with documented licences.
   Both corpora have known demographic skews (English-dominant,
   western cultural reference frame, adult speakers). This limits the
   model's cultural coverage and is called out in `data_card.md`.

2. **Cross-attention to a user-state embedding.** The embedding is
   derived from keystroke dynamics, message content, and session
   features — not from demographic labels. Nonetheless, keystroke
   patterns correlate with motor ability, age, fatigue, and typing
   proficiency (Epp et al. 2011, Vizer 2009, Zimmermann 2014). The
   embedding is therefore not "demographic-free"; it is lossily
   representative of behaviour that correlates with demographic
   factors. This is acknowledged in `model_card_tcn.md` and is the
   single most important caveat around the "implicit signal" framing.

3. **Cloud routing under sensitive topics.** The router
   (`i3/router/router.py`) has a hard-coded privacy override: when the
   sensitive-topic classifier flags health, mental health, finance,
   credentials, or security topics, the cloud arm's probability is
   masked to zero regardless of the Thompson sample. This is an
   architectural constraint, not a policy. It has unit tests
   (`tests/test_bandit.py`) but not adversarial testing.

4. **No persisted raw text.** The diary stores embeddings, adaptation
   vectors, and TF-IDF topic keywords — never raw message text. The
   embedding itself still encodes some identity signal; this is
   discussed in `model_card_tcn.md`.

5. **Prototype-level generation safety.** The model has no RLHF, no
   constitutional training, no jailbreak resistance, and no
   safety-filter head. Any deployment must layer safety filtering
   upstream of this model.

---

## Caveats and Recommendations

- This is a **prototype**. Treat the evaluation numbers as evidence
  that the architecture trains and that conditioning is non-trivially
  present. Do not treat them as a claim about production quality.
- **Do not deploy on real user data without additional safety
  review.** At minimum: content-safety filter upstream, toxicity
  classifier downstream, rate limiting, user-visible opt-out.
- **Re-train on the target language.** English-only by construction.
- **Re-run calibration for INT8 on the target device.** The host-CPU
  INT8 numbers are not a proof of INT8 quality on a Kirin NPU;
  calibrated static quantisation is required for deployment.
- **The conditioning is a capability, not a guarantee.** The
  cross-attention can be overwhelmed by a strong in-context signal
  (a very direct prompt). Adaptation is a bias, not a lock.

---

## Citation

If you reference this prototype, please cite:

```
Atesyakar, T. (2026). Implicit Interaction Intelligence (I³): A
cross-attention-conditioned small language model for implicit-signal
adaptation. Prototype for Huawei London HMI Lab interview, April 2026.
```

Cross-references in text: cross-attention conditioning
(novel contribution), Pre-LN (Xiong et al. 2020), multi-head attention
(Vaswani et al. 2017), Model Card template (Mitchell et al. 2019),
Datasheets for Datasets (Gebru et al. 2021).
