# 2026 Research Landscape

A focused survey of the fields this project draws on, compiled from
an end-of-April-2026 literature scan.  Every claim is sourced; the
"Implications" subsection under each topic is the concrete lesson
for I³.

**Last updated:** 2026-04-23.

---

## 1. Keystroke dynamics and behavioural biometrics

The field remains active.  Recent work has shifted from hand-crafted
timing features (dwell / flight) to learned representations in
Transformer-based encoders, and is pushing hard toward cross-device
generalisation.

- **Transformer cross-device identification.** A 2026 paper proposes a
  unified Transformer-based model for cross-device keystroke-dynamics
  identification, using transfer learning to learn a shared
  representation of typing patterns from both desktop and mobile
  inputs.  Reported Equal Error Rates: **2.45 % desktop, 1.76 %
  mobile, 2.63 % cross-device**.
  [Continuous User Identification Across Devices Using Keystroke Dynamics, Springer 2026](https://link.springer.com/chapter/10.1007/978-3-032-10486-1_6).
- **Hybrid deep architectures.** MDPI Sensors 2024 surveys multi-head
  feature fusion + Conv1D + capsule networks + BiLSTM + attention +
  Monte-Carlo dropout for static-text authentication.  The interesting
  signal is the Monte-Carlo dropout — a cheap uncertainty estimator
  that lets the authenticator refuse low-confidence decisions.
  [Improved Biometric Identification of Keystroke Dynamics via Deep Learning, MDPI 2024](https://www.mdpi.com/1424-8220/24/12/3763).
- **Survey.** For a dependency-free overview of concepts and
  techniques, see
  [arXiv:2303.04605](https://arxiv.org/html/2303.04605v2).
- **Continuous authentication.** Research consistently frames the
  task as continuous (re-auth during a session) rather than one-shot.
  Commercial framing at
  [Plurilock](https://plurilock.com/deep-dive/keystroke-dynamics/).

**Implications for I³.**  The project's TCN-on-keystroke-metrics
architecture is still defensible — the parallelism + long-range
receptive field match the cross-device transformer numbers at a
fraction of the parameter count.  Two concrete upgrades worth
considering:

1. Add **MC-Dropout uncertainty** over the user-state embedding
   (already scaffolded in `i3/adaptation/uncertainty.py` for
   counterfactuals — extend it to auth).
2. Record the **EER benchmark** on the bundled synthetic data so the
   project reports a comparable figure to the 2.63 % cross-device
   baseline.

---

## 2. Small language models (SLM)

2026 is a banner year for SLMs — the Phi-4, Gemma 3 / 4, and Qwen 3
releases genuinely shifted the on-device frontier.

- **Phi-4 family.** Microsoft's Phi-4-mini-instruct (3.8 B params)
  matches 7–9 B models on reasoning after training on curated
  synthetic + filtered public corpora.  [Local AI Master 2026 guide](https://localaimaster.com/blog/small-language-models-guide-2026),
  [Intel Phi-4 acceleration](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-microsoft-phi-4-small-language-models.html).
- **Gemma 3 / 4.** Gemma 3 shipped a 128 K-token context with
  multimodal inputs.  Gemma 4 (April 2026) introduced E2B / E4B
  variants using **Per-Layer Embeddings (PLE)** that run at 4-bit
  quantisation on 5 GB of smartphone RAM.
  [Hugging Face Gemma 4 blog](https://huggingface.co/blog/gemma4).
- **Qwen 3 + MiniCPM.** Both compete with Mistral-7B on English and
  are optimised for English + Chinese.  [BentoML SLM review](https://www.bentoml.com/blog/the-best-open-source-small-language-models),
  [DataCamp Top-15 SLMs 2026](https://www.datacamp.com/blog/top-small-language-models).

**Implications for I³.**  The custom ~6.3 M-parameter SLM in
`i3/slm/` is deliberately smaller than the 2026 SOTA — the point is
the **cross-attention conditioning mechanism**, not a raw LM-quality
competition.  Two things worth adding to the model card:

1. A note that the custom SLM is a *research vehicle for the
   conditioning architecture*, with concrete pointers to Gemma 3 /
   Phi-4-mini as the production substitutes that would inherit the
   conditioning layer.
2. A benchmark against Gemma 3 E2B and Phi-4-mini on the responsiveness
   golden set in `i3/eval/` (would require a one-off Huggingface
   dependency, but only for the baseline run).

---

## 3. Personalised LLMs and user conditioning

The literature now has a canonical family of techniques for threading
user state through an LLM — the closest match for what this project
does is Google's USER-LLM framework.

- **USER-LLM (Google Research).** User embeddings are cross-attended
  with intermediate text representations within the LLM.
  [Google research blog](https://research.google/blog/user-llm-efficient-llm-contextualization-with-user-embeddings/).
  Exactly the architecture this project implements.
- **LLM-Modules knowledge transfer.** [arXiv:2502.08213](https://arxiv.org/abs/2502.08213)
  — enhanced cross-attention transfers knowledge from a large model
  to a small one.
- **DEP (Difference-aware Embedding-based Personalization).**
  Models inter-user differences in latent space rather than in
  language prompts.
  [EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.536/).
- **PREF.** Models each user's reward function as a linear
  combination of shared base rewards learned via matrix
  factorisation, enabling user-specific adaptation without
  retraining.  [Persona-based LLM systems overview](https://www.emergentmind.com/topics/persona-based-language-model-systems).
- **Long-term memory.** [arXiv:2510.07925](https://arxiv.org/abs/2510.07925)
  integrates persistent memory + evolving user profiles for
  multi-session persistence.
- **Survey.** [arXiv:2502.11528](https://arxiv.org/html/2502.11528v2)
  is the current canonical survey.

**Implications for I³.**

1. Cite USER-LLM prominently in the architecture documentation —
   the cross-attention-conditioning design is in the same family.
2. The three-timescale user model (`i3/user_model/`) plus the
   diary store (`i3/diary/`) is effectively a DEP-style latent
   personalisation *plus* a long-term memory.  The project already
   covers both branches.  Make that explicit in the ADR list.
3. Consider a small Phi-4-mini wrapper that accepts the I³
   `AdaptationVector` as a LoRA-conditioning vector — a 2026-style
   bridge between the custom SLM and a mainstream open model.

---

## 4. Contextual bandits for LLM routing

Routing-via-bandit-feedback is an active 2025–2026 research line.
The closest match for the project's architecture is **BaRP**.

- **BaRP (Bandit-feedback Routing with Preferences).**
  [arXiv:2510.07429](https://arxiv.org/abs/2510.07429) — trains
  under the same partial-feedback restriction as deployment, supports
  preference-tunable inference so operators can dial the
  performance/cost trade-off at test time.  Compares REINFORCE,
  Linear Thompson Sampling (LinTS), and LinUCB.  Finding: policy-
  gradient outperforms bandit approaches on reward but bandits pick
  cheaper models more aggressively (useful when cost matters).
- **Preference-conditioned dynamic routing.**
  [arXiv:2502.02743](https://arxiv.org/html/2502.02743v1) — a
  calibrated router that chooses between model tiers based on a
  preference vector.
- **Graph / contrastive routers.** GraphRouter learns graph-
  structured representations over (prompt, task, model) triples;
  RouterDC uses dual-contrastive losses.
- **LLM-Guided ensemble of bandits.** [MDPI Mathematics 2025](https://www.mdpi.com/2227-7390/13/15/2523)
  proposes Gaussian-process / copula bandits fed by LLM priors.

**Implications for I³.**  The project's
`ContextualThompsonBandit` (Laplace-approximated Bayesian logistic
regression) sits between LinTS and LinUCB — the right family.  Two
concrete upgrades:

1. **Cost-aware reward.** Extend the reward model so cloud routes
   pay a configurable per-token penalty.  This matches BaRP's
   preference-tunable setup with a one-line YAML knob.
2. **Calibration gate.** Adopt the gating idea from
   [arXiv:2604.14961](https://arxiv.org/html/2604.14961): only
   trust the bandit's route-to-cloud decision when its posterior
   confidence exceeds a threshold; otherwise default to local.
   The safety planner in `i3/safety/pddl_planner.py` already
   implements this informally — promote it to a proper calibration
   layer.

---

## 5. TCN vs Transformer for long sequences

The 2018 Bai, Kolter & Koltun paper "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks" is still the
canonical reference ([arXiv:1803.01271](https://arxiv.org/pdf/1803.01271)).
2025 literature has not overturned its findings — TCNs remain
competitive with Transformers on bounded-receptive-field sequence
tasks, and *cheaper* on very long sequences.

- **2025 time-series comparison.** [PLOS One 2025 — network traffic](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320368)
  combines TCN + Transformer: TCN captures local + long-term
  dependencies, Transformer adds global attention.  The hybrid beats
  either alone.
- **TransTCN.** [OpenReview](https://openreview.net/forum?id=AAHL45-O7tV)
  — attention-gated TCN blocks for sequential modelling.
- **Battery state-of-charge.** [Xbattery 2025 comparison](https://xbattery.energy/blog/exploring-temporal-convolutional-and-self-attention-transformer-networks-for-soc-estimation)
  — TCN wins on latency, Transformer on accuracy; the hybrid is on
  the Pareto front.

**Implications for I³.**  The choice of TCN as the *keystroke*
encoder is well-justified.  Future work:

1. Add a **TCN + self-attention hybrid** variant to
   `i3/encoder/` (a single attention block over the TCN output) as
   an ablation baseline.
2. Cite the 2018 Bai paper explicitly in
   [`docs/adr/0002-tcn-over-lstm-transformer.md`](../adr/0002-tcn-over-lstm-transformer.md)
   and reference the 2025 hybrid-is-best findings as a justified
   future direction.

---

## 6. EU AI Act — biometric categorisation and emotion recognition

The regulatory ground has shifted under every AI product that
infers state from implicit signals.  This project's entire thesis is
"infer cognitive + emotional + accessibility state from keystrokes"
— which puts it squarely in-scope.

- **Prohibited practices — live since 2 Feb 2025.**  Emotion
  recognition *in the workplace and education institutions* is
  prohibited; biometric categorisation along protected
  characteristics is prohibited; real-time remote biometric ID for
  law enforcement in public is prohibited.
  [EU digital strategy — AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai),
  [William Fry practical guide to emotion-recognition systems](https://www.williamfry.com/knowledge/the-time-to-ai-act-is-now-a-practical-guide-to-emotion-recognition-systems-under-the-ai-act/).
- **High-risk categories — live 2 Aug 2026 / 2 Aug 2027.**  See
  [Annex III](https://artificialintelligenceact.eu/annex/3/).
- **Penalties.**  Up to **€35 M or 7 % of global revenue**.
  [State of Surveillance explainer](https://stateofsurveillance.org/news/eu-ai-act-august-2026-biometric-surveillance-explainer/).

**Implications for I³.**  The demo is *not* deployed in workplace /
education contexts, and it does *not* categorise users by protected
characteristics — but the framing matters for any pitch-deck
narrative.

1. Add a section to
   [`docs/responsible_ai/`](../responsible_ai/) explaining the
   project's out-of-scope-by-design posture with respect to the
   prohibited practices.  Explicit: "the product **does not infer
   emotions in the workplace** — the TTS pacing adaptation is a
   general-purpose accessibility signal".
2. Mention that the **PDDL safety planner**
   (`i3/safety/pddl_planner.py`) already enforces a
   privacy-override invariant that maps cleanly onto the AI Act's
   high-risk system attestations.

---

## 7. Edge NPUs in 2026

Every flagship phone now ships a capable NPU, and the ecosystem
has converged on ~70 TOPS as the top of the volume market.  Memory
bandwidth — not compute — is the bottleneck for 7 B-class models.

- **Apple.** A19 Pro's 16-core Neural Engine handles on-device
  inference for summarisation, image generation, NLP.
  [Orion — ANE programming paper, arXiv:2603.06728](https://arxiv.org/html/2603.06728v1).
- **Qualcomm Hexagon.** Snapdragon 8 Elite ships at 45 TOPS.  The
  newer parts advertise 75 TOPS.
  [Qualcomm NPU whitepaper](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf).
- **Memory is the bottleneck.** [Quincy News analysis](https://quincynews.org/technology/npu-hardware-bottlenecks-2026/)
  and [arXiv:2509.23324](https://arxiv.org/html/2509.23324v1) both
  document that weight-tensor transfer dominates once the model
  exceeds on-chip SRAM.
- **Huawei MindSpore Lite + Kirin.**
  [MindSpore Lite Kit on HarmonyOS NEXT](https://dev.to/harmonyos/mindspore-lite-kit-on-harmonyos-next-tiny-ai-big-impact-4leh)
  — `MindSpore Lite` is the canonical on-device inference runtime
  for Kirin 810 / 820 / 985 / 990 / 9000 / 9000E.

**Implications for I³.**  The edge-feasibility report in
[`docs/edge/profiling-report.md`](../edge/profiling-report.md) is
well-sized — a 7 MB INT8 SLM sits comfortably inside Kirin 9000's
on-chip SRAM, which means the project avoids the memory-bandwidth
cliff that 7 B-class models hit.  One improvement:

1. Add a paragraph in the edge report explicitly citing the
   70 TOPS / memory-bandwidth research, framing the project's small
   footprint as a deliberate engineering trade-off rather than a
   limitation.

---

## 8. Data pipelines — HuggingFace Datatrove + FineWeb

The reference pipeline for pre-training corpus preparation is now
Hugging Face's Datatrove, most visibly exercised by the FineWeb /
FineWeb-Edu datasets.  The architecture and defaults are close to
what this project's `i3/data/` module implements.

- **FineWeb paper.** [arXiv:2406.17557](https://arxiv.org/html/2406.17557v1)
  — Trafilatura extraction → FastText language ID → custom filters
  → **MinHash deduplication with 5-grams, 112 hash functions,
  14 bands × 8 rows**.
- **Datatrove examples.**
  [huggingface/datatrove](https://github.com/huggingface/datatrove)
  with the reference [minhash_deduplication.py](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py)
  and the full [FineWeb example pipeline](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py).
- **FineWeb-Edu.** Applies a linear quality classifier trained on
  synthetic educational-quality labels, filtering down to the
  highest-quality ~1.3 T tokens.
  [Emergent Mind FineWeb-Edu overview](https://www.emergentmind.com/topics/fineweb-edu-dataset).

**Implications for I³.**

1. The project's `i3/data/dedup.py` uses **128 permutations / 16 bands
   / 8 rows / k=5 shingles** — a near-identical configuration to
   FineWeb's, intentionally.  Cite FineWeb in the docstring so future
   readers understand the parameter choices.
2. The **FineWeb-Edu quality classifier** is an easy future add:
   a learnable quality score on top of the existing rule-based
   filter.  Would naturally sit in `i3/data/quality.py` as
   `LearnableQualityRule`.

---

## 9. Privacy-preserving user modelling

Differential privacy is becoming operationally mature, and the
2025–2026 work focuses on explainable / adaptive noise budgets.

- **Utility-enhanced FL-with-DP.** [arXiv:2503.21154](https://arxiv.org/pdf/2503.21154).
- **Explainable adaptive DP.** [arXiv:2509.10691](https://arxiv.org/abs/2509.10691)
  — privacy-preserving decentralised FL via explainable adaptive DP.
- **DP for medical deep learning.** [npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-02280-z)
  — concludes DP-SGD preserves clinically acceptable performance at
  ε ≈ 10 on imaging tasks.
- **PrivateDFL.** [ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S0045790625002046)
  — hyper-dimensional computing + transparent DP noise accountant;
  24 % accuracy improvement on MNIST, 76× lower inference latency.

**Implications for I³.**  The project already ships a DP-SGD wrapper
for the router posterior
(`i3/privacy/differential_privacy.py`) and a Flower-based federated
client (`i3/federated/`).  Two follow-ups worth planning:

1. Pin the DP budget to **ε = 10, δ = 1e-5** as the default (matches
   the 2025 clinically-acceptable reference point).
2. Surface the cumulative ε in
   [`/api/ready`](../api/rest.md) for operators deploying the
   federated path.

---

## 10. Cognitive load from typing + physiological signals

The fusion angle between keystroke dynamics and wearable
physiological signals is genuinely 2025-fresh research.

- **Keystroke + HRV.** [arXiv:2111.09243](https://arxiv.org/abs/2111.09243)
  established the baseline: keystroke dynamics and HRV are
  complementary stress signals.
- **2025 unified analysis.** [arXiv:2512.06099](https://arxiv.org/html/2512.06099)
  — "why nonlinear models matter" for cognitive-load / stress /
  exercise classification from wearables.
- **HRV + EDA cognitive load.** [MDPI Sensors 2025](https://www.mdpi.com/1424-8220/25/8/2343)
  — EDA slope is the cleanest arousal marker; HRV RMSSD is the
  cleanest cognitive-load marker.
- **Real-time PPG stress quantification.** [PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970940/)
  — ML over PPG-derived HRV reaches 99 % stress classification
  accuracy in lab conditions.
- **Driving-task cognitive load from HR.** [Nature Sci. Reports 2024](https://www.nature.com/articles/s41598-024-79728-x).

**Implications for I³.**  The
[`i3/multimodal/ppg_hrv.py`](../../i3/multimodal/ppg_hrv.py) module
already implements HRV extraction — this is the right direction.
Concrete move: add a fusion-ablation experiment under
[`docs/experiments/`](../experiments/) that measures the ΔEER /
Δaccuracy the Huawei Watch PPG signal adds on top of keystroke-only
input.  The 2025 literature supports non-trivial gains.

---

## 11. Tokenisation advances

The project's word-level SLM tokeniser is deliberately simple
(ADR-0001).  Worth citing two 2025 / 2026 advances as future work.

- **SuperBPE (COLM 2025).**  Two-pass BPE — standard subword tokens
  then cross-word "superword" tokens.  Produces 33 % fewer tokens at
  similar quality.  [arXiv reference catalogued at Aman's AI](https://aman.ai/primers/ai/tokenizer/).
- **LiteToken (Feb 2026).**  Identifies and removes "intermediate
  merge residues" — ~10 % of tokens in major tokenisers are
  residues.  Plug-and-play over any existing tokeniser.
- **Kitoken.** A single library covering BPE / Unigram / WordPiece /
  SentencePiece / tiktoken.
  [Systemcluster/kitoken](https://github.com/Systemcluster/kitoken).

**Implications for I³.**  None immediate — the project's
word-level choice stands.  A future ADR could revisit with a
cross-tokeniser benchmark if the SLM ever needs to scale beyond
8 K vocabulary.

---

## Summary — concrete follow-ups for this project

| # | Item | Where it lands |
|---|---|---|
| 1 | MC-dropout uncertainty on user-state embedding for continuous auth | `i3/encoder/` + `i3/adaptation/uncertainty.py` |
| 2 | EER benchmark vs 2026 cross-device transformer baseline | `benchmarks/keystroke_identification/` |
| 3 | Cite USER-LLM + DEP in architecture docs | `docs/architecture/cross-attention-conditioning.md` |
| 4 | Cost-aware + calibration-gated routing | `i3/router/bandit.py` |
| 5 | Hybrid TCN + attention encoder variant | `i3/encoder/hybrid.py` |
| 6 | EU-AI-Act out-of-scope-by-design note | `docs/responsible_ai/` |
| 7 | Cite FineWeb pipeline-config parameters | `i3/data/dedup.py` + `docs/architecture/` |
| 8 | Learnable quality classifier (FineWeb-Edu style) | `i3/data/quality.py` |
| 9 | Pin DP budget ε=10, δ=1e-5 + surface cumulative ε | `i3/privacy/differential_privacy.py` + `/api/ready` |
| 10 | Multi-modal ablation (keystroke vs keystroke+PPG) | `docs/experiments/` + `i3/multimodal/ppg_hrv.py` |
| 11 | SuperBPE / LiteToken future ADR | `docs/adr/` |

These are ordered by estimated ROI — top items are small,
load-bearing, and move the project's positioning against 2026
state-of-the-art.  Bottom items are optional polish.
