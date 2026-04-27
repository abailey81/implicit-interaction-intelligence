# Research reading list — what I3 stands on

> **Iter 51 (2026-04-27).**  Closes the JD's *"Stay up to date with the
> latest research in ML, LLMs, and HCI-related AI applications"*
> bullet.  15 papers, 2017–2026, each with a one-paragraph note
> explaining what the paper says and exactly where I³ uses or extends
> it.  Bibliography in BibTeX at
> [`docs/paper/references.bib`](../paper/references.bib).

---

## On-device personalisation primitives

### 1. Hu et al. 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*  (ICLR 2022)

Showed that constraining fine-tuning updates to ``ΔW = W_a W_b`` with
rank `r << d` recovers nearly the full-fine-tune quality at < 1 % of
the parameter count.

**Where I³ uses it.**
1. The per-biometric personalisation layer
   ([`i3/personalisation/lora_adapter.py`](../../i3/personalisation/lora_adapter.py))
   applies LoRA to the 64-d→8-d adaptation projection — rank=4,
   544 trainable params per user, gated by typing biometric.
2. The Qwen3-1.7B HMI command-intent fine-tune
   ([`training/train_intent_lora.py`](../../training/train_intent_lora.py))
   uses LoRA at rank=16 on every linear in attention + MLP, 17.4 M
   trainable params (1.0 % of the 1.7 B base).

### 2. Liu et al. 2024 — *DoRA: Weight-Decomposed Low-Rank Adaptation*  (arXiv 2402.09353)

Decomposes the LoRA update into magnitude + direction.  Reported
+1–2 pts over vanilla LoRA at the same parameter count.  ~10–15 %
slower training (extra norm computation).

**Where I³ uses it.**  `--use-dora` flag in
[`training/train_intent_lora.py`](../../training/train_intent_lora.py),
on by default.  Toggleable so the comparison vs vanilla LoRA can be
measured cleanly.

### 3. Jain et al. 2023 — *NEFTune: Noisy Embeddings Improve Instruction Finetuning*  (NeurIPS 2023)

Adds uniform noise to the input-embedding tensor during training,
scaled by `α / sqrt(L · d)`.  Reported +5–10 pts on instruction-
following benchmarks at zero compute cost.

**Where I³ uses it.**  Hooked directly into the embedding `forward`
in `train_intent_lora.py` so it's compatible with any LoRA / DoRA /
quantisation backend.  Default `α = 5.0`.

### 4. Houlsby et al. 2019 — *Parameter-Efficient Transfer Learning for NLP*  (ICML 2019)

The bottleneck-adapter pattern — small per-task modules layered on
top of a shared backbone, trained online from a few labelled examples.

**Where I³ uses it.**  The per-biometric LoRA adapter pattern is
exactly this, mirrored at per-user granularity rather than per-task
— exactly the right grain for an on-device companion that adapts to
one human's preferences over weeks of use.

---

## SLM training & efficient inference

### 5. Vaswani et al. 2017 — *Attention Is All You Need*  (NeurIPS 2017)

The transformer.  Foundational.

**Where I³ uses it.**  The from-scratch `AdaptiveTransformerV2`
([`i3/slm/model.py`](../../i3/slm/model.py)) is a Pre-LN transformer
with cross-attention conditioning on every block.

### 6. Shazeer et al. 2017 — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*  (ICLR 2017)

Mixture-of-Experts: route each token through a sparse subset of
experts to scale capacity without proportional compute.

**Where I³ uses it.**  2-expert MoE on every transformer block in
`AdaptiveTransformerV2`.  See [`i3/slm/blocks.py`](../../i3/slm/blocks.py).

### 7. Graves 2016 — *Adaptive Computation Time for Recurrent Neural Networks*  (arXiv 1603.08983)

ACT: per-step halt probability so the network adapts compute to the
input difficulty.

**Where I³ uses it.**  ACT in `AdaptiveTransformerV2` — early-exit
when the halt threshold (default 0.99) is reached.  Saves compute on
easy turns.

### 8. Dettmers et al. 2022 — *8-bit Optimizers via Block-wise Quantization*  (NeurIPS 2022)

bitsandbytes' `AdamW8bit` halves the optimiser-state memory footprint
at the cost of < 0.5 % SFT loss.

**Where I³ uses it.**  `--use-8bit-adam` flag on the LoRA training
loop.  Critical on the 6.4 GB RTX 4050.

### 9. Loshchilov & Hutter 2017 — *SGDR: Stochastic Gradient Descent with Warm Restarts*  (ICLR 2017)

Cosine annealing with periodic warm restarts — empirically helps
escape narrow loss basins on small datasets.

**Where I³ uses it.**  The LR schedule in `train_intent_lora.py` —
linear warmup over 30 steps then two cycles of cosine annealing
(amplitude 1.0 then 0.7).

---

## Conversational coherence & dialogue

### 10. Lewis et al. 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*  (NeurIPS 2020)

RAG: combine a sparse/dense retriever with a generator so the model
grounds in retrieved evidence instead of memorising.

**Where I³ uses it.**  `i3/slm/retrieval.py` — embedding-based
retrieval over the curated overlay; results condition the SLM
generation pass.

### 11. Saunders et al. 2022 — *Self-Critiquing Models for Assisting Human Evaluators*  (arXiv 2206.05802)

Self-refine: generate, critique, revise.

**Where I³ uses it.**  The SLM on-topic gate
([`i3/pipeline/engine.py`](../../i3/pipeline/engine.py) §"SLM
on-topic gate") rejects grammatically-clean nonsense by checking
content-keyword overlap with the query topic — a cheaper but
spiritually-similar self-check.

---

## Reasoning & alignment

### 12. Rafailov et al. 2023 — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*  (NeurIPS 2023)

DPO: optimise the Bradley-Terry objective directly on preference data,
no PPO.

**Where I³ uses it.**  `i3/router/preference_learning.py` (871 LOC)
— DPO over the bandit's reward signal so cloud-vs-local routing
learns from explicit user A/B picks.

### 13. Mehta et al. 2025 — *Active Learning for Direct Preference Optimization*  (ICLR 2025)

D-optimal active query selection for DPO — near-optimal sample
efficiency (~10–20 pairs per user).

**Where I³ uses it.**  Active query-selection in
`preference_learning.py` so each A/B pick is chosen to maximally
disambiguate the reward model.

### 14. DeepSeek-AI 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*

671 B MoE reasoning model trained with RL; distilled into Qwen-arch
1.5B / 7B / 14B / 32B and Llama-arch 8B / 70B.

**Where I³ relates.**  The 1.5 B distill was a candidate for the
HMI command-intent fine-tune target; we ultimately picked Qwen3-1.7B
(newer, multimodal-adjacent thinking mode, broader baseline) but
the reasoning-distill brand carries weight in 2026 ML interviews.
See [`finetune_artefact.md`](finetune_artefact.md) §1.

---

## HCI / multimodal

### 15. Kartynnik et al. 2019 — *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs*  (CVPR Workshop)

MediaPipe Face Mesh — 468 facial landmarks at real-time on commodity
mobile.

**Where I³ uses it.**  `i3/multimodal/vision.py` extracts 8
facial-affect features (eye-aspect ratio, mouth-aspect ratio, gaze
offset, head-pose pitch/yaw, brow-furrow AU4, smile AU12) from the
landmark grid for the affect-shift detector.

### 16. Soukupova & Cech 2016 — *Real-Time Eye Blink Detection using Facial Landmarks*  (CVWW)

Eye-aspect ratio (EAR) for blink / drowsiness detection.

**Where I³ uses it.**  EAR is one of the eight features in
`i3/multimodal/vision.py:FacialAffectExtractor`.  Feeds into the
accessibility-mode controller when sustained drowsiness is detected.

---

## Safety & interpretability

### 17. Bai et al. 2022 — *Constitutional AI: Harmlessness from AI Feedback*  (Anthropic)

Constitutional principles as an alternative to per-output human
labels — model self-evaluates against a small principle set.

**Where I³ uses it.**  The safety classifier in
`i3/safety/classifier.py` plus the constitutional principle strings
displayed in side-channel safety chips (iter 51 moved these out of
the chat bubble).

### 18. Cunningham et al. 2023 — *Sparse Autoencoders Find Highly Interpretable Features in Language Models*  (arXiv 2309.08600)

Sparse autoencoders trained on activations expose human-interpretable
feature directions.

**Where I³ uses it.**  `i3/interpretability/` (G3 batch from the
Advancement Plan v3) — applied to the SLM's cross-attention
activations to see *what dimension of the user-state vector
correlates with what generation behaviour*.

---

## Continual / meta learning

### 19. Kirkpatrick et al. 2017 — *Overcoming Catastrophic Forgetting in Neural Networks*  (PNAS, EWC)

Elastic Weight Consolidation: regularise updates to weights important
for prior tasks so new-task fine-tuning doesn't catastrophically
forget.

**Where I³ uses it.**  `i3/continual/` (F-5 batch).  Applied to the
per-biometric LoRA adapters so a long-term user's adapter doesn't
forget its early-session learning when later sessions push it.

### 20. Finn et al. 2017 — *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*  (ICML 2017)

MAML: meta-train so the inner-loop few-shot adaptation converges in
~10 examples.

**Where I³ uses it.**  `i3/meta_learning/` (G5 batch).  Initialises
new-user LoRA adapters from a meta-trained prior so the user reaches
useful personalisation faster.

---

## Honesty notes

* The papers above are real and well-known; the paper-to-I3-mapping
  is faithful to the actual code (every "where I³ uses it" cites a
  file that exists).
* This list is not exhaustive — `docs/research/` has 26 entries.
  This is the *interview-grade* subset.
* I³ does not currently *advance* any of these papers.  The
  contribution of I³ is the *combination* — per-biometric LoRA gated
  by continuous typing-biometric authentication is the patent-
  disclosure-grade novelty
  ([`docs/patent/provisional_disclosure.md`](../patent/provisional_disclosure.md)).
