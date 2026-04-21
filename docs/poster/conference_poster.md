# Implicit Interaction Intelligence (I³)

## Cross-Attention Conditioning for On-Device Adaptive Language Models from Behavioural Signals

**Tamer Atesyakar**  ·  *Huawei London Human-Machine Interaction Laboratory (candidate submission)*  ·  April 2026

---

> **Rendering.** This poster is authored as single-page Markdown and
> is intended to be rendered at A0 via pandoc and a poster LaTeX
> template. Recommended command:
>
> ```bash
> pandoc conference_poster.md \
>   --template=templates/a0-poster.latex \
>   -V papersize=a0 -V orientation=portrait \
>   -V fontsize=22pt -V columns=3 \
>   --pdf-engine=xelatex \
>   -o poster.pdf
> ```
>
> A `beamerposter` or `tikzposter` backend works equally well;
> substitute the template argument accordingly. For live demos an
> HTML render (`pandoc -s --mathjax -o poster.html`) is adequate for
> projector use.

---

## Panel 1 — The Idea

**Abstract.** Most on-device assistants personalise by adding a
paragraph of instructions to the model's input. That wastes
context, dilutes with conversation length, and relies on
instruction-following small language models do not reliably have.
I³ replaces prompt-based personalisation with architectural
conditioning: keystroke dynamics, linguistic complexity, and
session rhythm are compressed into four conditioning tokens that
condition every transformer block at every token position.

**Highlights.**

- 32-dim implicit-signal feature vector, language-agnostic, <2 ms.
- No tokens consumed from the conversation context budget.

---

## Panel 2 — The Architecture

**Abstract.** Seven sequential layers plus two cross-cutting
concerns. Keystroke events enter; a 32-dim feature vector comes
out. A dilated causal TCN trained with NT-Xent contrastive loss
produces a 64-dim user-state embedding. Three-timescale EMAs
(instant, session α=0.3, long-term α=0.1) plus Welford online
statistics maintain per-user baselines without storing raw text.
A four-adapter controller emits an 8-dim adaptation vector.
A contextual Thompson-sampling router picks between the local
SLM and a cloud LLM — with an architectural override for sensitive
topics.

**Highlights.**

- Seven layers, nine async pipeline steps, everything on-device.
- Welford-online per-feature statistics; no raw text persisted.

```
 signals -> 32-dim -> TCN(L=4, d=1..8) -> 64-dim z_user
                                              |
            [z_user(64) ; v_adapt(8)] -> MLP -> 4 x 256 tokens  C
                                              |
           token seq x -> block x L (self-attn; CROSS-ATTN to C; FFN)
                                              |
                                          response
```

---

## Panel 3 — The Novel Contribution: Cross-Attention Conditioning

**Abstract.** The four conditioning tokens are consumed by a
dedicated cross-attention sub-layer positioned between
self-attention and feed-forward at every transformer block.
The conditioning is dynamic per forward pass (unlike LoRA,
prefix-tuning, adapters), carries no context cost (unlike prompt
tuning), and cannot be ignored by the generator (attention weights
are trained end-to-end with the generation loss). The mechanism is
compatible with pretrained or from-scratch generators; the
prototype is from-scratch in PyTorch with no HuggingFace.

**Highlights.**

- ≈ 2.1× KL shift vs. noise-equivalent perturbation across
  adaptation states on held-out dialogue (§6.2 of the paper [1]).
- Parameter overhead over a bare transformer: ≈ 5 %.

$$\text{CrossAttn}(X, \mathbf{C}) = \text{softmax}\!\left(\frac{X W_Q (\mathbf{C} W_K)^\top}{\sqrt{d_k}}\right) \mathbf{C} W_V$$

---

## Panel 4 — The Live Demo

**Abstract.** A four-phase live demo: *cold-start* (2 min, baseline
establishing); *energetic* (1 min, cognitive load rises, style
mirrors up); *fatigue* (2 min, embedding dot visibly migrates,
responses warmer and shorter, router shifts to local SLM); and
*accessibility* (2 min, concurrent correction/pause elevation
without complexity rise triggers simplification — no settings menu,
no toggle, it just adapts). Dashboard panels show 2D embedding
projection with trail, four adaptation gauges, router confidence,
engagement score, and the privacy-safe diary.

**Highlights.**

- Run locally: `git clone … && bash scripts/run_demo.sh`.
- Dashboard at `http://localhost:8000` via FastAPI + WebSocket.

```
  +-------------------+   (scan to view the live demo recording;
  |                   |    placeholder QR — replace with the
  |     [ QR code ]   |    actual demo video URL at print time.)
  |                   |
  +-------------------+
```

---

## Panel 5 — Edge Feasibility

**Abstract.** Full system occupies ≈ 6.4M parameters, 25 MB FP32,
7 MB INT8. Host P50 latency on Apple M2 single-threaded CPU is
170 ms for the full local pipeline (100 iterations + 5 warmup).
TOPS-ratio extrapolation to Kirin 9000 (2.0 INT8 TOPS) with a
conservative κ=1.5 kernel-efficiency factor gives 50–80 ms.
The 50 % memory-budget rule is satisfied for all four Huawei edge
targets (Kirin 9000, 820, A2, Smart Hanhan); for the 64 MB Smart
Hanhan class the recommended deployment is TCN-on-device with SLM
on the paired phone.

**Highlights.**

- 25 MB → 7 MB (3.5×) under INT8 dynamic quantisation.
- Conversion path: PyTorch → ONNX (opset ≥ 17) → MindSpore Lite.

---

## Panel 6 — Future Work

**Abstract.** Three directions. (i) Multi-modal extension — the TCN
is modality-agnostic; voice prosody, touch pressure, gaze dwell,
and accelerometer streams extend the 32-dim vector naturally. (ii)
Federated learning with differential privacy for cross-user base-
model improvement without breaking the privacy-by-architecture
guarantee. (iii) Climbing the L1–L5 device-intelligence ladder —
I³ sits at L1–L2 today; the ≈ 680-byte user-state sync payload
plus CRDT merging of Welford statistics is the designed L3 path,
and a session-checkpoint format is the designed L4 path.

**Highlights.**

- L3 plan: HarmonyOS Distributed Data Management, per-user key rotation.
- Accessibility-detection research: multi-modal, bias not gate, user-visible.

---

## References

[1] Atesyakar, T. *Implicit Interaction Intelligence: Cross-Attention
Conditioning for On-Device Adaptive Language Models from Behavioural
Signals.* Candidate paper, April 2026.
[4] Chen et al. *SimCLR / NT-Xent.* ICML 2020.
[5] Bai, Kolter, Koltun. *Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling.*
arXiv:1803.01271, 2018.
[6] Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
[7] Xiong et al. *On Layer Normalization in the Transformer
Architecture.* ICML 2020.
[8] Hu et al. *LoRA.* ICLR 2022.
[9] Russo et al. *A Tutorial on Thompson Sampling.* FnT-ML 11(1),
2018.
[1]–[3] keystroke-dynamics HCI references: Epp et al. 2011;
Vizer et al. 2009; Zimmermann et al. 2014.
[12] Huawei MindSpore Team. *MindSpore Lite Technical Report.*
[27] Huawei Technologies. *HarmonyOS 6 and the Harmony Multi-Agent
Framework (HMAF): Four-Pillar Architecture.* HDC 2025.

Full bibliography: see `docs/paper/references.bib`.

## Acknowledgements

The author thanks the reviewers at the Huawei London HMI Lab for
their consideration, the Edinburgh Joint Lab's published research
direction on implicit-signal personalisation (Prof. Malvina Nissim,
March 2026) for framing, and the authors of the primary references
above. All errors are the author's own. This work represents a
candidate submission and not an organisational product claim.

---

*Contact: `t.ates232004@gmail.com` · Code + docs: see `README.md` at repo root.*
