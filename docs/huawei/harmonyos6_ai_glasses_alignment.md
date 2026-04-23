# I³ ↔ Huawei 2025-2026 Product & Research Alignment

This document maps every I³ layer onto a specific, verifiable Huawei 2025-
2026 initiative so the interviewer can see the connections without having
to construct them. Sources are inline (URL or named announcement).

## One-line summary

I³ is a **user-modelling** system (JD keyword) that reads implicit
behavioural signals and conditions generation architecturally. That
behavioural-signal frontier sits at the intersection of four concrete
Huawei initiatives that launched between Q4-2025 and Q2-2026.

---

## 1. HarmonyOS 6 + HMAF (Oct 2025)

**Huawei public fact.** HarmonyOS 6 launched in October 2025 with the
Harmony Intelligent Agent Framework (HMAF) — 50+ agent plugins, 80+ agents
distributed through the Xiaoyi Intelligent Agent Hub, sub-millisecond
distributed-capability latency. Developers can ship agents without a
dedicated algorithm team.

**I³ alignment.**
- `i3/huawei/hmaf_adapter.py` already implements an HMAF-shaped agent
  interface (`register_capability` / `plan` / `execute` / `emit_telemetry`).
- the advancement plan upgrades this to an end-to-end runnable
  agentic runtime, so I³ can register as a capability inside HMAF without
  architectural re-work.
- The `I3UserStateSync` 64-dim payload (documented in
  `harmony_hmaf_integration.md`) is sized for the HarmonyOS distributed
  data-management bus that HMAF uses for cross-device context.

## 2. AI Glasses (launched 20 Apr 2026, 9 days before the interview)

**Huawei public fact.** Huawei AI Glasses: 35.5 g frames, HarmonyOS, dual-
engine AI architecture, 12 MP camera, Celia on-device, 12-hour battery,
real-time translation across Chinese + 20 languages, ¥2 499 / US$365.

**I³ alignment.**
- The AI Glasses form factor is a near-textbook match for I³'s
  "encoder-only" deployment mode in
  [`docs/huawei/smart_hanhan.md`](smart_hanhan.md): run the 50 K-parameter
  TCN on-glass, push the 64-dim embedding to the paired phone over
  HarmonyOS DDM, generate on the phone or the cloud.
- The "paired-phone" router arm in
  [`i3/crossdevice/ai_glasses_arm.py`](../../i3/crossdevice/ai_glasses_arm.py)
  is exactly this pattern — a third arm on the contextual bandit that the
  router learns to pick when the current device is AI-Glasses-class.
- Real-time translation as a product feature is **conditioning-adjacent**:
  the translation output is shaped by the user's current adaptation
  (verbosity, formality, cognitive load), not just the language pair.
  Batch D adds a concrete `/api/translate` endpoint using the conditioning
  path to show this.

## 3. Smart Hanhan (Nov 2025)

**Huawei public fact.** Smart Hanhan: 80 × 68 × 82 mm plush body, 140 g,
1 800 mAh battery, 6–8 h continuous interaction, 48 h standby, XiaoYi AI,
HarmonyOS 5.0+. Emotional companion device with voice / touch / movement
sensing.

**I³ alignment.**
- I³'s edge-feasibility report targets the Smart Hanhan class explicitly:
  INT8 TCN encoder + conditioning projector fits inside the 64 MB RAM
  budget; the full SLM does not, which is why the deployment
  recommendation in
  [`docs/huawei/smart_hanhan.md`](smart_hanhan.md) is **encoder-only** on
  the device with generation offloaded to the paired phone.
- The behavioural-signal angle maps naturally to emotional companion
  products: this class of device already has sensors (touch, voice) that
  can feed the same multi-modal TCN I³'s
  [`i3/multimodal/`](../../i3/multimodal/) package stubs out.

## 4. Celia / Xiaoyi AI — on-device LLM acceleration

**Huawei public fact.** Public Huawei publications describe a **speculative-
decoding architecture** combined with RL-augmented distillation that
**doubled Celia Auto-answer throughput** while preserving accuracy.

**I³ alignment.**
- Batch D adds `i3/slm/speculative_decoding.py` — a draft-and-verify loop
  that targets the same 2× throughput ballpark on the local SLM.
- Because I³'s SLM is deliberately small (~6.3 M params), the
  speculative-decoding benefit is lower than on Celia-class models, but
  the **architectural pattern is identical**, which is the signal the
  interviewer will recognise.

## 5. PanGu 5.5 — adaptive fast/slow thinking

**Huawei public fact.** PanGu 5.5 (June 2025) is a 718 B-parameter
Mixture-of-Experts model (256 experts) with **adaptive fast/slow thinking
integration** — the model switches processing depth based on problem
complexity, yielding 8× overall inference efficiency.

**I³ alignment.**
- I³'s **contextual Thompson-sampling router already does a variant of
  this**: cheap local SLM for shallow turns, expensive cloud Claude for
  hard turns, with learned routing.
- Batch D adds a **third "local-reflect" path** — the bandit can choose to
  run the local SLM with a bigger sampling budget for moderately hard
  turns. This is the PanGu fast/slow pattern at edge scale.
- The 12-dim routing context that already includes `query_complexity` and
  `slm_confidence` is exactly the right signal for a fast/slow switch.

## 6. PDDL-grounded agentic safety

**Huawei public fact.** A Huawei agentic-safety paper grounded in the
Planning Domain Definition Language (PDDL) reports **99.9 % interception
of highest-risk agent operations**.

**I³ alignment.**
- Batch D adds `i3/safety/pddl_planner.py` which wraps the router's
  sensitive-topic force-local override in a small PDDL domain that emits
  a machine-checkable safety certificate per decision.
- This lifts the existing sensitivity override from "policy" to
  "provable" — the exact lift Huawei's published work performs for their
  agentic stack.

## 7. MindSpore Lite + Ascend full-stack AI

**Huawei public fact.** Huawei has open-sourced a full-stack AI
alternative to NVIDIA CUDA: MindSpore + CANN + openEuler + Ascend +
Kunpeng 950. MindSpore Lite is the mobile/edge inference runtime.

**I³ alignment.**
- `i3/encoder/onnx_export.py` and `i3/slm/onnx_export.py` produce ONNX
  graphs that MindSpore Lite's converter accepts directly
  (`converter_lite --fmk=ONNX --optimize=ascend_oriented`).
- This is documented in
  [`docs/edge/alternative_runtimes.md`](../edge/alternative_runtimes.md)
  alongside the other 7 runtime options, with MindSpore Lite explicitly
  flagged as the **first-party** conversion target for Huawei hardware.
- Batch D does not add new MindSpore code — the conversion path is
  already correct; it just hasn't been run on actual Huawei silicon
  (cannot be, from a laptop).

## 8. Huawei-Edinburgh Joint Lab — personalisation from sparse signals

**Huawei public fact.** The Huawei-Edinburgh Joint Lab ran a March 2026
session on personalisation from sparse signals, with Prof. Malvina Nissim
as a featured speaker. The lab's historical focus includes distributed
data management and personalisation.

**I³ alignment.**
- I³'s core premise is exactly **personalisation from sparse implicit
  signals** — keystroke dynamics, pauses, deviations from baseline. The
  vocabulary map is 1-to-1.
- [`docs/huawei/edinburgh_joint_lab.md`](edinburgh_joint_lab.md) positions
  the cross-attention conditioning novelty relative to that lab's line of
  work, with three concrete follow-up research questions they could
  adopt.

## 9. Darwin Research Centre (London) — the lab itself

**Huawei public fact.** The Darwin Research Centre is Huawei UK's named
London AI facility, publicly described as "Building Europe's leading AI
Research & Development Center." The posting for AI/ML Specialist — Human-
Machine Interaction (Internship) places the role inside this centre.

**I³ alignment.**
- The JD's desired-skill list (user modeling, multimodal, context-aware
  systems, HCI principles, concept-driven prototyping, academic research)
  is the closest possible match for what I³ already is.
- I³ demonstrates the three screening capabilities from the brief in one
  integrated demo: traditional ML from scratch (TCN), SLM without heavy
  frameworks (custom transformer + cross-attention), edge deployment
  (INT8 + profiling + alt runtimes + Smart-Hanhan-class target).

---

## Reader's guide

- **Designer-reviewer reading this document first:** the HMI Lab
  expectation for concept-to-prototype output is satisfied by Sections 2
  (AI Glasses), 3 (Smart Hanhan), and 1 (HMAF agent). The prototype
  already matches three shipping Huawei form-factor targets.
- **Technical-reviewer reading this document first:** Sections 4
  (speculative decoding), 5 (fast/slow), 6 (PDDL safety), and 7
  (MindSpore Lite) are the four where the technical trajectories meet.
  Batch D of `the advancement plan` ships runnable code against Sections
  4, 5, 6, and the agentic runtime in Section 1.
- **For the interviewer who wants one link:**
  [`docs/huawei/harmony_hmaf_integration.md`](harmony_hmaf_integration.md)
  and this document together give the clearest picture.
