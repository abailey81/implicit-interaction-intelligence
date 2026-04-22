# I³ — Advancement Plan (v2, Huawei-centred)

> Superseded v1 (which over-weighted the hiring manager). This version is
> grounded in (a) the verified JD for Huawei London HMI AI/ML Intern, (b)
> Huawei's 2025-2026 public research direction, and (c) the 2025-2026
> research frontier where HMI + on-device AI + user modelling meet.
>
> Execution: sequential, one implementation agent at a time (user preference).

---

## 0. Verified landscape (what I researched before writing this)

### 0.1 The lab

**Huawei AI lab in London** is the **Darwin Research Centre** — Huawei UK's
named London AI facility, tagged on their careers site as "Building Europe's
leading AI Research & Development Center." It sits inside Huawei's broader
UK R&D portfolio:

| Centre                     | Location   | Focus                                     |
|---------------------------|------------|--------------------------------------------|
| **Darwin Research Centre** | **London** | **AI (the target of this application)**   |
| Newton Research Centre    | Cambridge  | Core computing                             |
| Boole Research Centre     | Edinburgh  | Software competencies                      |
| Bragg Research Centre     | Ipswich    | III-V optical chip                         |
| Design Centre             | Bristol    | Design                                     |

Within Darwin, the **HMI Laboratory** is the concept-to-prototype unit the
intern posting targets. Public partnerships:

- **Huawei-Edinburgh Joint Lab** — distributed data management +
  personalisation (the Nissim talk in March 2026 on sparse-signal
  personalisation is directly adjacent to I³).
- **Imperial College-Huawei Data Science Innovation Lab**.
- Prior presence at 1 St Pancras Square, King's Cross (the original 2019
  computer-vision lab with 60 engineers — now the wider Darwin site).

### 0.2 The job description (verified from the live posting)

Required:
- Supervised / unsupervised / deep learning.
- From-scratch model building + fine-tune of pre-trained models.
- Training, inference pipelines, research deployment.
- Rapid prototyping in open-ended contexts.
- Cross-disciplinary communication.

Desired (these are the *differentiators*, and where I³ scores):
- **Human-AI interaction / user modeling** ← load-bearing phrase.
- NLP, multimodal, context-aware systems.
- **HCI principles, design thinking, concept-driven prototyping**.
- AI product development OR academic research.

The word **user modeling** is the single most aligned phrase in the JD. Lead
with it. I³ is a user-modelling system first; the conversation UI is only
the demo vehicle.

### 0.3 Huawei's 2025-2026 technical direction (public record)

What Huawei is actually publishing and shipping right now:

| Theme                            | Public evidence                                             | I³ alignment vector                                    |
|----------------------------------|-------------------------------------------------------------|--------------------------------------------------------|
| **HarmonyOS 6 + HMAF**           | Oct 2025 launch; 50+ agent plugins; 80+ agents in Xiaoyi Hub | Agentic runtime + multi-device handoff                 |
| **Smart Hanhan**                 | Nov 2025, 1800 mAh, 140 g, Xiaoyi AI, HarmonyOS 5.0+         | 64 MB-class edge footprint story                       |
| **AI Glasses**                   | **Apr 20 2026**, 35.5 g, HarmonyOS, dual-engine AI, Celia, 12MP cam, real-time translation across 20 languages, 12 h battery. Launched **9 days before the interview.** | Ultra-low-power wearable target; real-time translation as a concrete use case |
| **Celia on-device**              | Speculative decoding + RL-augmented distillation doubled Celia Auto-answer throughput | Add speculative decoding to the local SLM              |
| **PanGu 5.5**                    | 718 B-param MoE (256 experts), adaptive fast/slow thinking, 5 basic models + embodied AI platform, R2C protocol | Two-speed routing (cheap local / deep cloud) already matches this model pattern |
| **Agentic Core Framework**       | PDDL-grounded reliability; 99.9 % risky-op interception     | Formal-methods safety around the privacy override      |
| **MindSpore + CANN + openEuler** | Full-stack CUDA alternative; MindSpore Lite as conversion target | ONNX → MindSpore Lite conversion path (already doc'd)  |
| **Edinburgh Joint Lab**          | Nissim's March 2026 talk on personalisation from sparse signals | Direct citation in the pitch                           |
| **Huawei-Edinburgh distributed data management** | Historical research theme of the joint lab                  | Cross-device DDM sync mock (already sketched)          |

Every addition in this plan maps to at least one row in that table.

---

## 1. Personnel facts (defensive note, not a work item)

The `BRIEF_ANALYSIS.md` header now carries a corrections notice listing:
- No Apple employee history is supported by public sources for Matthew
  Riches (VocalIQ branding Feb 2015 only, pre-Apple acquisition).
- TextSpaced was created by Celina Riches, not Matthew.
- LinkedIn describes him as a senior HMI designer, not an ML engineer.

**Action for this plan:** no further edits. The correction header prevents
walking in with wrong facts; that is all this plan owes the topic. The
interview materials must not name him at all in claims — just "the panel."
I've stripped the TextSpaced reference from the speaker notes (covered in
Phase 1 below).

---

## 2. Huawei-anchored research advancements (sequential agents)

Ordered by expected impact, with each batch directly aligned to a row in
§0.3's table.

### Phase 1 (done by me, not an agent): quick interviewer-facing cleanups

- [x] `BRIEF_ANALYSIS.md` — corrections header added (Commit pending below).
- [ ] `docs/slides/speaker_notes.md` — drop TextSpaced / Apple product names.
- [ ] `docs/huawei/interview_talking_points.md` — replace
  "Matthew's-values moment" framing with "the accessibility moment — the
  hardest to fake, and the one that justifies `user modeling`."
- [ ] Add a new short doc `docs/huawei/harmonyos6_ai_glasses_alignment.md`
  mapping every I³ layer to a row in Huawei's 2025-2026 product / research
  table above, so the interviewer sees explicit connections.

### Batch A (1 agent) — Empirical ablation study

**Alignment:** academic research experience (JD desired skill).

Demonstrates that cross-attention conditioning actually shapes output vs.
(i) no adaptation and (ii) prompt-based adaptation. Runnable on random-init
SLM — no trained checkpoint needed.

Deliverables:
- `docs/experiments/preregistration.md` — hypothesis, metrics, analysis
  plan (Nosek et al. 2018 pre-registration convention).
- `i3/eval/ablation_experiment.py` — three-condition study, KL divergence
  + style-fidelity + length-distribution metrics, bootstrap 95 % CIs,
  Cohen's d effect sizes.
- `scripts/run_ablation_study.py` — CLI.
- `docs/experiments/ablation_report.md` — results with figures.
- `tests/test_ablation_statistics.py`.

### Batch B (1 agent) — Mechanistic interpretability study

**Alignment:** MIT 2026 Breakthrough Technology; matches Huawei's stated
interest in understanding their models (from HuaweiTech publications).

Use the already-committed `CrossAttentionExtractor` to run a real study:
activation patching, causal tracing, linear probes on each conditioning
token.

Deliverables:
- `i3/interpretability/activation_patching.py` (Meng et al. 2022 ROME
  methodology).
- `i3/interpretability/probing_classifiers.py`.
- `i3/interpretability/attention_circuits.py`.
- `scripts/run_interpretability_study.py`.
- `docs/research/mechanistic_interpretability.md`.

### Batch C (1 agent) — ImplicitAdaptBench (a real benchmark)

**Alignment:** "academic research" (JD); parallels the 2025-2026 benchmark
wave (PersonaLens, AlpsBench, PersoBench) but fills their gap: none
measure adaptation from **implicit behavioural signals**.

Deliverables:
- `benchmarks/implicit_adapt_bench/README.md` — benchmark spec, task,
  metrics, baselines, leaderboard scaffold.
- `benchmarks/implicit_adapt_bench/{schema,metrics,baselines}.py`.
- `docs/research/implicit_adapt_bench.md` — paper-style spec.

### Batch D (1 agent) — Huawei-ecosystem integrations

**Alignment:** rows in §0.3's table. Three concrete runnable features
directly tied to Huawei's 2025-2026 direction.

1. **Speculative decoding for the local SLM** (Huawei Celia parallel):
   draft-and-verify loop; targets the same 2× throughput ballpark. File:
   `i3/slm/speculative_decoding.py` + tests.
2. **PDDL-grounded safety planner for the privacy override**
   (Huawei agentic-safety parallel): wraps the router's sensitive-topic
   force-local path in a PDDL domain that produces a machine-checkable
   safety certificate per decision. File: `i3/safety/pddl_planner.py`.
3. **Adaptive fast/slow thinking router** (PanGu 5.5 parallel): adds a
   third path — "local-reflect" — where the bandit can run the SLM with a
   bigger compute budget (more sampling steps) for harder queries,
   mirroring PanGu's fast/slow switching. File: `i3/router/adaptive_
   compute.py`.
4. **Runnable HMAF agentic runtime** (HarmonyOS 6 HMAF parallel): upgrade
   `i3/huawei/hmaf_adapter.py` to a working end-to-end agent harness that
   simulates an HMAF caller invoking the registered capabilities. File:
   `i3/huawei/agentic_core_runtime.py`.
5. **Real-time translation demo** (AI Glasses parallel): a `/api/translate`
   endpoint using the adaptation-conditioned path (so a user's style
   profile is applied to the translation output too, not just the response
   language). File: `server/routes_translate.py`.

Each with a small doc and a CLI demo.

### Batch E (1 agent) — Public artifacts + arXiv-ready paper

**Alignment:** "AI product development or academic research" (JD).

- `docs/paper/build_latex.sh` + pandoc/latex build recipe.
- `docs/paper/paper.tex` output.
- `docs/blog/cross_attention_conditioning.md` (2500-word blog).
- `scripts/build_github_pages.sh` + site config.
- `docs/video/script.md` (6-minute backup-video shot list).
- arXiv submission instructions in `docs/paper/submission.md`.

---

## 3. Expected outcome per batch — what the interviewer actually asks

| Batch | Question the panel is likely to ask                    | What I can show them                            |
|:-----:|:-------------------------------------------------------|:------------------------------------------------|
| A     | "Did it actually work?"                                | Pre-registered ablation with effect sizes + CIs |
| B     | "Why do I believe the conditioning matters?"          | Causal tracing + linear-probe results           |
| C     | "What's novel here? What could we publish?"           | A benchmark spec the field doesn't yet have     |
| D     | "How does this connect to what Huawei is doing?"      | 5 runnable features each mapped to a 2025-2026 Huawei initiative |
| E     | "Is this public? Can I read it before we meet?"       | arXiv-formatted paper + blog + live URL         |

## 4. Explicit non-goals (what I am NOT adding)

- More YAML, more containers, more policy frameworks — already over-invested.
- New ML frameworks — 2026 toolchain batch already covered every logo.
- A new UI framework — vanilla JS design language set.
- Another optional Poetry group unless strictly required.

## 5. Explicit human-action items (user owns these)

- Run `make train-all` to fill `checkpoints/` (~2 h on laptop CPU).
- Record the 6-minute backup video (script is ready in Batch E).
- Submit to arXiv (requires user account + endorsement).
- Deploy live demo (requires hosting credentials).
- Send slides on 28 Apr to `matthew.riches@huawei.com` (verified email).

---

*Version history: v1 over-weighted individuals; v2 re-centred on the lab and
Huawei's 2025-2026 direction.*
