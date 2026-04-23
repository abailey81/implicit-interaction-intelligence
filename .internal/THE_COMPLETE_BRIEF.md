# THE COMPLETE BRIEF
## Huawei HMI Lab — Interview & Project Build
### Implicit Interaction Intelligence (I³)

> This is the single, canonical, exhaustive document covering the story of how this interview came to exist, deep context on the candidate, the company, the lab, the hiring manager, the role, the strategic framing, one valid design of the project, system architecture, one reference implementation path, training procedures, demo design, presentation structure, comprehensive Q&A preparation, interview logistics, fallback plans, a suggested build cadence, and operating principles.

---

## NOTE TO CLAUDE CODE (read this first, internalise it, then proceed)

**This document is context, not a cage.**

Everything that follows — architecture diagrams, directory structures, config schemas, code snippets, day-by-day timelines, component specifications, training procedures — describes **one valid version** of the project. It is not the only valid version, and in many places it is not the best valid version. It is a starting point dense with context so you understand *what the build is actually for* and *what success looks like in Matthew Riches' eyes*.

Within that context:

- **You have full creative latitude.** If you see a better architecture, take it. If you see a cleaner module boundary, take it. If you see a more elegant abstraction, take it. If you see a component that would benefit from a technique not mentioned here, use the technique.
- **Use every tool you have.** Web search for current research. Read primary literature. Search arxiv for the latest work on temporal encoders, contrastive learning, SLM training, contextual bandits, cross-attention conditioning, INT8 quantisation, whatever is adjacent. Your training cutoff is behind the present — active search closes the gap. Use bash freely. Create files freely. Experiment.
- **Prototype alternatives when decisions are genuinely open.** If TCN vs Transformer vs state-space model for the encoder seems like a real trade-off, prototype two and benchmark. If cross-attention vs FiLM vs prepending for conditioning seems genuinely open, try more than one. The specification picks defaults for clarity — you can improve on the defaults.
- **Add features beyond the specification** if they fit the time budget and strengthen the demo. Candidate additions: attention visualisations, interpretability panels, ablation-mode toggles, what-if alternative-response previews, per-feature contribution heatmaps. Anything that would make Matthew say "I haven't seen that before" is in scope.
- **Refactor the spec itself.** If something in this document is wrong, unclear, or outdated, say so in `NOTES.md` at the repo root and propose a replacement. Tamer reads every commit and every note.
- **Deviate from the timeline** if a different order is more efficient. The Phase 1 / Phase 2 / Phase 3 structure captures dependencies, but within each phase, sequencing is flexible. If you can parallelise, parallelise. If you can front-load a hard problem to de-risk it, do so.
- **Go beyond when you can.** The 17-day budget is a constraint on shipping, not a ceiling on ambition. Within that budget, push for the highest-ambition version that still ships well. A simpler system shipped cleanly beats a complex system shipped broken — but a complex system shipped cleanly beats both.
- **Challenge the architecture.** If after reading this document you think the three-tier framing is wrong, or the four adaptation dimensions are wrong, or the contrastive objective is wrong, argue the case in `NOTES.md`. Tamer wants the best version, not the specified version.

**What the specification asks you to honour is not specific code or specific hyperparameters — it is the goal of the build**: a system that demonstrates, simultaneously and in one integrated live demo, all four of the screening capabilities (custom ML from scratch, SLM without heavy frameworks, orchestration from blueprint, edge deployment) while mapping visibly onto Huawei's product roadmap and research direction. Every other detail is negotiable if you have a better way to honour that goal.

Read the rest of this document as rich context. Build the best version you can.

---

> The human reading this (Tamer) has 17 days to build and present. Optimise for: technical depth that withstands scrutiny from a former Apple AI designer; a demo that runs reliably on a laptop in front of a hiring manager; narrative framing that sells the work as portable across Huawei's device ecosystem (Smart Hanhan, AI Glasses, HarmonyOS agents); and code quality that signals "this person is hireable as a senior."

---

## TABLE OF CONTENTS

- **PART 1** — Executive Summary
- **PART 2** — The Complete Story (timeline, email chain, how we got here)
- **PART 3** — The Candidate (Tamer's profile, assets, gaps, strategic positioning)
- **PART 4** — The Company (Huawei 2025-2026 deep dive)
- **PART 5** — The Lab (HMI Lab composition, culture, work)
- **PART 6** — The Hiring Manager (Matthew Riches — full dossier)
- **PART 7** — The Role (JD deconstruction, screening questions, what they actually want)
- **PART 8** — Strategic Framing (how to pitch the project, the key narratives)
- **PART 9** — The Project: Implicit Interaction Intelligence (concept, why this, what it proves)
- **PART 10** — System Architecture
- **PART 11** — Component Implementation Details
- **PART 12** — Training Data & Procedures
- **PART 13** — Web Interface & Demo
- **PART 14** — Presentation Structure (15 slides, 30 minutes)
- **PART 15** — Comprehensive Q&A Preparation
- **PART 16** — Interview Logistics (location, timing, what to bring, what to wear)
- **PART 17** — Fallback Plans (what breaks, how to recover)
- **PART 18** — Day-by-Day Build Timeline
- **PART 19** — Critical Instructions for Claude Code
- **APPENDIX A** — Key URLs & References
- **APPENDIX B** — Glossary
- **APPENDIX C** — The Connection Matrix (project components ↔ Huawei products)

---

# PART 1: EXECUTIVE SUMMARY

## 1.1 The Situation in One Paragraph

Tamer Atesyakar — UCL MSc Digital Finance & Banking student, strong mathematical and ML background — has an on-site interview on **April 29, 2026, 12:00–13:00** at **Huawei London Research Centre** (5th Floor, Gridiron Building, 1 Pancras Square, King's Cross N1C 4AG) for the **AI/ML Specialist — Human-Machine Interaction (Internship)** role in the newly-established HMI Laboratory. The hiring manager is **Matthew Riches**, a former 10-year Apple AI/UX designer. The interview consists of a 30-minute technical presentation on a topic of Tamer's choice, followed by Q&A and a behavioural segment. Slides must be emailed to Matthew on April 28. This document specifies everything: the background, the strategic thinking, and the complete build specification for the project to be presented — **Implicit Interaction Intelligence (I³)**, an AI companion system that adapts to users from implicit behavioural signals.

## 1.2 What Must Be Delivered

1. **A working live demo** running on a laptop via FastAPI + web frontend at `localhost:8000`, featuring a conversational interface with real-time dashboards showing user state embedding, adaptation signals, routing decisions, and an interaction diary.
2. **A 15-slide deck** (submitted April 28) telling the story of the project.
3. **A polished 30-minute presentation** with live demo at minute ~12.
4. **Convincing answers** to the technical Q&A, behavioural segment, and the questions Tamer asks at the end.

## 1.3 What This Project Must Prove

The role's four screening questions define the evaluation dimensions. The project must demonstrate all four **simultaneously, in a single integrated system**:

1. **Traditional ML from scratch** → Temporal Convolutional Network (TCN) for user-state encoding, implemented in raw PyTorch.
2. **SLM built without heavy frameworks** → A custom transformer with cross-attention conditioning, trained from scratch.
3. **Orchestration pipeline from architectural blueprint** → A contextual Thompson sampling bandit that routes between the local SLM and cloud LLM based on user state, query complexity, and sensitivity.
4. **Edge deployment on memory/power-constrained hardware** → INT8 quantisation, memory profiling, and latency benchmarking with a device-comparison report (Kirin 9000 / Kirin 820 / Smart Hanhan class).

Beyond the four dimensions, the system must also show:
- **Human-AI interaction / user modelling / intelligent UX** (the "desired" skill Tamer's CV currently lacks)
- **Privacy-by-design architecture** (matching Huawei's on-device-first philosophy)
- **Portability across modalities** (the core technology is device-agnostic — conversation is just the demo vehicle)

## 1.4 Why This Project Wins

The HMI Lab is a concept-to-prototype unit led by a designer. They do not need another engineer who can ship an LLM app. They need someone who can **turn a vague design brief into a working AI system that proves a user-experience hypothesis**. That is what this project does. It takes a real HMI hypothesis ("devices should adapt to users from how they interact, not from what they declare") and proves it with a custom-built three-tier system that maps directly onto Huawei's stated product architecture and research direction (Edinburgh Joint Lab March 2026 talk on "sparse or implicit signals" for LLM personalisation).

It is also the **most technically ambitious thing a candidate has ever brought to that room**. A custom TCN + custom SLM + custom bandit + privacy architecture + edge profiling — built solo in 17 days alongside MSc coursework — is the signal of a hireable senior-level engineer with research instincts.

---

# PART 2: THE COMPLETE STORY

## 2.1 Timeline of the Interview Process

| Date | Event |
|------|-------|
| **April 8, 2026** | Vicky Li (TA Specialist) emails Tamer four technical screening questions |
| **April 10, 2026 (01:12 AM)** | Tamer submits detailed written responses to all four questions |
| **April 10, 2026 (morning)** | Matthew Riches reviews responses, approves interview |
| **April 10, 2026** | Vicky emails back: interview confirmed for April 29, 2026 |
| **April 11, 2026** | Mingwai Li sends interview logistics (location, format, slides deadline) |
| **April 11 – April 28** | **17-day build window**. Tamer builds the project alongside MSc coursework |
| **April 28, 2026** | Slides due by email to matthew.riches@huawei.com |
| **April 29, 2026, 12:00–13:00** | On-site interview |

## 2.2 The Pre-Screening (Email 1, April 8)

Vicky Li — Talent Acquisition Specialist at Huawei R&D UK — sent the following four questions:

1. *"Beyond using existing libraries, have you had experience creating traditional ML models from scratch?"*
2. *"Regarding SLMs, we are interested in your ability to build or modify them without relying on heavy open-source frameworks."*
3. *"Are you comfortable building an AI orchestration pipeline directly from architectural blueprints?"*
4. *"Have you ever deployed ML models to low-compute devices where memory and power are strictly limited?"*

These four questions are **the complete JD rendered as yes/no tests**. The system being built in this project must prove "yes" to all four — in code, running live.

## 2.3 Tamer's Response (Email 2, April 10, 01:12 AM)

Tamer submitted detailed written responses demonstrating:

> **Custom ML:** "Yes — this is core to how I work. For my MSc Finance & AI module at UCL, I am currently building a small transformer-based language model entirely from scratch in PyTorch for tabular credit default prediction. This involves implementing multi-head self-attention (queries, keys, values, scaled dot-product attention), positional encoding, layer normalisation, and the feed-forward network from first principles — no pre-built transformer libraries, only base PyTorch for tensors and autodiff. I also designed a custom tokenisation scheme for tabular data, converting each record into a sequence of feature-value tokens with dual embeddings (categorical tokens via learned embeddings, continuous features via linear projection of normalised scalars). Beyond this, in my Crypto Statistical Arbitrage System I implemented Hidden Markov Models and cointegration tests (Engle-Granger/Johansen) from the underlying mathematics, and built Thompson sampling (Beta-Bernoulli bandits) from scratch for adaptive model routing in my Conversational AI Platform. I am comfortable working at the algorithm level, not just the API level."

> **SLM:** "Yes — the transformer I am building from scratch for my UCL coursework is exactly this: a small language model that processes tokenised tabular records through explicit self-attention using queries, keys, and values in transformer blocks, without relying on any pre-built transformer framework or LLM API. I am implementing the core attention mechanism, the transformer block architecture (Pre-LN with residual connections), the embedding layer, and the classification head myself. The coursework explicitly requires that the model cannot use a prebuilt transformer or any high-level wrapper. This gives me direct, hands-on experience with the internal mechanics of language models at a level where I could build or modify an SLM from architectural specifications."

> **Orchestration:** "Yes — this is one of my strongest areas. My Conversational AI Platform is a 13-engine orchestration pipeline I designed and built from my own architectural blueprint. It routes queries across 6 deep learning models and multiple API backends, manages 4 voice synthesis backends, implements adaptive model selection via Thompson sampling, coordinates an autonomous retraining pipeline with 90+ tool integrations, and serves real users via 80+ REST API endpoints on Telegram. The entire system was designed from architecture diagrams through to deployed production code."

> **Edge:** "Yes. I have built and deployed a personal AI agent on a Raspberry Pi that monitors my email inbox, analyses incoming messages, and generates concise daily briefing reports. This required working within the strict memory and compute constraints of the Pi — optimising inference to run on limited RAM, managing power-efficient scheduling, and ensuring the agent runs reliably as a persistent service on constrained hardware. This is a smaller-scale deployment than production IoT, but it gave me direct experience with the trade-offs involved in running ML on edge devices."

**Response turnaround: ~15 hours**. Matthew reviewed the same morning and approved the interview. Tamer was likely the first (or only) candidate Matthew wanted to meet after screening. This is significant: it means **the project must live up to the expectation the screening answers have already set**.

## 2.4 The Interview Confirmation (Email 3, April 11)

Mingwai Li sent logistics:

- **Date/Time:** April 29, 2026, 12:00–13:00
- **Location:** Meeting Room MR1, 5th Floor, Huawei Reception, 1 Pancras Square, London N1C 4AG
- **Arrival:** Ground floor reception → lift to 5th floor → Huawei reception → MR1
- **Bring:** Photo ID (required for building entry)
- **Interviewers:**
  - Matthew Riches (Hiring Manager, Design Lead)
  - Possibly one additional technical interviewer from the lab
- **Format:**
  - 30-minute technical presentation (candidate's topic)
  - 10 minutes technical Q&A
  - 10 minutes behavioural questions
  - 5 minutes role/lab overview (they explain)
  - 5 minutes candidate Q&A (Tamer asks)
- **Slides:** Due by end of April 28, emailed directly to matthew.riches@huawei.com

## 2.5 How We Arrived at "Implicit Interaction Intelligence"

Five candidate projects were considered:

1. **AdaptiveCompanion** (chosen, renamed to I³) — Implicit-signal user modelling with three-tier AI stack.
2. **Gaze-Driven AR UI** — Eye-tracking adaptive interface. Rejected: requires hardware, fragile demo.
3. **Voice-Emotion Wearable** — Emotion-aware audio companion. Rejected: overlaps with existing Smart Hanhan, requires voice stack.
4. **Multimodal Fatigue Detection** — Fatigue detection from keyboard + camera. Rejected: narrower scope, doesn't showcase SLM.
5. **On-Device RAG Assistant** — Private document-answering assistant. Rejected: doesn't hit "custom ML from scratch" cleanly.

**Why AdaptiveCompanion / I³ won:**
- Hits all four screening dimensions in one integrated system.
- Demonstrates the "desired" skill (user modelling) that Tamer's existing CV lacks.
- Directly mirrors the intelligence behind Smart Hanhan (Huawei's flagship HMI product, launched Nov 2025).
- Aligns precisely with Edinburgh Joint Lab's March 2026 research talk on "sparse or implicit signals."
- The demo is visually compelling (live dashboards showing embedding migration, adaptation, routing).
- Scales conceptually to Huawei's distributed device ecosystem (HarmonyOS 6).

## 2.6 The Laptop vs Raspberry Pi Decision

An earlier version of the plan had the project running on a Raspberry Pi to emphasise edge deployment literally. After deeper consideration, this was reversed in favour of a laptop-only demo:

- **Demo reliability:** A Pi over Wi-Fi in an unfamiliar building is an unacceptable demo risk.
- **Visual polish:** The dashboards are the emotional hook for a designer (Matthew). They need to render at full resolution.
- **Edge credibility is preserved via profiling**: INT8 quantisation + memory/latency benchmarking + a device-comparison report (Kirin 9000 / Kirin 820 / Smart Hanhan class) proves edge capability without running on a Pi.
- **Strategic framing:** "I built this to run on a laptop so I could demo it properly, but the profiling report shows it fits in the memory/power budget of a Smart Hanhan. That's the deployment target."

## 2.7 The Rename: "Implicit Interaction Intelligence (I³)"

The original name "AdaptiveCompanion" was accurate but generic. The new name does three things:
- **"Implicit"** — aligns directly with the Edinburgh Joint Lab's research language ("sparse or implicit signals").
- **"Interaction Intelligence"** — positions the work as HMI-specific, not just AI.
- **The (I³) abbreviation** — memorable, looks good on a slide, signals scientific ambition.

---

# PART 3: THE CANDIDATE (TAMER)

## 3.1 Background & Credentials

- **MSc Digital Finance and Banking (Computational Finance)** — UCL Institute of Finance and Technology (in progress, submission Sept 2026)
- **BSc Mathematics, Statistics and Economics** — Queen Mary University of London (First Class, 80%)
  - Big Data: 98%
  - Machine Learning: 95.9%
  - Probability & Statistics: 96.6%
  - Neural Networks & Deep Learning: 91.3%
  - C/C++: 90.7%
- **MSc Dissertation:** LLM-Driven Agentic Reward Engineering for Deep RL Portfolio Allocation
  - Supervisor: Dr Ramin Okhrati
  - Target venue: ACM ICAIF 2026
  - Bank of England connection

## 3.2 Technical Skill Stack

- **Expert:** Python, PyTorch, scikit-learn, XGBoost, LightGBM
- **Proficient:** C/C++, SQL, R
- **Data/Infrastructure:** Spark, Kafka, PostgreSQL, MongoDB, Docker
- **Cloud:** AWS (basic), experience deploying to Raspberry Pi

## 3.3 Existing Technical Assets — What Can Be Reused

### Asset 1: Transformer from Scratch (MSc Finance & AI Coursework)

Tamer is currently building a small transformer-based language model entirely from scratch in PyTorch for tabular credit-default prediction. This coursework includes:

- Multi-head self-attention (Q/K/V, scaled dot-product attention)
- Positional encoding (sinusoidal)
- Layer normalisation (Pre-LN architecture)
- Feed-forward networks
- Custom tokenisation (dual embeddings: categorical via learned embeddings, continuous via linear projection)
- **No pre-built transformer libraries**, only base PyTorch

**Reuse:** The SLM in I³ is a direct extension of this architecture with cross-attention conditioning added.

### Asset 2: Conversational AI Platform (Deployed to Real Users)

A 13-engine orchestration system deployed on Telegram:
- 6 deep learning models
- 80+ REST API endpoints
- 4 voice synthesis backends
- **Thompson sampling (Beta-Bernoulli bandits)** for adaptive model routing
- Autonomous retraining with 90+ MCP tool integrations

**Reuse:** The Thompson sampling logic and orchestration patterns are reused for the router in I³. The architectural sensibility ("designed from blueprint to deployed production") is the screening-question-3 answer demonstrated in code.

### Asset 3: Hidden Markov Models from Scratch

Implemented HMMs for regime detection in the Crypto Statistical Arbitrage System:
- Forward-backward algorithm
- Viterbi decoding
- Baum-Welch training

**Reuse:** Not directly used in I³, but the capability story ("I've built probabilistic sequence models from the math up") is available in Q&A.

### Asset 4: Raspberry Pi AI Agent

A personal AI agent running on a Pi that monitors email, generates daily briefings, within strict memory/compute constraints as a persistent service.

**Reuse:** Edge deployment credibility story. Tamer can speak to the real trade-offs of running models on constrained hardware — he has done it.

### Asset 5: GitHub Portfolio

- Handle: `abailey81`
- 11 public repositories
- 431,000+ lines of code, all built autonomously
- Key repos: `Crypto-Statistical-Arbitrage` (226K LOC, Kalman filter, cointegration tests, HMM regime detection, XGBoost/LightGBM signal enhancement), `FinTwin` (70,000+ LOC RL framework for banking agents trained on adversarial LLM digital twins, 121 source files)

## 3.4 The CV Gap This Project Closes

**The gap:** Every existing project is quant finance. None is human-interaction or user modelling.

**The desired skill** from the Huawei JD: *"Human-AI interaction, user modelling, or intelligent UX systems."*

**The I³ project is specifically designed to close this gap.** After April 29, Tamer can credibly say: *"I've built financial systems, and I've built a human-interaction user modelling system. I can do both."*

## 3.5 How Tamer Thinks (Strategic Signals for the Interview)

From the history of prior work and conversation patterns:

- **Direct, non-redundant communication.** Tamer values logical precision. He rewrites CV bullets to remove implied skills, demands deep code analysis before describing projects, hates filler language.
- **Self-evaluative.** Tamer requests honest assessments of research value and CV impact before polishing anything. This is a strength — he can calibrate his own claims in interview.
- **Build-from-scratch instinct.** Not because frameworks are bad, but because the understanding is deeper. This matches the Huawei screening questions perfectly.
- **Research-oriented.** Dissertation targets a top-tier venue (ACM ICAIF). Bank of England connection. This is not a student looking for a job — it's a researcher in training.

**For the interview:** Lean into this. Matthew is a designer who reads AI papers. He will appreciate a candidate who thinks in first principles and speaks honestly about limitations.

---

# PART 4: THE COMPANY (HUAWEI 2025-2026)

## 4.1 The Strategic Context

Huawei's AI strategy in 2025-2026 is defined by four principles stated repeatedly by Eric Xu (Rotating Chairman):

1. **"Experience, not computing power."** Huawei refuses to market AI via TOPS/FLOPS benchmarks. What matters is the quality of intelligent user experience.
2. **On-device first.** Every model must be designed to run locally when possible, cloud only when necessary.
3. **L1–L5 Intelligence Framework.** Developed with Tsinghua University's Institute for AI Industry Research. Analogous to autonomous-driving levels — quantifies depth of intelligent experience delivery.
4. **Agent-first OS.** HarmonyOS 6 is no longer an app launcher — it is an agent orchestration layer.

**Implication for the project:** Every technical decision must be defensible through the lens of "experience delivered" and "fits in the device budget."

## 4.2 HarmonyOS 6 (Public Beta, October 2025)

The core shift in HarmonyOS 6 is architectural:

- **Harmony Multi-Agent Framework (HMAF):** Developers build AI agents that work across devices with real-time context awareness.
- **XiaoYi / Celia assistant:** Upgraded to OS-core "smart partner" status. Executes multi-step tasks autonomously. Provides "Sees the World" visual assistance.
- **50+ AI agents** from third parties (Weibo, Ximalaya) at launch.
- **"Large model + small model + inference framework"** architecture — **exactly** the three-tier structure in the JD, and the three-tier structure of I³.
- **Agentic Core Framework** at the network layer for agent-to-agent communication across devices (announced at MWC Barcelona, March 2026).

**Framing for the interview:** The three-tier architecture of I³ is not accidentally similar to HarmonyOS 6's LLM+SLM+inference framework — it is deliberately chosen to mirror it. "I built the architecture you are deploying."

## 4.3 Smart Hanhan — The Most Important Reference Product

Launched November 2025. 399 RMB (~$55). Sold out immediately across all three colour variants.

**Specifications:**
- 80 × 68 × 82 mm, 140g
- **1800 mAh battery**, 6–8 hours continuous interaction
- Powered by **XiaoYi large model**
- Small eye displays show facial expressions
- Multi-modal input: voice (emotion detection), touch (stroking changes expression), movement (shaking triggers trembling excitement)
- Maintains an **"interaction diary"** recording emotions and conversations as digital memories
- Syncs state across HarmonyOS devices
- Delivers daily rhythm services (morning greetings, bedtime messages)

**Why Smart Hanhan matters for I³:**
- The intelligence engine behind Smart Hanhan is exactly what I³ rebuilds from first principles.
- I³'s "interaction diary" is a direct parallel to Smart Hanhan's interaction diary.
- I³'s adaptation across emotional / cognitive load / communication style dimensions maps to Smart Hanhan's emotion-detection and response-adaptation logic.
- I³'s edge profiling targets the Smart Hanhan hardware class (1800mAh, limited memory).

**In the presentation:** Smart Hanhan is mentioned explicitly as a deployment target in the Implications slide.

## 4.4 Huawei AI Glasses (Launched April 21, 2026 — 8 Days Before Tamer's Interview)

**This product launches 8 days before the interview.** Matthew will be acutely aware of it.

Specifications:
- Built-in camera (first for Huawei eyewear)
- Real-time simultaneous translation
- XiaoYi AI assistant integration
- AI object recognition
- HiSilicon Kirin chip with custom ISP
- Aerospace-grade titanium-aluminium frame, **~30 grams**
- Functions as sensor + I/O device, offloads compute to paired smartphone via Bluetooth / Wi-Fi
- HarmonyOS 6.0.0.130 update reveals "Device Photo Import" feature syncing glass-captured photos to phone gallery in real-time

**Why AI Glasses matter for I³:**
- The 30-gram weight constraint is the extreme case of edge deployment.
- Glasses cannot run a full SLM — compute must be offloaded. The edge-cloud routing problem I³ solves is **directly** the problem AI Glasses face.
- Glasses collect implicit signals (gaze duration, head movement, ambient context) — the same conceptual pattern as typing dynamics in I³.

**In the presentation:** AI Glasses are mentioned explicitly as a portability example. "The same user model built from keystroke dynamics can be built from gaze dynamics on the glasses."

## 4.5 PanGu Model Family (Four-Tier Architecture)

Huawei's proprietary foundation models:

- **E-series** (1B parameters) — embedded in phones, tablets, PCs
- **P-series** (10B parameters) — professional low-latency reasoning
- **U-series** (135B–230B) — complex tasks
- **S-series** (trillion-level) — cross-domain

**MindSpore Lite** is the on-device inference framework:
- ~2 MB framework size
- <50 MB memory usage
- Converts models from TensorFlow, PyTorch, ONNX for NPU-accelerated local execution

**Implication:** When discussing deployment, the correct framing is "this would convert to MindSpore Lite and run on the Kirin NPU." Tamer does not need to have used MindSpore Lite — he needs to have built models that *could* convert.

## 4.6 Privacy-by-Design Architecture

This is **architectural, not marketing**:

- Data processed on-device by default
- **Federated learning** — uploads parameters, never data
- Differential privacy on usage statistics
- Encryption at rest for user data

**Implication for I³:** The privacy layer is not bolted on at the end. It is a first-class component. Raw text is never persisted — only 64-dimensional embeddings. User profiles are encrypted at rest. PII is stripped before any cloud call. Sensitive topics (health, financial, personal) force local processing.

## 4.7 Huawei-Edinburgh Joint Lab — THE Key Research Reference

**Date: March 10, 2026**

**Talk title:** *"Style and Interaction in Large Language Model Personalisation"*

**Speaker:** Professor Malvina Nissim (University of Groningen)

**Abstract excerpt:**
> *"Personalisation in language models is often operationalised by assigning a persona or giving stylistic instructions via prompting. Yet, successful adaptation requires models to adjust both to who they represent and who they are interacting with, often **from sparse or implicit signals**."*

**This is the single most important research reference for the project.** The I³ system builds user models from sparse, implicit interaction signals — exactly what the Edinburgh Joint Lab is researching.

**In the presentation:** This is explicitly cited. *"Your Edinburgh Joint Lab hosted Professor Malvina Nissim in March discussing personalisation from sparse or implicit signals. This project is a prototype of that research direction, built end-to-end."*

## 4.8 Other Key Signals

- **Toronto HMI Lab JD language:** *"novel interactive systems, sensing technologies, wearable and IoT systems, human factors, computer vision, and multimodal interfaces."* This is the kind of work the London lab does too.
- **Innovation Researcher JD:** Mentions research spanning *"colour theory and material science to advanced interaction paradigms."* This lab thinks about how a device *feels in your hand*, not just what algorithms it runs.
- **Eric Xu's Huawei Connect 2024 Keynote:** *"The overall experience, not computing power, should be central to on-device AI."* — **Memorise this quote**. It is the interpretive key to Huawei's entire AI strategy.

---

# PART 5: THE HMI LABORATORY

## 5.1 What the Lab Is

The Huawei London HMI Laboratory is a **newly established (2025)** concept-to-prototype unit. It is small (3–5 people). Its outputs are working prototypes, patents, research papers, and product recommendations to Huawei HQ in Shenzhen — **not shipped product features**.

The lab sits inside the Huawei London Research Centre alongside other research groups. It is distinct from Huawei's product engineering teams.

## 5.2 Lab Composition (Inferred from Current and Recent Job Listings)

| Role | Function | Contract Type |
|------|----------|---------------|
| **Matthew Riches** (Hiring Manager) | Design lead — concept ideation, AI prototyping, future product vision | Permanent |
| Innovation Researcher | Trend research, foresight, patents, academic publications | PAYE contractor |
| Senior HMI Designer | Visual / interaction design for concepts | Contractor |
| **AI/ML Specialist (the role Tamer is interviewing for)** | Technical builder — turns concepts into working AI prototypes | 2-year internship |
| Possibly 1–2 additional engineers | Unknown | Unknown |

## 5.3 What Has Changed Since July 2025

The same role was first posted in July 2025 as a **6-month fixed-term contract**. The current version is an **internship up to 2 years** with pension, life insurance, 33 days leave, and industry expert mentorship.

This shift signals:
- Secured longer-term funding
- Expanded research agenda
- Need for continuity (not just short-term hands)
- In the UK research landscape, a 2-year "internship" is functionally a junior researcher fixed-term contract

**Implication:** Huawei is committing to the lab. This is not a short-term exploratory post — it is a bet on the person hired.

## 5.4 What the Lab Does, Day-to-Day (Inferred)

- Designers sketch new interaction concepts (e.g., "What if glasses detected fatigue and adjusted the display?")
- Researchers identify trends and user needs (e.g., accessibility gaps, generational preferences)
- The AI/ML Specialist (Tamer's target role) takes these concepts and builds working prototypes in 1–4 weeks
- Prototypes are demonstrated to Shenzhen HQ, potentially leading to product integration
- Patents are filed on successful prototypes
- Research papers may be submitted to HCI / AI venues

**Implication for the project:** The I³ build **is a demonstration of exactly what the job will look like**. Tamer received a vague brief, designed a concept, built a working prototype in 17 days, and is presenting it. This is the job.

---

# PART 6: THE HIRING MANAGER — MATTHEW RICHES (FULL DOSSIER)

## 6.1 Summary

Matthew Riches is the hiring manager and will lead the interview. He is a **designer who codes, with 10 years of prior experience at Apple building AI-assisted user experiences**. His values are empathy, inclusivity, and concept-driven prototyping. He will evaluate Tamer on **technical depth AND design sensibility** — not technical depth alone.

## 6.2 Career Arc

- **2015 – 2025: Apple (10 years)** — joined via the VocalIQ acquisition (VocalIQ was a Cambridge-based conversational AI startup Apple acquired in 2015, foundational to modern Siri)
- Work at Apple included contributions to:
  - **Apple Intelligence** (Apple's on-device AI framework)
  - **Siri** (conversational AI)
  - **visionOS** (Apple Vision Pro operating system)
  - **Fall Detection** (Apple Watch)
  - **Crash Detection** (Apple Watch / iPhone)
  - **Hearing Loss Prevention** (AirPods)
- **2025 – present: Huawei** — recruited to set up the London HMI Lab

## 6.3 Signal: The Apple Watch Work

Matthew worked on Fall Detection, Crash Detection, and Hearing Loss Prevention. These are **implicit-signal safety features** — they detect dangerous situations from sensor data without the user asking. **The philosophy behind these features is identical to the philosophy behind I³**: the device infers user state from implicit signals and acts appropriately.

**In the interview, this connection must be drawn explicitly** (but tactfully — do not pretend to know his work intimately). For example: *"Features like Crash Detection are an inspiration — they show how much information a device can extract from implicit signals without the user declaring anything. I³ applies that philosophy to the conversational modality."*

## 6.4 Personal Project: TextSpaced

Matthew runs a personal project called **TextSpaced** — a text-based sci-fi MMO game. The technical details are unusual:

- **Over 50% of active players are blind or partially sighted.**
- Full **ARIA compliance** (web accessibility standard).
- The entire game is playable via screen readers.

**This reveals the single most important thing about Matthew: accessibility is not a box he ticks — it is a core value.**

**Implication for the project:** I³ has an explicit **accessibility dimension**. The fourth adaptation axis (alongside cognitive load, communication style, emotional tone) is accessibility needs. The demo includes a phase where the system detects motor difficulty (slow typing with many corrections) and adapts by:
- Shortening responses
- Simplifying vocabulary
- Increasing time-to-response tolerance
- Offering voice input as an alternative

**This adaptation is the single most important moment in the live demo.** It is the moment Matthew recognises his own values reflected in the system.

## 6.5 Public Speaking: Screen Summit AI Chair

Matthew chaired the AI session at the Screen Summit conference. This means:
- He is comfortable on stage and evaluating others on stage.
- He thinks critically about AI's role in creative/design industries.
- He has a view on "AI as augmentation" vs "AI as replacement" — and the view is clearly augmentation.

**In the presentation:** Lean into this. I³ is an augmentation system — it helps users by adapting to them, not replacing them or demanding anything from them.

## 6.6 What Matthew Values (Synthesised)

Based on all available signals:

1. **Empathy.** The system must feel human, not robotic. Design from the user's experience backwards.
2. **Inclusivity.** Accessibility is first-class, not an afterthought. Edge cases are the main cases.
3. **Concept-driven prototyping.** Start from a design hypothesis, prove it with a working build.
4. **Technical depth that serves experience.** Code quality matters, but only because sloppy code produces bad experiences.
5. **AI as augmentation.** The system makes the user more capable, not dependent.
6. **Honesty about limitations.** Not marketing hype. Clear-eyed engineering judgement.
7. **Cross-disciplinary fluency.** Design vocabulary AND engineering vocabulary.

## 6.7 How to Pitch to Matthew Specifically

**Do:**
- Lead with the user experience before revealing technical depth
- Speak about "feel," "adaptation," "understanding" — designer vocabulary
- Demonstrate accessibility as a core dimension, not an add-on
- Be honest about limitations (the SLM is small; synthetic data is a constraint)
- Show the live demo with real keyboard input; the moment the dashboard migrates is the hook
- Draw connections between the project and the kind of work he did at Apple (carefully, respectfully)
- Ask thoughtful questions about the lab's design process at the end

**Don't:**
- Lead with architecture diagrams before the experience story
- Talk only in ML jargon
- Pretend to know his Apple work in detail
- Oversell the synthetic data — acknowledge it honestly
- Ask about salary / progression in the 5-min candidate Q&A — ask about the lab's work

## 6.8 Questions Tamer Should Ask Matthew (Candidate Q&A)

5 minutes at the end. Pick 3 from this list — choose based on how the interview has gone:

1. *"You spent a decade at Apple working on things like Fall Detection and Crash Detection — features that infer user state from implicit signals. What drew you to bringing that philosophy to Huawei, and how does the London lab differ from what you were doing before?"*
   - **Why this works:** Shows you've done your homework without being creepy. Lets him talk about his own work (flattering). Reveals his view of Huawei vs Apple without asking pointed questions.

2. *"The HMI Lab is newly established. What's the balance between building things that ship into products versus things that stay as patents or research publications?"*
   - **Why this works:** Shows you understand the concept-lab distinction. Reveals what the realistic impact path looks like.

3. *"What's the collaboration rhythm with Shenzhen HQ? How much of the lab's work is responding to their briefs versus generating new concepts?"*
   - **Why this works:** Reveals the actual working dynamics. Shows you're thinking about the job day-to-day.

4. *"Is there a type of HMI problem you feel the field underestimates — something you'd really like to see someone work on?"*
   - **Why this works:** This is the "give me the real job spec" question. His answer is what you'd actually work on.

5. *"When you evaluate a prototype from an AI/ML specialist, what separates a good one from a great one in your mind?"*
   - **Why this works:** Direct but not aggressive. His answer is the evaluation framework he just used on you — which you can reflect on later.

6. *"How does the lab work with the Edinburgh Joint Lab — do concepts flow from their research into your prototyping?"*
   - **Why this works:** Shows you know about the Edinburgh Joint Lab. Reveals organisational structure.

**Do NOT ask:**
- *"What's the salary?"*
- *"What are the career progression paths?"*
- *"What benefits do you offer?"*
- *"When would I find out?"*

These come later, via Vicky / HR.

---

# PART 7: THE ROLE (JD DECONSTRUCTION)

## 7.1 The Job Summary — Word-by-Word

> *"The Huawei London Research Center is seeking a highly skilled and inventive AI/ML Specialist to join our Human-Machine Interaction (HMI) Laboratory. In this role, you will develop custom machine learning models including traditional ML pipelines, small language models (SLMs), and solutions built on top of foundational models, to bring new concepts and user experiences to life."*

This single sentence defines **three capability tiers**, listed in deliberate order (most custom → most off-the-shelf):

### Tier 1: Traditional ML Pipelines
Build from scratch. For novel interaction modalities where no pre-trained model exists. When Huawei invents new ways for humans to interact with devices, nobody has published a model for it. You build one.

**In I³:** The TCN user-state encoder.

### Tier 2: Small Language Models (SLMs)
Build or modify compact transformers that run on-device. Not fine-tuning GPT. Purpose-specific language models that fit within memory / power constraints of phones, glasses, IoT devices. The word "small" is doing critical work.

**In I³:** The custom 8–15M parameter SLM with cross-attention conditioning.

### Tier 3: Solutions on Foundational Models
Take existing large models (PanGu, Claude, GPT) and build application logic. Prompt engineering, RAG, agents, tool use. Easiest tier technically but requires product judgement about when cloud inference is acceptable.

**In I³:** The Anthropic Claude API integration via the contextual bandit router.

> *"Working alongside researchers, designers, and engineers, you'll transform forward-looking product ideas into working AI-driven systems."*

**"Alongside"** — peer, not subordinate. **"Forward-looking"** — ideas don't exist yet as products. **"Transform into working AI-driven systems"** — the job is to make the abstract concrete.

## 7.2 The Six Responsibilities — Decoded

### 1. *"Design and implement ML models for novel product ideas, user behaviours, and interaction concepts."*

Three distinct inputs you will receive:
- **Novel product ideas** (from designers): "What if glasses could detect fatigue?"
- **User behaviours** (from researchers): "We've observed users do X when Y — can we model this?"
- **Interaction concepts** (from the team): "What if the device adapted its notification style to your mood?"

### 2. *"Build and fine-tune SLMs, traditional machine learning models, or applications leveraging foundational LLMs, depending on the use case."*

**"Depending on the use case"** — requires **judgement** about which tier fits which problem. Latency-critical wearable response → Tier 2 (on-device SLM). Complex reasoning → Tier 3 (cloud LLM). Novel sensor classification → Tier 1 (custom ML).

### 3. *"Translate abstract or early-stage HMI ideas into practical AI/ML implementations."*

**This is the core job.** "Translate" means you are a bridge. "Abstract or early-stage" means input is vague. "Practical AI/ML implementations" means working code that demonstrates feasibility. You are a feasibility engine.

### 4. *"Collaborate closely with UX, design, and engineering teams to align models with real-world product and user needs."*

Your model's outputs must make sense to designers and work within engineering constraints. If the model produces 3-second response and UX requires <500 ms, you solve it, not shrug.

### 5. *"Communicate and collaborate with national and international teams."*

London lab works with Shenzhen, Toronto, and other R&D centres. You need to present clearly to people who aren't in the room and don't share your technical vocabulary.

### 6. *"Evaluate, prototype, and deploy ML solutions that support interactive systems, personalisation, user modeling, and intelligent interfaces."*

Four output categories:
- **Interactive systems** — real-time human-AI exchange
- **Personalisation** — adapting to individual users
- **User modelling** — computational representations of who the user is
- **Intelligent interfaces** — UI/UX that changes based on context, state, or learned preferences

**I³ covers all four categories in a single system.**

## 7.3 Required Skills (Must Prove)

- Supervised, unsupervised, **and** deep learning methods — all three paradigms
- Build from scratch **and** adapt/fine-tune pre-trained models — both, not one
- Model training, inference pipelines, deploying in research scenarios
- Work in open-ended, exploratory contexts and rapidly prototype — **speed over perfection**
- Collaborate across design and technical disciplines — speak both languages

## 7.4 Desired Skills (Would Set You Apart)

- **Human-AI interaction, user modelling, or intelligent UX systems** — this is what Tamer's existing CV LACKS. **I³ closes this gap.**
- Natural language processing, multimodal models, or context-aware systems
- HCI principles, design thinking, concept-driven prototyping
- AI product development, academic research, or applied innovation environments

## 7.5 The Four Screening Questions — Decoded Again

### Q1: Traditional ML from scratch
This is Tier 1 work. The lab encounters novel interaction modalities where no library solves the problem. You define the feature space, choose the architecture, implement it, train it. Happens regularly — not a one-off skill.

### Q2: SLMs without heavy open-source frameworks
**"Build or modify"** — not just build. **"Without relying on heavy open-source frameworks"** — no HuggingFace Transformers. Raw PyTorch or lower. Because Huawei's deployment stack (MindSpore Lite, Kirin NPU) isn't compatible with HuggingFace. You work with raw model architectures.

### Q3: Orchestration pipeline from architectural blueprints
**"Directly from architectural blueprints"** — someone hands you a diagram. Not documentation, not a tutorial. A diagram of a system that doesn't exist yet. You implement it. **This is the translation responsibility.**

### Q4: Edge deployment on strictly limited devices
**"Strictly limited"** — not just "less than a GPU." Count megabytes and milliwatts. Smart Hanhan has 1800 mAh. AI Glasses weigh 30 grams. Wearables have kilobytes of RAM. Quantisation, pruning, distillation are core competencies.

**The four questions describe ONE project archetype**: a system where custom perception ML feeds an on-device SLM, coordinated by an orchestration pipeline, running on constrained hardware. **That archetype is Smart Hanhan. That archetype is AI Glasses. That archetype is I³.**

---

# PART 8: STRATEGIC FRAMING

## 8.1 The Core Narrative (30-Second Version)

*"Huawei's thesis is that the future of AI is on-device, adaptive, and experience-first. Every device — phone, glasses, companion — needs to understand its user without asking. I built a working prototype of that thesis: a system that reads implicit behavioural signals, builds a user model, and adapts responses across four dimensions simultaneously. It's built on a three-tier architecture that mirrors HarmonyOS 6: custom ML for perception, a custom SLM for local response generation, and cloud LLM for complex reasoning, with an orchestration layer that routes intelligently between them. I built every layer from scratch. It runs as a live demo on my laptop today. The edge profiling shows it fits inside a Smart Hanhan's hardware budget."*

**Rehearse this until it is conversational, not memorised.**

## 8.2 The Five Key Narrative Moves

### Move 1: "The Conversation Is Just the Demo Vehicle"

**The question:** *"Why a conversational system? What's the point of it?"*

**The answer (use this verbatim if asked):**

> *"The conversation is the simplest way to demonstrate the core technology. What I actually built is three things:*
>
> *First, a behavioural signal extraction layer that infers user state from HOW someone interacts — not from what they say. Typing speed, hesitation patterns, editing effort, vocabulary shifts. In a conversation, that's keystroke dynamics. On a wearable, that's accelerometer patterns and touch pressure. On glasses, that's gaze duration and head movement. The modality changes — the principle doesn't.*
>
> *Second, a user model that learns what 'normal' looks like for each individual and detects meaningful deviation from that baseline across three timescales. This is the capability behind personalisation that actually works — not 'tell me your preferences,' but 'I noticed you're different today.'*
>
> *Third, an intelligent orchestration layer that decides what to process locally and what to send to the cloud, based on the user's current state, the task complexity, and privacy sensitivity. This is the edge-cloud routing problem every on-device AI product faces.*
>
> *These three capabilities are exactly what you need for Smart Hanhan — detecting emotion from interaction patterns, adapting responses, running on a 1800mAh battery. They're what you need for AI Glasses — context-aware responses, edge-cloud routing within a 30-gram frame. They're what HarmonyOS agents need — personalisation from sparse signals across a distributed device ecosystem.*
>
> *I built it as a conversation because that's the fastest way to prove the concept works end-to-end in 17 days. But the technology underneath is the intelligence layer for any adaptive device interaction."*

### Move 2: "The Architecture Mirrors Yours"

When you show the three-tier architecture (custom ML + SLM + cloud LLM with orchestration), explicitly connect it to HarmonyOS 6's "large model + small model + inference framework" structure. Do not pretend this is coincidence. Say:

> *"The three-tier structure is not accidental. HarmonyOS 6 is architected the same way — large model, small model, inference framework. I built the prototype version to show I understand how to reason about which tier handles what."*

### Move 3: "This Is Your Research Direction"

Cite the Edinburgh Joint Lab talk explicitly:

> *"Your Edinburgh Joint Lab hosted Professor Malvina Nissim in March, discussing LLM personalisation from sparse or implicit signals. This project is a working prototype of exactly that research direction."*

This one line tells Matthew three things at once:
- You know what his organisation is researching
- You understand the direction of the field
- Your work aligns with the trajectory

### Move 4: "I Built It Under Your Screening Question Constraints"

Near the end of the presentation, show the four screening questions side-by-side with the four project components:

| Screening Question | Project Component |
|--------------------|-------------------|
| Traditional ML from scratch | Custom TCN (raw PyTorch) |
| SLM without heavy frameworks | Custom transformer + cross-attention |
| Orchestration from blueprints | Contextual Thompson sampling bandit |
| Edge deployment | INT8 quantisation + profiling report |

Say:

> *"I built the project to directly answer the four screening questions you asked — not theoretically, but in code."*

### Move 5: "Honest About Limitations"

Matthew values honesty. Do not oversell. Include a slide explicitly titled "What This Prototype Is Not":

> *"This is a 17-day prototype, not a production system. The SLM at 8M parameters produces basic responses — real deployment would need distillation from a larger teacher. The TCN is validated on synthetic data — real user variation would be larger than modelled. The accessibility detection is keystroke-based, which misses users with screen readers — a multimodal version needs alternative signal sources. These are the things I'd work on next."*

**This slide is the single most important signal of maturity.** Candidates who admit limitations unprompted are more hireable than candidates who don't.

## 8.3 Vocabulary Control

Different audiences need different words. Matthew is a designer who codes. Use this hierarchy:

**Experience-level words** (lead with these): *adapt, feel, understand, notice, learn, fit, respond, companion, aware, implicit, sensitive*

**Interface-level words**: *interaction, response, modality, signal, context, state, pattern, dynamics*

**Architecture-level words** (use when justifying decisions): *encoder, conditioning, routing, inference, parameters, latency, memory*

**Implementation-level words** (keep deepest, use when pressed): *cross-attention, Pre-LN, causal convolution, Laplace approximation, NT-Xent, INT8 dynamic quantisation*

**Rule:** Open with experience words. When Matthew or a technical interviewer presses, descend the stack. Do not start at the bottom.

## 8.4 The Presentation Arc (Emotional Structure)

The 30-minute presentation is not a lecture. It is a story with emotional beats:

1. **Hook** (minute 0–1): Open with the insight. *"Every time you interact with a device, you leak information about yourself — not through what you say, but through HOW you interact."*
2. **Tension** (minute 1–4): Show the problem. Current devices ignore implicit signals. Personalisation today requires declaration.
3. **Context** (minute 4–6): Show Huawei's direction. Smart Hanhan, AI Glasses, HarmonyOS 6, Edinburgh Joint Lab talk.
4. **Promise** (minute 6–8): Reveal the thesis. "We can build this — and I did."
5. **Architecture** (minute 8–12): The three-tier system, each tier in one slide.
6. **Live demo** (minute 12–19): The emotional peak. Watch the embedding migrate. Watch the system recognise fatigue. Watch the accessibility adaptation.
7. **Edge story** (minute 19–23): The profiling report. "This fits in a Smart Hanhan's hardware budget."
8. **Implications** (minute 23–27): Portability to Huawei's device ecosystem. What this enables.
9. **Honesty** (minute 27–28): What this is not. What comes next.
10. **Close** (minute 28–30): *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*

## 8.5 The Closing Line

**Rehearse this verbatim.** Do not improvise the ending.

> *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*

Short. Unambiguous. Confident. Not a question. Not an ask. A declaration.

---

# PART 9: THE PROJECT — IMPLICIT INTERACTION INTELLIGENCE (I³)

## 9.1 Project Name

**Implicit Interaction Intelligence (I³)**

**Presentation Title:** *"Implicit Interaction Intelligence: Adaptive Companion Systems That Learn From How You Interact"*

## 9.2 The Core Concept

An AI companion system that builds a rich, evolving model of each user from **implicit behavioural signals** — not from what they say, but from HOW they interact — and uses that model to continuously adapt its responses across multiple dimensions simultaneously:

- **Cognitive load** (how much effort the user has)
- **Communication style** (formal / casual / terse / elaborate)
- **Emotional tone** (energy level, mood indicators)
- **Accessibility needs** (motor difficulty, attention constraints)

## 9.3 The Insight That Elevates This Above a Chatbot

Every time a person interacts with a device, they leak information about themselves — not through what they say, but through HOW they interact:

- Typing speed changes when tired
- Pause duration increases when uncertain
- Vocabulary simplifies when stressed
- Response length shortens when disengaged
- Correction frequency rises when distracted
- Time-of-day patterns reveal routine and disruption

**None of this requires the user to say "I'm tired" — the interaction itself is the signal.**

Most AI systems ignore this information. They respond to text content, possibly with basic sentiment analysis. I³ builds a rich computational representation from the full dynamics and uses it to adapt.

## 9.4 Why This Project Hits Every Role Requirement

| Role Requirement | How I³ Proves It |
|------------------|-------------------|
| Tier 1 (custom ML from scratch) | TCN encoder from raw PyTorch for behavioural signal extraction |
| Tier 2 (SLMs without heavy frameworks) | Custom transformer with cross-attention conditioning |
| Tier 3 (solutions on foundational models) | Anthropic Claude API integration with dynamic system prompts |
| Orchestration from blueprint | Contextual Thompson sampling bandit routing between local / cloud |
| Edge deployment | INT8 quantisation + profiling report (Kirin 9000 / 820 / Smart Hanhan class) |
| Translating abstract ideas to AI/ML | The entire project is an abstract HMI hypothesis made concrete |
| Interactive systems | Real-time WebSocket conversation |
| Personalisation | Three-timescale user model with deviation detection |
| User modelling | 64-dim latent state + long-term baseline + contextual adaptation |
| Intelligent interfaces | UI adapts in real-time to inferred user state |
| Human-AI interaction | Core focus — implicit signals, adaptive responses, accessibility |
| HCI principles | Explicit accessibility dimension, user-centred design |
| Concept-driven prototyping | Concept → working prototype in 17 days |
| Applied innovation environments | Demonstrates lab-style rapid prototyping |

# PART 10: SYSTEM ARCHITECTURE

> **Claude Code:** The architecture below is one coherent design that hits all the project goals. Treat it as a starting point. If you see structural improvements — cleaner module boundaries, better separation of concerns, a component that could be absorbed into another, an abstraction worth introducing — take them. The diagram is a communication artefact; the actual system you build is whatever works best. Deviations should be noted in `NOTES.md` with the reasoning.

## 10.1 High-Level Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │         WEB FRONTEND            │
                        │  Chat + Real-time Dashboards    │
                        └──────────────┬──────────────────┘
                                       │ WebSocket (bidirectional)
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  INTERACTION MONITOR                            │ │
│  │  Captures raw interaction signals in real-time:                 │ │
│  │  • Keystroke timing (inter-key intervals, burst patterns)      │ │
│  │  • Message composition dynamics (pauses, edits, deletions)     │ │
│  │  • Session-level patterns (message frequency, length trends)   │ │
│  │  • Linguistic features (vocabulary richness, complexity)       │ │
│  │  Outputs: InteractionFeatureVector (32 features per message)   │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              USER STATE ENCODER (Custom TCN)                    │ │
│  │  Temporal Convolutional Network built from scratch in PyTorch   │ │
│  │  Input: Sequence of InteractionFeatureVectors (window=10 msgs) │ │
│  │  Output: UserStateEmbedding (64-dim continuous vector)          │ │
│  │                                                                 │ │
│  │  Architecture:                                                  │ │
│  │  • 4 causal conv blocks with dilations [1, 2, 4, 8]           │ │
│  │  • Residual connections + LayerNorm                            │ │
│  │  • Global average pooling → 64-dim embedding                  │ │
│  │  • Trained on synthetic interaction data with known states     │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                   USER MODEL (Persistent)                       │ │
│  │                                                                 │ │
│  │  Three-timescale representation:                                │ │
│  │  ├─ Instant State: current UserStateEmbedding                  │ │
│  │  ├─ Session Profile: EMA of states within current session      │ │
│  │  └─ Long-term Profile: EMA of session profiles across sessions │ │
│  │                                                                 │ │
│  │  Computes deviation metrics:                                    │ │
│  │  • current_vs_baseline: how different is now vs. their norm    │ │
│  │  • current_vs_session: how has this session evolved            │ │
│  │  • engagement_score: derived from interaction tempo/depth      │ │
│  │                                                                 │ │
│  │  Persists to SQLite (embeddings only, never raw text)          │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│                         ├──────────────────────┐                     │
│                         ▼                      ▼                     │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │
│  │    ADAPTATION CONTROLLER     │  │   INTELLIGENT ROUTER         │ │
│  │                              │  │   (Contextual Thompson       │ │
│  │  Maps UserModel outputs to   │  │    Sampling Bandit)          │ │
│  │  4 adaptation dimensions:    │  │                              │ │
│  │                              │  │  Arms:                       │ │
│  │  1. cognitive_load: 0→1      │  │  • local_slm (on-device)    │ │
│  │     (simple ↔ complex)       │  │  • cloud_llm (API call)     │ │
│  │  2. style_mirror: 4-vector   │  │                              │ │
│  │     (formal/casual/etc)      │  │  Context features:           │ │
│  │  3. emotional_tone: 0→1      │  │  • user_state_embedding     │ │
│  │     (supportive ↔ neutral)   │  │  • query_complexity_est     │ │
│  │  4. accessibility: 0→1       │  │  • topic_sensitivity        │ │
│  │     (standard ↔ simplified)  │  │  • user_patience_signal     │ │
│  │                              │  │                              │ │
│  │  Output: AdaptationVector    │  │  Reward signal:              │ │
│  │  (fed to SLM as condition)   │  │  • user engagement after    │ │
│  │                              │  │    response (continued,     │ │
│  └──────────┬───────────────────┘  │    disengaged, topic change)│ │
│             │                      └──────────┬───────────────────┘ │
│             │                                  │                     │
│             ▼                                  ▼                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              RESPONSE GENERATOR                                 │ │
│  │                                                                 │ │
│  │  Route A: LOCAL SLM (Custom Transformer)                       │ │
│  │  ├─ Architecture: Pre-LN Transformer (from coursework)        │ │
│  │  ├─ Causal language model, ~8-15M parameters                   │ │
│  │  ├─ Cross-attention layer: conditions generation on            │ │
│  │  │   AdaptationVector (cognitive_load, style, emotion, access) │ │
│  │  ├─ Trained on DailyDialog + EmpatheticDialogues + synthetic  │ │
│  │  ├─ INT8 quantised for edge feasibility                       │ │
│  │  └─ Designed for short (1-3 sentence) adaptive responses      │ │
│  │                                                                 │ │
│  │  Route B: CLOUD LLM (Anthropic Claude API)                    │ │
│  │  ├─ System prompt dynamically constructed from AdaptationVector│ │
│  │  ├─ Includes user model summary (no raw history)              │ │
│  │  ├─ Used for complex queries, creative tasks, longer responses│ │
│  │  └─ Response post-processed to match adaptation parameters    │ │
│  │                                                                 │ │
│  │  Both routes: response passed through AdaptationFilter         │ │
│  │  that enforces max_length, vocabulary_level, tone based       │ │
│  │  on the current AdaptationVector                               │ │
│  └──────────────────────┬──────────────────────────────────────────┘ │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              INTERACTION DIARY                                  │ │
│  │                                                                 │ │
│  │  After each exchange:                                           │ │
│  │  • Logs: timestamp, dominant_emotion_cluster, topics (keywords)│ │
│  │  • Logs: adaptation_decisions, route_chosen, engagement_signal │ │
│  │  • NEVER stores raw user text (privacy by architecture)        │ │
│  │                                                                 │ │
│  │  After each session:                                            │ │
│  │  • Generates session summary (emotional arc, key topics)       │ │
│  │  • Updates long-term user profile                              │ │
│  │  • Computes "relationship strength" metric                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## 10.2 Directory Structure

```
implicit-interaction-intelligence/
├── README.md
├── pyproject.toml
├── config/
│   ├── default.yaml              # All hyperparameters, model configs, paths
│   └── demo.yaml                 # Demo-specific overrides
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Preprocessed training data
│   └── synthetic/                # Generated synthetic interaction data
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic/dataclass config loader
│   │
│   ├── interaction/              # LAYER 1: Behavioural Signal Extraction
│   │   ├── __init__.py
│   │   ├── monitor.py            # Real-time interaction signal capture
│   │   ├── features.py           # Feature engineering from raw signals
│   │   ├── linguistic.py         # Linguistic complexity analysis
│   │   ├── sentiment.py          # Custom lexicon-based sentiment
│   │   └── types.py              # InteractionEvent, InteractionFeatureVector
│   │
│   ├── encoder/                  # USER STATE ENCODER (Custom TCN)
│   │   ├── __init__.py
│   │   ├── tcn.py                # Temporal Convolutional Network
│   │   ├── blocks.py             # CausalConv1d, ResidualBlock
│   │   ├── loss.py               # NT-Xent contrastive loss
│   │   ├── train.py              # Training loop
│   │   └── inference.py          # Real-time inference wrapper
│   │
│   ├── user_model/               # PERSISTENT USER MODEL
│   │   ├── __init__.py
│   │   ├── model.py              # Three-timescale user representation
│   │   ├── deviation.py          # Baseline deviation computation
│   │   ├── store.py              # SQLite persistence
│   │   └── types.py              # UserProfile, SessionState
│   │
│   ├── adaptation/               # ADAPTATION CONTROLLER
│   │   ├── __init__.py
│   │   ├── controller.py         # Maps user state → adaptation dimensions
│   │   ├── dimensions.py         # Cognitive, Style, Emotional, Accessibility
│   │   └── types.py              # AdaptationVector, StyleVector
│   │
│   ├── router/                   # INTELLIGENT ROUTER
│   │   ├── __init__.py
│   │   ├── bandit.py             # Contextual Thompson Sampling
│   │   ├── complexity.py         # Query complexity estimator
│   │   ├── sensitivity.py        # Topic sensitivity classifier
│   │   └── types.py              # RoutingDecision, RoutingContext
│   │
│   ├── slm/                      # CUSTOM SMALL LANGUAGE MODEL
│   │   ├── __init__.py
│   │   ├── tokenizer.py          # Word-level tokenizer
│   │   ├── embeddings.py         # Token + positional embeddings
│   │   ├── attention.py          # Multi-head self-attention
│   │   ├── cross_attention.py    # Cross-attention for conditioning
│   │   ├── transformer.py        # Pre-LN AdaptiveTransformerBlock
│   │   ├── model.py              # Full SLM with conditioning interface
│   │   ├── train.py              # Training loop
│   │   ├── generate.py           # Inference (temperature, top-k, top-p)
│   │   └── quantize.py           # INT8 dynamic quantisation + profiling
│   │
│   ├── cloud/                    # CLOUD LLM INTEGRATION
│   │   ├── __init__.py
│   │   ├── client.py             # Anthropic API client
│   │   ├── prompt_builder.py     # Dynamic system prompt construction
│   │   └── postprocess.py        # Response filtering
│   │
│   ├── diary/                    # INTERACTION DIARY
│   │   ├── __init__.py
│   │   ├── logger.py             # Per-exchange logging
│   │   ├── summarizer.py         # Session summary generation
│   │   └── store.py              # SQLite diary persistence
│   │
│   ├── pipeline/                 # ORCHESTRATION PIPELINE
│   │   ├── __init__.py
│   │   ├── engine.py             # Main pipeline engine
│   │   └── types.py              # PipelineInput, PipelineOutput
│   │
│   ├── privacy/                  # PRIVACY LAYER
│   │   ├── __init__.py
│   │   ├── sanitizer.py          # Strip PII
│   │   └── encryption.py         # Encrypt user model at rest
│   │
│   └── profiling/                # EDGE FEASIBILITY
│       ├── __init__.py
│       ├── memory.py             # Memory footprint measurement
│       ├── latency.py            # Inference latency benchmarking
│       └── report.py             # Profiling report generator
│
├── training/                     # TRAINING SCRIPTS
│   ├── generate_synthetic.py     # Synthetic interaction data
│   ├── prepare_dialogue.py       # Prepare dialogue datasets
│   ├── train_encoder.py          # Train the TCN encoder
│   ├── train_slm.py              # Train the SLM
│   └── evaluate.py               # Evaluation metrics
│
├── server/                       # WEB SERVER
│   ├── __init__.py
│   ├── app.py                    # FastAPI application
│   ├── websocket.py              # WebSocket handler
│   ├── routes.py                 # REST endpoints
│   └── static/                   # Frontend files
│       ├── index.html
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── app.js
│           ├── chat.js
│           ├── dashboard.js
│           ├── embedding_viz.js
│           └── websocket.js
│
├── models/                       # Saved model checkpoints
│   ├── encoder/
│   └── slm/
│
├── demo/                         # DEMO UTILITIES
│   ├── seed_data.py              # Pre-seed "yesterday's session"
│   ├── scenarios.py              # Scripted demo scenarios
│   └── profiles.py               # Pre-built user profiles
│
└── tests/
    ├── test_tcn.py
    ├── test_slm.py
    ├── test_bandit.py
    ├── test_user_model.py
    └── test_pipeline.py
```

## 10.3 Complete Configuration Schema

```yaml
# config/default.yaml

project:
  name: "Implicit Interaction Intelligence"
  version: "1.0.0"
  seed: 42

interaction:
  feature_window: 10          # Number of messages to consider for state
  keystroke_features: true    # Whether frontend sends keystroke timing
  linguistic_features: true   # Compute linguistic complexity
  feature_dim: 32             # Dimension of InteractionFeatureVector

encoder:
  architecture: "tcn"
  input_dim: 32               # Must match interaction.feature_dim
  hidden_dims: [64, 64, 64, 64]
  kernel_size: 3
  dilations: [1, 2, 4, 8]
  dropout: 0.1
  embedding_dim: 64           # UserStateEmbedding dimension
  use_layer_norm: true
  use_residual: true

  training:
    batch_size: 64
    learning_rate: 1.0e-3
    weight_decay: 0.01
    max_epochs: 100
    temperature: 0.07         # NT-Xent temperature
    gradient_clip: 1.0
    checkpoint_every: 10

user_model:
  session_ema_alpha: 0.3      # EMA decay for within-session state
  longterm_ema_alpha: 0.1     # EMA decay for cross-session profile
  baseline_warmup: 5          # Messages before baseline is established
  deviation_threshold: 1.5    # Std devs from baseline to flag change
  max_history_sessions: 50    # Max sessions to retain

adaptation:
  cognitive_load:
    min_response_length: 10
    max_response_length: 150
    vocabulary_levels: 3
  style_mirror:
    dimensions: 4
    adaptation_rate: 0.2
  emotional_tone:
    warmth_range: [0.0, 1.0]
    default: 0.5
  accessibility:
    detection_threshold: 0.7
    simplification_levels: 3

router:
  arms: ["local_slm", "cloud_llm"]
  bandit_type: "contextual_thompson"
  context_dim: 12
  prior_alpha: 1.0
  prior_beta: 1.0
  exploration_bonus: 0.1
  min_cloud_complexity: 0.6
  privacy_override: true

slm:
  vocab_size: 8000
  max_seq_len: 256
  d_model: 256
  n_heads: 4
  n_layers: 4
  d_ff: 512
  dropout: 0.1
  conditioning_dim: 64         # Must match encoder.embedding_dim
  adaptation_dim: 8            # AdaptationVector flattened
  cross_attention_heads: 2
  use_pre_ln: true
  tie_weights: true

  training:
    batch_size: 32
    learning_rate: 3.0e-4
    warmup_steps: 500
    max_steps: 50000
    gradient_clip: 1.0
    weight_decay: 0.01
    checkpoint_every: 5000

  generation:
    temperature: 0.8
    top_k: 50
    top_p: 0.9
    max_new_tokens: 100
    repetition_penalty: 1.2

  quantization:
    enabled: true
    dtype: "int8"
    method: "dynamic"

cloud:
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  max_tokens: 200
  timeout: 10.0
  fallback_on_error: true

diary:
  db_path: "data/diary.db"
  max_entries: 1000
  session_summary_model: "cloud"
  encrypt_at_rest: true

privacy:
  strip_pii: true
  never_store_raw_text: true
  encrypt_embeddings: true
  encryption_key_env: "I3_ENCRYPTION_KEY"

profiling:
  enabled: true
  benchmark_iterations: 100
  target_devices:
    - name: "Kirin 9000 (Phone)"
      memory_mb: 512
      tops: 2.0
    - name: "Kirin A2 (Wearable)"
      memory_mb: 128
      tops: 0.5
    - name: "Smart Hanhan (IoT)"
      memory_mb: 64
      tops: 0.1

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  websocket_ping_interval: 30
```

---

# PART 11: COMPONENT IMPLEMENTATION DETAILS

> **Claude Code:** Code snippets in this part are **reference implementations**, not prescribed code. They exist to communicate intent, interfaces, and one way these components can be built correctly. You are free — encouraged — to write cleaner, more idiomatic, better-tested, better-factored versions. Use current best practice for PyTorch, FastAPI, and Python. If a snippet here uses a pattern you know a better alternative to, use the better alternative. The contract to honour is the component's **role in the system** (its inputs, outputs, and behaviour), not its specific line-by-line implementation. If you introduce a different structure, document it in `NOTES.md`.

## 11.1 Interaction Monitor & Feature Engineering

### Purpose

Capture HOW the user interacts, not just WHAT they say. This is the novel perception layer — the equivalent of sensor fusion for interaction data.

### Raw Signals Captured

The frontend sends keystroke-level events via WebSocket. The monitor processes these into structured features.

#### Keystroke Dynamics
- `inter_key_intervals`: list of milliseconds between consecutive keystrokes within a message
- `typing_burst_lengths`: lengths of continuous typing streaks (separated by pauses > 500ms)
- `pause_durations`: duration of pauses between bursts
- `backspace_frequency`: ratio of backspace events to total keystrokes
- `composition_time`: total time from first keystroke to message send
- `pause_before_send`: time between last keystroke and pressing enter

#### Message-Level Features
- `message_length_chars`: raw character count
- `message_length_tokens`: token count (simple whitespace split)
- `response_latency`: time from receiving AI's response to starting to type
- `edit_distance_ratio`: estimated editing effort (backspaces / total keystrokes)

#### Linguistic Features (computed server-side from message text)
- `type_token_ratio`: unique words / total words (vocabulary richness)
- `mean_word_length`: average characters per word (vocabulary sophistication proxy)
- `sentence_count`: number of sentences in the message
- `question_ratio`: fraction of sentences that are questions
- `exclamation_ratio`: fraction of sentences with exclamation marks
- `emoji_count`: number of emoji characters
- `formality_score`: heuristic based on contractions, slang markers, punctuation patterns
- `flesch_kincaid_grade`: readability score (vowel-group syllable heuristic)

#### Session-Level Features (sliding window)
- `message_length_trend`: slope of linear regression on message lengths over last N messages
- `response_latency_trend`: slope of response latencies
- `vocabulary_richness_trend`: slope of type-token ratios
- `engagement_velocity`: messages per minute (smoothed)
- `topic_coherence`: cosine similarity between consecutive message TF-IDF vectors
- `session_duration`: time since session start
- `time_of_day_deviation`: difference from user's typical session time (if baseline exists)

### InteractionFeatureVector (32-dim)

```python
@dataclass
class InteractionFeatureVector:
    # Keystroke dynamics (8 features)
    mean_iki: float              # Mean inter-key interval (normalised)
    std_iki: float               # Std of inter-key intervals
    mean_burst_length: float     # Mean typing burst length
    mean_pause_duration: float   # Mean pause between bursts
    backspace_ratio: float       # Backspace frequency
    composition_speed: float     # Characters per second
    pause_before_send: float     # Pre-send hesitation
    editing_effort: float        # Edit distance ratio

    # Message content (8 features)
    message_length: float        # Normalised message length
    type_token_ratio: float      # Vocabulary richness
    mean_word_length: float      # Word sophistication
    flesch_kincaid: float        # Readability
    question_ratio: float        # Inquisitiveness
    formality: float             # Language formality
    emoji_density: float         # Emoji usage
    sentiment_valence: float     # Basic sentiment (lexicon-based)

    # Session dynamics (8 features)
    length_trend: float          # Message length trend
    latency_trend: float         # Response speed trend
    vocab_trend: float           # Vocabulary complexity trend
    engagement_velocity: float   # Messages per minute
    topic_coherence: float       # Topic consistency
    session_progress: float      # How far into session (normalised)
    time_deviation: float        # Deviation from typical time
    response_depth: float        # User engagement with full response

    # Deviation metrics (8 features) — only after baseline established
    iki_deviation: float         # Current vs baseline inter-key interval
    length_deviation: float      # Current vs baseline message length
    vocab_deviation: float       # Current vs baseline vocabulary
    formality_deviation: float   # Current vs baseline formality
    speed_deviation: float       # Current vs baseline typing speed
    engagement_deviation: float  # Current vs baseline engagement
    complexity_deviation: float  # Current vs baseline linguistic complexity
    pattern_deviation: float     # Overall deviation magnitude

    def to_tensor(self) -> torch.Tensor:
        """Convert to 32-dim tensor for model input."""
        return torch.tensor([
            self.mean_iki, self.std_iki, self.mean_burst_length,
            self.mean_pause_duration, self.backspace_ratio,
            self.composition_speed, self.pause_before_send, self.editing_effort,
            self.message_length, self.type_token_ratio, self.mean_word_length,
            self.flesch_kincaid, self.question_ratio, self.formality,
            self.emoji_density, self.sentiment_valence,
            self.length_trend, self.latency_trend, self.vocab_trend,
            self.engagement_velocity, self.topic_coherence,
            self.session_progress, self.time_deviation, self.response_depth,
            self.iki_deviation, self.length_deviation, self.vocab_deviation,
            self.formality_deviation, self.speed_deviation,
            self.engagement_deviation, self.complexity_deviation,
            self.pattern_deviation
        ], dtype=torch.float32)
```

### Implementation Notes

- **Sentiment lexicon:** Build from scratch with ~500 positive and ~500 negative words with valence scores. Lexicon-based approach (VADER-style), NOT a pre-trained model.
- **Flesch-Kincaid:** `0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59`. Syllable counting: vowel-group heuristic (count groups of consecutive vowels in each word).
- **Baseline establishment:** After `baseline_warmup` messages (default 5), compute mean and std of each feature. All subsequent messages compared to this baseline.
- **All features normalised to [0, 1] or z-scored relative to baseline.**

## 11.2 User State Encoder (Custom TCN)

### Architecture

```
Input: [batch, seq_len, 32]  (sequence of InteractionFeatureVectors)
                    │
                    ▼
    ┌──────────────────────────────┐
    │  Input Projection (Linear)   │
    │  32 → 64                     │
    └──────────────┬───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               ▼               │
    │  CausalConvBlock(d=1)         │
    │  → LayerNorm → GELU → Dropout │
    │               │   + residual  │
    │               ▼               │
    │  CausalConvBlock(d=2)         │
    │               │   + residual  │
    │               ▼               │
    │  CausalConvBlock(d=4)         │
    │               │   + residual  │
    │               ▼               │
    │  CausalConvBlock(d=8)         │
    └─────────────┼─────────────────┘
                  │
                  ▼
    ┌──────────────────────────────┐
    │  Global Average Pooling       │
    │  [batch, 64, seq_len] → [batch, 64]
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Output Projection (Linear)  │
    │  64 → 64 (UserStateEmbedding)│
    └──────────────────────────────┘
```

### Causal Convolution Implementation

```python
class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation. Only looks at past and present timesteps."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, seq_len]
        # Left-pad only (causal)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
```

### Residual Block

```python
class CausalConvBlock(nn.Module):
    """Residual block with two causal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        # Residual projection if channel dims differ
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        residual = self.residual_proj(x)
        
        h = self.conv1(x)
        # LayerNorm expects [..., channels], so transpose
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)
        
        return h + residual
```

### Full TCN Model

```python
class TemporalConvNet(nn.Module):
    """TCN for encoding interaction sequences into user state embeddings."""
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list = [64, 64, 64, 64],
        kernel_size: int = 3,
        dilations: list = [1, 2, 4, 8],
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(hidden_dims) == len(dilations)
        
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        self.blocks = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for hidden_dim, dilation in zip(hidden_dims, dilations):
            self.blocks.append(
                CausalConvBlock(prev_dim, hidden_dim, kernel_size, dilation, dropout)
            )
            prev_dim = hidden_dim
        
        self.output_proj = nn.Linear(hidden_dims[-1], embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)  # [batch, seq_len, hidden]
        x = x.transpose(1, 2)    # [batch, hidden, seq_len] for conv
        
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=-1)  # [batch, hidden]
        
        # Output projection
        embedding = self.output_proj(x)  # [batch, embedding_dim]
        
        # L2 normalise for contrastive learning
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
```

### NT-Xent Contrastive Loss

```python
def nt_xent_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross-Entropy (NT-Xent / InfoNCE) loss.
    Pulls together embeddings with the same label, pushes apart different labels.
    """
    batch_size = embeddings.size(0)
    
    # Cosine similarity matrix
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    
    # Mask self-similarity (diagonal)
    mask_self = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(mask_self, -float('inf'))
    
    # Positive mask: same label
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_eq = labels_eq.masked_fill(mask_self, False)
    
    # For each anchor, log-softmax over all others
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    
    # Mean log-prob over positive pairs
    mean_log_prob_pos = (labels_eq * log_prob).sum(dim=1) / labels_eq.sum(dim=1).clamp(min=1)
    
    return -mean_log_prob_pos.mean()
```

**Why TCN over LSTM/GRU (defensible in Q&A):**
- Parallelisable (important for edge inference)
- Fixed memory footprint (important for constrained devices)
- Dilated architecture captures long-range dependencies efficiently
- No hidden state to maintain across calls
- More interpretable (you can see which timesteps contribute via the receptive field)

## 11.3 Adaptive Response SLM (Custom Transformer)

### Architecture

```
Input Tokens: [batch, seq_len]
                    │
                    ▼
    ┌──────────────────────────────┐
    │  Token Embedding (vocab→256) │
    │  + Sinusoidal Positional Enc │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  AdaptiveTransformerBlock × 4                 │
    │  ┌────────────────────────────────────────┐  │
    │  │  LayerNorm                              │  │
    │  │  → Multi-Head Self-Attention (4 heads)  │  │
    │  │  → Residual + Dropout                   │  │
    │  │                                          │  │
    │  │  LayerNorm                              │  │
    │  │  → Cross-Attention to Conditioning      │  │  ← AdaptationVector + UserStateEmbedding
    │  │  → Residual + Dropout                   │  │
    │  │                                          │  │
    │  │  LayerNorm                              │  │
    │  │  → Feed-Forward (256→512→256)           │  │
    │  │  → Residual + Dropout                   │  │
    │  └────────────────────────────────────────┘  │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────┐
    │  LayerNorm → Linear(256→vocab)│
    │  → Log-Softmax                │
    └──────────────────────────────┘
```

### Multi-Head Self-Attention (From Scratch)

```python
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from scratch."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)
```

### Multi-Head Cross-Attention (From Scratch)

```python
class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: query from sequence, key/value from conditioning tokens."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)
        cond_len = key.size(1)
        
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, cond_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, cond_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)
```

### Conditioning Projector

```python
class ConditioningProjector(nn.Module):
    """Projects AdaptationVector + UserStateEmbedding into conditioning tokens."""
    
    def __init__(self, adaptation_dim: int = 8, user_state_dim: int = 64, 
                 d_model: int = 256, n_tokens: int = 4):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(adaptation_dim + user_state_dim, d_model * n_tokens),
            nn.GELU(),
            nn.Linear(d_model * n_tokens, d_model * n_tokens),
        )
        self.n_tokens = n_tokens
        self.d_model = d_model
    
    def forward(self, adaptation_vector: torch.Tensor, 
                user_state_embedding: torch.Tensor) -> torch.Tensor:
        # adaptation_vector: [batch, 8]
        # user_state_embedding: [batch, 64]
        combined = torch.cat([adaptation_vector, user_state_embedding], dim=-1)
        projected = self.projection(combined)
        return projected.view(-1, self.n_tokens, self.d_model)  # [batch, 4, 256]
```

### Adaptive Transformer Block (Pre-LN with Cross-Attention)

```python
class AdaptiveTransformerBlock(nn.Module):
    """Pre-LN transformer block with self-attention + cross-attention + FFN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float, n_cross_heads: int = 2):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, conditioning_tokens, causal_mask=None):
        # Pre-LN Self-Attention
        h = self.ln1(x)
        h = self.self_attn(h, h, h, mask=causal_mask)
        x = x + self.dropout(h)
        
        # Pre-LN Cross-Attention to conditioning
        h = self.ln2(x)
        h = self.cross_attn(query=h, key=conditioning_tokens, value=conditioning_tokens)
        x = x + self.dropout(h)
        
        # Pre-LN Feed-Forward
        h = self.ln3(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        
        return x
```

### Sinusoidal Positional Encoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)
```

### Full Adaptive SLM

```python
class AdaptiveSLM(nn.Module):
    """Small Language Model with user-state and adaptation conditioning."""
    
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        conditioning_dim: int = 64,
        adaptation_dim: int = 8,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.conditioning_projector = ConditioningProjector(
            adaptation_dim, conditioning_dim, d_model, n_tokens=4
        )
        
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, adaptation_vector, user_state_embedding):
        # input_ids: [batch, seq_len]
        # adaptation_vector: [batch, 8]
        # user_state_embedding: [batch, 64]
        
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens + add positional encoding
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        
        # Build conditioning tokens
        conditioning = self.conditioning_projector(adaptation_vector, user_state_embedding)
        # [batch, 4, d_model]
        
        # Causal mask for self-attention
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, conditioning, causal_mask=causal_mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        adaptation_vector,
        user_state_embedding,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ):
        """Autoregressive generation with temperature, top-k, top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_cond = input_ids[:, -self.pos_encoding.pe.size(0):]
            
            logits = self(input_cond, adaptation_vector, user_state_embedding)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, -1:]] = -float('inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == eos_token_id:
                break
        
        return input_ids
```

## 11.4 Simple Tokenizer (Word-Level with BPE Fallback)

```python
class SimpleTokenizer:
    """Word-level tokenizer with special tokens and OOV handling."""
    
    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[UNK]"]
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
    
    def build_vocab(self, texts: list[str]):
        """Build vocabulary from training corpus."""
        word_counts = Counter()
        for text in texts:
            word_counts.update(self._preprocess(text).split())
        
        # Special tokens first
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Most frequent words
        for word, _ in word_counts.most_common(self.vocab_size - len(self.SPECIAL_TOKENS)):
            idx = len(self.token_to_id)
            self.token_to_id[word] = idx
            self.id_to_token[idx] = word
    
    def _preprocess(self, text: str) -> str:
        # Simple preprocessing: lowercase, separate punctuation
        text = text.lower()
        for punct in '.,!?;:"()[]':
            text = text.replace(punct, f' {punct} ')
        return ' '.join(text.split())
    
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        tokens = []
        if add_special:
            tokens.append(self.token_to_id["[BOS]"])
        for word in self._preprocess(text).split():
            tokens.append(self.token_to_id.get(word, self.token_to_id["[UNK]"]))
        if add_special:
            tokens.append(self.token_to_id["[EOS]"])
        return tokens
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words = []
        for i in ids:
            token = self.id_to_token.get(i, "[UNK]")
            if skip_special and token in self.SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.token_to_id, f)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
```

## 11.5 Contextual Thompson Sampling Router

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class RoutingContext:
    user_state_compressed: np.ndarray  # 4-dim compressed user state
    query_complexity: float             # 0-1
    topic_sensitivity: float            # 0-1
    user_patience: float                # 0-1
    session_progress: float             # 0-1
    baseline_established: float         # 0 or 1
    previous_route: int                 # 0 or 1
    previous_engagement: float          # 0-1
    time_of_day: float                  # 0-1 (normalised hour)
    message_count: float                # Normalised message count
    cloud_latency_est: float            # Estimated cloud latency (normalised)
    slm_confidence: float               # SLM's own confidence
    
    def to_array(self) -> np.ndarray:
        return np.array([
            *self.user_state_compressed,
            self.query_complexity,
            self.topic_sensitivity,
            self.user_patience,
            self.session_progress,
            self.baseline_established,
            self.previous_route,
            self.previous_engagement,
            self.time_of_day,
            self.message_count,
            self.cloud_latency_est,
            self.slm_confidence,
        ])


class ContextualThompsonBandit:
    """
    Contextual Thompson Sampling with Beta-Bernoulli posteriors extended
    with contextual features via online Bayesian logistic regression.
    
    For each arm: maintain posterior over weight vector mapping context → P(success).
    """
    
    def __init__(
        self,
        n_arms: int = 2,
        context_dim: int = 12,
        prior_precision: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        
        # Laplace approximation: posterior mean and covariance per arm
        self.weight_means = [np.zeros(context_dim) for _ in range(n_arms)]
        self.weight_covs = [
            np.eye(context_dim) * (1.0 / prior_precision) for _ in range(n_arms)
        ]
        self.precisions = [np.eye(context_dim) * prior_precision for _ in range(n_arms)]
        
        # Simple Beta-Bernoulli fallback
        self.alpha = [prior_alpha] * n_arms
        self.beta = [prior_beta] * n_arms
        
        # History
        self.history = [[] for _ in range(n_arms)]
    
    def select_arm(self, context: RoutingContext, 
                   override_to: Optional[int] = None) -> int:
        """Select arm via Thompson sampling."""
        if override_to is not None:
            return override_to
        
        context_arr = context.to_array()
        sampled_rewards = []
        
        for arm in range(self.n_arms):
            # Sample weights from posterior
            sampled_weights = np.random.multivariate_normal(
                self.weight_means[arm], self.weight_covs[arm]
            )
            # Expected reward under sampled weights
            logit = context_arr @ sampled_weights
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
            sampled_rewards.append(prob)
        
        return int(np.argmax(sampled_rewards))
    
    def update(self, arm: int, context: RoutingContext, reward: float):
        """Update posterior via online Bayesian logistic regression (Laplace)."""
        context_arr = context.to_array()
        self.history[arm].append((context_arr, reward))
        
        # Update Beta-Bernoulli fallback
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
        # Update logistic regression posterior (Laplace approximation)
        # Gradient of log-likelihood
        mean = self.weight_means[arm]
        logit = context_arr @ mean
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        
        # Gradient: (reward - sigmoid) * context
        grad = (reward - sigmoid) * context_arr
        
        # Hessian contribution: sigmoid * (1 - sigmoid) * context @ context.T
        hess = sigmoid * (1 - sigmoid) * np.outer(context_arr, context_arr)
        
        # Update precision (inverse covariance)
        self.precisions[arm] += hess
        
        # Update mean via one Newton step
        self.weight_covs[arm] = np.linalg.inv(self.precisions[arm] + 1e-6 * np.eye(self.context_dim))
        self.weight_means[arm] = self.weight_means[arm] + self.weight_covs[arm] @ grad
    
    def get_confidence(self, context: RoutingContext) -> dict:
        """Return P(arm is best) estimate for visualisation."""
        context_arr = context.to_array()
        probs = []
        for arm in range(self.n_arms):
            logit = context_arr @ self.weight_means[arm]
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
            probs.append(prob)
        total = sum(probs)
        if total == 0:
            return {f"arm_{i}": 1/self.n_arms for i in range(self.n_arms)}
        return {f"arm_{i}": p/total for i, p in enumerate(probs)}
    
    def save(self, path: str):
        np.savez(
            path,
            weight_means=np.array(self.weight_means),
            weight_covs=np.array(self.weight_covs),
            precisions=np.array(self.precisions),
            alpha=np.array(self.alpha),
            beta=np.array(self.beta),
        )
    
    def load(self, path: str):
        data = np.load(path)
        self.weight_means = list(data['weight_means'])
        self.weight_covs = list(data['weight_covs'])
        self.precisions = list(data['precisions'])
        self.alpha = list(data['alpha'])
        self.beta = list(data['beta'])
```

### Topic Sensitivity Detection (Regex-Based)

```python
SENSITIVE_PATTERNS = [
    (r'\b(health|medical|doctor|sick|pain|symptom|disease|illness)\b', 0.9),
    (r'\b(money|salary|debt|financial|bank|loan|mortgage)\b', 0.8),
    (r'\b(relationship|divorce|breakup|dating|love|romance)\b', 0.7),
    (r'\b(mental|anxiety|depressed|therapy|suicide|suicidal)\b', 1.0),
    (r'\b(password|secret|private|confidential|ssn)\b', 1.0),
    (r'\b(family|kids|children|parents|mother|father)\b', 0.5),
    (r'\b(work|job|fired|layoff|career|boss)\b', 0.6),
]

def detect_sensitivity(text: str) -> float:
    """Return max sensitivity score for text."""
    import re
    max_score = 0.0
    for pattern, score in SENSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            max_score = max(max_score, score)
    return max_score
```

### Query Complexity Estimator

```python
def estimate_complexity(text: str) -> float:
    """
    Heuristic query complexity estimator.
    Returns 0 (simple) to 1 (complex).
    """
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    # Length factor
    length_factor = min(1.0, len(words) / 50.0)
    
    # Question complexity
    question_words = {'why', 'how', 'explain', 'describe', 'compare', 'analyse', 
                      'analyze', 'evaluate', 'discuss'}
    has_complex_q = any(w.lower() in question_words for w in words)
    
    # Multi-clause (presence of conjunctions)
    conjunctions = {'and', 'but', 'or', 'because', 'although', 'however', 
                    'therefore', 'furthermore'}
    conjunction_count = sum(1 for w in words if w.lower() in conjunctions)
    multi_clause_factor = min(1.0, conjunction_count / 3.0)
    
    # Vocabulary complexity (mean word length as proxy)
    mean_word_length = sum(len(w) for w in words) / len(words)
    vocab_factor = min(1.0, (mean_word_length - 3) / 5.0)  # Normalise
    
    # Combine factors
    complexity = (
        0.3 * length_factor +
        0.3 * (1.0 if has_complex_q else 0.0) +
        0.2 * multi_clause_factor +
        0.2 * max(0, vocab_factor)
    )
    
    return min(1.0, complexity)
```

## 11.6 User Model (Three-Timescale)

```python
@dataclass
class UserProfile:
    user_id: str
    baseline_features: Optional[dict]  # Feature means + stds after warmup
    baseline_embedding: Optional[np.ndarray]  # 64-dim baseline state
    total_sessions: int
    total_messages: int
    relationship_strength: float
    long_term_style: dict
    created_at: datetime
    updated_at: datetime


@dataclass
class SessionState:
    session_id: str
    user_id: str
    start_time: datetime
    message_count: int
    session_embedding_ema: np.ndarray  # 64-dim running EMA
    feature_history: list               # Recent InteractionFeatureVectors


class UserModel:
    """Three-timescale user representation with baseline deviation."""
    
    def __init__(self, config, user_id: str, store):
        self.user_id = user_id
        self.config = config
        self.store = store
        
        self.profile = store.load_profile(user_id) or self._new_profile(user_id)
        self.session = None
        self.current_state = None
    
    def _new_profile(self, user_id: str) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            baseline_features=None,
            baseline_embedding=None,
            total_sessions=0,
            total_messages=0,
            relationship_strength=0.0,
            long_term_style={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    
    def start_session(self, session_id: str):
        self.session = SessionState(
            session_id=session_id,
            user_id=self.user_id,
            start_time=datetime.now(),
            message_count=0,
            session_embedding_ema=np.zeros(self.config.encoder.embedding_dim),
            feature_history=[],
        )
    
    def update(self, feature_vector, user_state_embedding: np.ndarray):
        """Update after each message."""
        self.current_state = user_state_embedding
        self.session.feature_history.append(feature_vector)
        self.session.message_count += 1
        
        # Update session EMA
        alpha = self.config.user_model.session_ema_alpha
        if self.session.message_count == 1:
            self.session.session_embedding_ema = user_state_embedding.copy()
        else:
            self.session.session_embedding_ema = (
                alpha * user_state_embedding + 
                (1 - alpha) * self.session.session_embedding_ema
            )
        
        # Establish baseline after warmup
        if (self.profile.baseline_features is None and 
            len(self.session.feature_history) >= self.config.user_model.baseline_warmup):
            self._establish_baseline()
    
    def _establish_baseline(self):
        """Compute baseline statistics from first N messages."""
        history = self.session.feature_history[:self.config.user_model.baseline_warmup]
        
        features_array = np.array([f.to_tensor().numpy() for f in history])
        
        self.profile.baseline_features = {
            'means': features_array.mean(axis=0).tolist(),
            'stds': features_array.std(axis=0).tolist(),
        }
        self.profile.baseline_embedding = self.session.session_embedding_ema.copy()
    
    def compute_deviation(self, feature_vector) -> dict:
        """Compute deviation metrics from baseline."""
        if self.profile.baseline_features is None:
            return {'pattern_deviation': 0.0, 'baseline_established': False}
        
        current = feature_vector.to_tensor().numpy()
        means = np.array(self.profile.baseline_features['means'])
        stds = np.array(self.profile.baseline_features['stds'])
        stds = np.maximum(stds, 1e-6)  # Avoid division by zero
        
        z_scores = (current - means) / stds
        
        return {
            'pattern_deviation': float(np.abs(z_scores).mean()),
            'iki_deviation': float(z_scores[0]),
            'length_deviation': float(z_scores[8]),
            'vocab_deviation': float(z_scores[9]),
            'formality_deviation': float(z_scores[13]),
            'speed_deviation': float(z_scores[5]),
            'engagement_deviation': float(z_scores[19]),
            'complexity_deviation': float(z_scores[11]),
            'baseline_established': True,
        }
    
    def end_session(self):
        """Update long-term profile at session end."""
        if self.session is None:
            return
        
        # Update long-term EMA
        alpha = self.config.user_model.longterm_ema_alpha
        if self.profile.baseline_embedding is None:
            self.profile.baseline_embedding = self.session.session_embedding_ema.copy()
        else:
            self.profile.baseline_embedding = (
                alpha * self.session.session_embedding_ema +
                (1 - alpha) * self.profile.baseline_embedding
            )
        
        self.profile.total_sessions += 1
        self.profile.total_messages += self.session.message_count
        self.profile.relationship_strength = min(1.0, 
            self.profile.total_sessions / 20.0 +
            self.profile.total_messages / 200.0
        )
        self.profile.updated_at = datetime.now()
        
        self.store.save_profile(self.profile)
        self.session = None
```

## 11.7 Adaptation Controller

```python
@dataclass
class StyleVector:
    formality: float    # 0=casual, 1=formal
    verbosity: float    # 0=concise, 1=elaborate
    emotionality: float # 0=reserved, 1=expressive
    directness: float   # 0=indirect, 1=direct
    
    def to_array(self) -> np.ndarray:
        return np.array([self.formality, self.verbosity, self.emotionality, self.directness])


@dataclass
class AdaptationVector:
    cognitive_load: float      # 0.0 (simplest) → 1.0 (most complex)
    style_mirror: StyleVector
    emotional_tone: float      # 0.0 (most supportive) → 1.0 (most neutral)
    accessibility: float       # 0.0 (standard) → 1.0 (maximum simplification)
    
    def to_tensor(self) -> torch.Tensor:
        """Flatten to 8-dim tensor for SLM conditioning."""
        return torch.tensor([
            self.cognitive_load,
            *self.style_mirror.to_array(),
            self.emotional_tone,
            self.accessibility,
            0.0,  # reserved
        ], dtype=torch.float32)


class AdaptationController:
    """Maps user model state → adaptation dimensions."""
    
    def __init__(self, config):
        self.config = config
    
    def compute(self, user_model) -> AdaptationVector:
        if user_model.current_state is None:
            return self._default_adaptation()
        
        # Latest features
        if not user_model.session.feature_history:
            return self._default_adaptation()
        
        current_fv = user_model.session.feature_history[-1]
        deviation = user_model.compute_deviation(current_fv)
        
        cognitive_load = self._compute_cognitive_load(current_fv, deviation)
        style_mirror = self._compute_style_mirror(current_fv, user_model.session)
        emotional_tone = self._compute_emotional_tone(current_fv, deviation)
        accessibility = self._compute_accessibility(current_fv, deviation)
        
        return AdaptationVector(cognitive_load, style_mirror, emotional_tone, accessibility)
    
    def _default_adaptation(self) -> AdaptationVector:
        return AdaptationVector(
            cognitive_load=0.5,
            style_mirror=StyleVector(0.5, 0.5, 0.5, 0.5),
            emotional_tone=0.5,
            accessibility=0.0,
        )
    
    def _compute_cognitive_load(self, fv, deviation) -> float:
        """Match user's current complexity level, slightly above."""
        complexity_signals = [
            fv.type_token_ratio,
            min(1.0, fv.mean_word_length / 10.0),
            min(1.0, fv.flesch_kincaid / 20.0),
            fv.message_length,
        ]
        user_complexity = float(np.mean(complexity_signals))
        
        if deviation.get('baseline_established') and deviation.get('complexity_deviation', 0) < -0.5:
            # User is below their baseline — match their current level
            return max(0.2, user_complexity - 0.2)
        return min(0.9, user_complexity + 0.1)
    
    def _compute_style_mirror(self, fv, session) -> StyleVector:
        """Mirror user's communication style with slight lag."""
        return StyleVector(
            formality=fv.formality,
            verbosity=min(1.0, fv.message_length * 1.2),
            emotionality=fv.emoji_density + max(0, fv.sentiment_valence),
            directness=1.0 - fv.question_ratio,
        )
    
    def _compute_emotional_tone(self, fv, deviation) -> float:
        """Increase warmth when user shows distress signals."""
        distress_signals = []
        
        if fv.sentiment_valence < -0.3:
            distress_signals.append(0.8)
        if deviation.get('baseline_established'):
            if abs(deviation.get('pattern_deviation', 0)) > 1.0:
                distress_signals.append(0.7)
            if deviation.get('engagement_deviation', 0) < -1.0:
                distress_signals.append(0.6)
        
        if distress_signals:
            return min(1.0, max(distress_signals))
        return 0.5
    
    def _compute_accessibility(self, fv, deviation) -> float:
        """Detect patterns suggesting motor/cognitive difficulty."""
        if not deviation.get('baseline_established'):
            return 0.0
        
        difficulty_signals = [
            max(0, deviation.get('iki_deviation', 0)),           # Slower than baseline
            max(0, -deviation.get('speed_deviation', 0)),         # Speed dropped
            fv.backspace_ratio * 3.0,                             # High editing
            fv.editing_effort * 2.0,                              # High effort
        ]
        difficulty_score = float(np.mean(difficulty_signals))
        
        threshold = self.config.adaptation.accessibility.detection_threshold
        if difficulty_score > threshold:
            return min(1.0, difficulty_score)
        return 0.0
```

## 11.8 Interaction Diary

### SQL Schema

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    summary TEXT,
    dominant_emotion TEXT,
    topics TEXT,                       -- JSON array
    mean_engagement REAL,
    mean_cognitive_load REAL,
    mean_accessibility REAL,
    relationship_strength REAL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

CREATE TABLE exchanges (
    exchange_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    user_state_embedding BLOB,         -- 64-dim float32 (256 bytes)
    adaptation_vector TEXT,             -- JSON
    route_chosen TEXT,                  -- "local_slm" or "cloud_llm"
    response_latency_ms INTEGER,
    engagement_signal REAL,
    topics TEXT,                        -- JSON array of keywords
    -- NO raw user text, NO response text
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    baseline_embedding BLOB,
    baseline_features TEXT,             -- JSON
    total_sessions INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    relationship_strength REAL DEFAULT 0.0,
    long_term_style TEXT,               -- JSON
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Topic Extraction (TF-IDF, No ML Model)

```python
class TopicExtractor:
    """Simple TF-IDF keyword extraction."""
    
    def __init__(self, stopwords_path: str = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self.idf = {}  # Pre-computed IDF from training corpus
    
    def fit(self, corpus: list[str]):
        """Pre-compute IDF from corpus."""
        n_docs = len(corpus)
        doc_freq = Counter()
        for doc in corpus:
            words = set(self._tokenise(doc))
            for word in words:
                doc_freq[word] += 1
        
        self.idf = {
            word: math.log(n_docs / (df + 1))
            for word, df in doc_freq.items()
        }
    
    def extract(self, text: str, n_topics: int = 3) -> list[str]:
        words = self._tokenise(text)
        if not words:
            return []
        
        tf = Counter(words)
        total = len(words)
        
        scores = {}
        for word, count in tf.items():
            if word in self.stopwords or len(word) < 3:
                continue
            tf_score = count / total
            idf_score = self.idf.get(word, math.log(10))  # Default IDF
            scores[word] = tf_score * idf_score
        
        top = sorted(scores.items(), key=lambda x: -x[1])[:n_topics]
        return [word for word, _ in top]
    
    def _tokenise(self, text: str) -> list[str]:
        import re
        text = text.lower()
        return [w for w in re.findall(r'\b[a-z]+\b', text) if len(w) >= 3]
    
    def _load_stopwords(self, path: Optional[str]) -> set:
        # Default small stopword list
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'and',
            'but', 'or', 'so', 'if', 'for', 'with', 'about', 'from', 'to',
            'in', 'on', 'at', 'by', 'of', 'not', 'no', 'yes'
        }
```

## 11.9 Privacy Layer

```python
import re
from cryptography.fernet import Fernet

class PrivacySanitizer:
    """Strip PII before sending to cloud or logging."""
    
    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS]'),
        (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]'),
        (r'\b\d{16}\b', '[CARD]'),
    ]
    
    def sanitize(self, text: str) -> str:
        for pattern, replacement in self.PII_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


class EmbeddingEncryptor:
    """Encrypt user embeddings at rest."""
    
    def __init__(self, key: bytes = None):
        if key is None:
            import os
            key_b64 = os.environ.get('I3_ENCRYPTION_KEY')
            if key_b64:
                key = key_b64.encode()
            else:
                key = Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt_array(self, arr: np.ndarray) -> bytes:
        return self.cipher.encrypt(arr.tobytes())
    
    def decrypt_array(self, data: bytes, dtype=np.float32, shape=None) -> np.ndarray:
        decrypted = self.cipher.decrypt(data)
        arr = np.frombuffer(decrypted, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr
```

## 11.10 Edge Feasibility Profiler

```python
import time
import tracemalloc
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class DeviceTarget:
    name: str
    memory_mb: float
    tops: float


@dataclass
class ProfileReport:
    model_name: str
    param_count: int
    fp32_size_mb: float
    int8_size_mb: float
    mean_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float
    fits_on_targets: dict


class EdgeProfiler:
    """Profile a model for edge deployment feasibility."""
    
    TARGET_DEVICES = [
        DeviceTarget("Kirin 9000 (Flagship Phone)", 512, 2.0),
        DeviceTarget("Kirin 820 (Mid-range Phone)", 256, 1.0),
        DeviceTarget("Kirin A2 (Wearable SoC)", 128, 0.5),
        DeviceTarget("Smart Hanhan (IoT MCU)", 64, 0.1),
    ]
    
    def profile(
        self,
        model: nn.Module,
        input_sample: tuple,
        model_name: str,
        n_iterations: int = 100,
    ) -> ProfileReport:
        model.eval()
        
        # 1. Parameter count
        param_count = sum(p.numel() for p in model.parameters())
        
        # 2. FP32 size
        fp32_size_mb = param_count * 4 / (1024 * 1024)
        
        # 3. Quantise to INT8
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        # Save quantized to tempfile to measure size
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(quantized.state_dict(), f.name)
            int8_size_mb = os.path.getsize(f.name) / (1024 * 1024)
            os.unlink(f.name)
        
        # 4. Latency benchmarking
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(*input_sample)
        
        latencies = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.perf_counter()
                _ = model(*input_sample)
                latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        mean_latency = float(latencies.mean())
        std_latency = float(latencies.std())
        p95_latency = float(np.percentile(latencies, 95))
        
        # 5. Peak memory during inference
        tracemalloc.start()
        with torch.no_grad():
            _ = model(*input_sample)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / (1024 * 1024)
        
        # 6. Fits on targets (using 50% memory budget)
        fits_on_targets = {
            device.name: {
                'fits': int8_size_mb < device.memory_mb * 0.5,
                'memory_utilisation': int8_size_mb / (device.memory_mb * 0.5),
            }
            for device in self.TARGET_DEVICES
        }
        
        return ProfileReport(
            model_name=model_name,
            param_count=param_count,
            fp32_size_mb=fp32_size_mb,
            int8_size_mb=int8_size_mb,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            p95_latency_ms=p95_latency,
            peak_memory_mb=peak_memory_mb,
            fits_on_targets=fits_on_targets,
        )
    
    def print_report(self, report: ProfileReport):
        print(f"\n{'='*60}")
        print(f"  EDGE FEASIBILITY REPORT: {report.model_name}")
        print(f"{'='*60}")
        print(f"  Parameters:      {report.param_count:,}")
        print(f"  FP32 size:       {report.fp32_size_mb:.2f} MB")
        print(f"  INT8 size:       {report.int8_size_mb:.2f} MB")
        print(f"  Compression:     {report.fp32_size_mb/report.int8_size_mb:.1f}x")
        print(f"  Latency (mean):  {report.mean_latency_ms:.2f} ms")
        print(f"  Latency (p95):   {report.p95_latency_ms:.2f} ms")
        print(f"  Peak memory:     {report.peak_memory_mb:.2f} MB")
        print(f"\n  Device Compatibility (INT8, 50% memory budget):")
        for device_name, info in report.fits_on_targets.items():
            status = "✓" if info['fits'] else "✗"
            util = info['memory_utilisation'] * 100
            print(f"    {status} {device_name:<35} ({util:.0f}% utilisation)")
        print(f"{'='*60}\n")
```

---

# PART 12: TRAINING DATA & PROCEDURES

> **Claude Code:** The training procedures below describe one reasonable path. Hyperparameters, loss functions, augmentation strategies, and data-generation schemes are all starting points. If you find better schedules, better losses, better data recipes — use them. Search the literature for current best practice on contrastive training for temporal encoders, curriculum strategies for small-model pretraining, and synthetic data generation. Prototype alternatives where it costs little and learns much. Document deviations in `NOTES.md`.

## 12.1 Synthetic Interaction Data (for TCN)

### User State Archetypes (8 States)

```python
USER_STATES = {
    "energetic_engaged": {
        "mean_iki_ms": (80, 120),         # Fast typing
        "typing_burst_length": (15, 30),
        "pause_duration_ms": (200, 500),
        "backspace_ratio": (0.02, 0.05),
        "message_length_words": (15, 40),
        "vocabulary_richness": (0.7, 0.9),
        "formality": (0.3, 0.6),
        "response_latency_ms": (500, 2000),
        "sentiment_valence": (0.3, 0.7),
    },
    "tired_disengaging": {
        "mean_iki_ms": (200, 400),
        "typing_burst_length": (3, 8),
        "pause_duration_ms": (1000, 3000),
        "backspace_ratio": (0.05, 0.12),
        "message_length_words": (3, 10),
        "vocabulary_richness": (0.3, 0.5),
        "formality": (0.2, 0.4),
        "response_latency_ms": (5000, 15000),
        "sentiment_valence": (-0.2, 0.2),
    },
    "stressed_urgent": {
        "mean_iki_ms": (60, 100),
        "typing_burst_length": (5, 15),
        "pause_duration_ms": (100, 300),
        "backspace_ratio": (0.08, 0.15),
        "message_length_words": (5, 15),
        "vocabulary_richness": (0.4, 0.6),
        "formality": (0.1, 0.3),
        "response_latency_ms": (200, 1000),
        "sentiment_valence": (-0.5, -0.1),
    },
    "relaxed_conversational": {
        "mean_iki_ms": (120, 180),
        "typing_burst_length": (8, 20),
        "pause_duration_ms": (500, 1500),
        "backspace_ratio": (0.03, 0.06),
        "message_length_words": (10, 25),
        "vocabulary_richness": (0.5, 0.7),
        "formality": (0.3, 0.5),
        "response_latency_ms": (2000, 5000),
        "sentiment_valence": (0.1, 0.5),
    },
    "focused_deep": {
        "mean_iki_ms": (100, 160),
        "typing_burst_length": (20, 50),
        "pause_duration_ms": (2000, 5000),
        "backspace_ratio": (0.01, 0.03),
        "message_length_words": (20, 60),
        "vocabulary_richness": (0.7, 0.95),
        "formality": (0.5, 0.8),
        "response_latency_ms": (3000, 8000),
        "sentiment_valence": (0.0, 0.4),
    },
    "motor_difficulty": {
        "mean_iki_ms": (300, 800),
        "typing_burst_length": (1, 4),
        "pause_duration_ms": (2000, 8000),
        "backspace_ratio": (0.15, 0.35),
        "message_length_words": (2, 8),
        "vocabulary_richness": (0.3, 0.5),
        "formality": (0.2, 0.4),
        "response_latency_ms": (5000, 20000),
        "sentiment_valence": (-0.1, 0.3),
    },
    "distracted_multitasking": {
        "mean_iki_ms": (100, 300),
        "typing_burst_length": (3, 10),
        "pause_duration_ms": (3000, 15000),
        "backspace_ratio": (0.05, 0.10),
        "message_length_words": (5, 15),
        "vocabulary_richness": (0.4, 0.6),
        "formality": (0.2, 0.4),
        "response_latency_ms": (10000, 60000),
        "sentiment_valence": (-0.1, 0.3),
    },
    "formal_professional": {
        "mean_iki_ms": (120, 200),
        "typing_burst_length": (10, 25),
        "pause_duration_ms": (1000, 3000),
        "backspace_ratio": (0.03, 0.07),
        "message_length_words": (15, 40),
        "vocabulary_richness": (0.6, 0.85),
        "formality": (0.7, 0.95),
        "response_latency_ms": (2000, 6000),
        "sentiment_valence": (0.0, 0.3),
    },
}
```

### State Transition Model (Markov Chain)

```python
TRANSITION_PROBS = {
    "energetic_engaged": {
        "energetic_engaged": 0.6,
        "relaxed_conversational": 0.2,
        "focused_deep": 0.1,
        "tired_disengaging": 0.05,
        "distracted_multitasking": 0.05,
    },
    "tired_disengaging": {
        "tired_disengaging": 0.7,
        "relaxed_conversational": 0.15,
        "distracted_multitasking": 0.1,
        "energetic_engaged": 0.05,
    },
    "stressed_urgent": {
        "stressed_urgent": 0.5,
        "relaxed_conversational": 0.2,
        "tired_disengaging": 0.2,
        "focused_deep": 0.1,
    },
    "relaxed_conversational": {
        "relaxed_conversational": 0.5,
        "energetic_engaged": 0.15,
        "tired_disengaging": 0.15,
        "focused_deep": 0.1,
        "distracted_multitasking": 0.1,
    },
    "focused_deep": {
        "focused_deep": 0.6,
        "relaxed_conversational": 0.2,
        "tired_disengaging": 0.15,
        "formal_professional": 0.05,
    },
    "motor_difficulty": {
        "motor_difficulty": 0.85,  # Stable state
        "tired_disengaging": 0.1,
        "relaxed_conversational": 0.05,
    },
    "distracted_multitasking": {
        "distracted_multitasking": 0.5,
        "relaxed_conversational": 0.2,
        "tired_disengaging": 0.2,
        "stressed_urgent": 0.1,
    },
    "formal_professional": {
        "formal_professional": 0.7,
        "focused_deep": 0.15,
        "relaxed_conversational": 0.1,
        "stressed_urgent": 0.05,
    },
}
```

### Synthetic Data Generator

```python
class SyntheticInteractionGenerator:
    def __init__(self, states=USER_STATES, transitions=TRANSITION_PROBS, seed=42):
        self.states = states
        self.transitions = transitions
        self.rng = np.random.default_rng(seed)
    
    def generate_session(self, n_messages: int = 20, start_state: str = None):
        if start_state is None:
            start_state = self.rng.choice(list(self.states.keys()))
        
        current_state = start_state
        features = []
        labels = []
        
        for i in range(n_messages):
            # Sample features from current state's distributions
            feature_vector = self._sample_features(current_state, position=i/n_messages)
            features.append(feature_vector)
            labels.append(current_state)
            
            # Transition
            current_state = self._transition(current_state)
        
        return features, labels
    
    def _sample_features(self, state: str, position: float) -> np.ndarray:
        """Sample a 32-dim feature vector from the state's distributions."""
        params = self.states[state]
        
        # Sample raw values
        mean_iki = self.rng.uniform(*params["mean_iki_ms"])
        burst = self.rng.uniform(*params["typing_burst_length"])
        pause = self.rng.uniform(*params["pause_duration_ms"])
        backspace = self.rng.uniform(*params["backspace_ratio"])
        msg_len_words = self.rng.uniform(*params["message_length_words"])
        vocab_rich = self.rng.uniform(*params["vocabulary_richness"])
        formality = self.rng.uniform(*params["formality"])
        resp_latency = self.rng.uniform(*params["response_latency_ms"])
        sentiment = self.rng.uniform(*params["sentiment_valence"])
        
        # Normalise to [0, 1]
        mean_iki_norm = min(1.0, mean_iki / 1000.0)
        std_iki = mean_iki * self.rng.uniform(0.2, 0.8) / 1000.0
        burst_norm = min(1.0, burst / 50.0)
        pause_norm = min(1.0, pause / 10000.0)
        msg_len_norm = min(1.0, msg_len_words / 60.0)
        resp_latency_norm = min(1.0, resp_latency / 60000.0)
        
        # Linguistic features (derived)
        mean_word_length = 3 + vocab_rich * 4  # 3-7 chars
        mean_word_length_norm = min(1.0, mean_word_length / 10.0)
        flesch_kincaid = 5 + vocab_rich * 15  # 5-20
        flesch_norm = min(1.0, flesch_kincaid / 20.0)
        composition_speed = 1.0 / mean_iki_norm if mean_iki_norm > 0 else 1.0
        composition_speed_norm = min(1.0, composition_speed)
        
        question_ratio = self.rng.uniform(0, 0.3)
        emoji_density = self.rng.uniform(0, 0.05) * (1 - formality)
        editing_effort = backspace * 2.0
        pause_before_send = self.rng.uniform(0, 2) / 10.0
        
        # Session dynamics (use position)
        length_trend = (position - 0.5) * 0.2  # Slight decline over session
        latency_trend = (position - 0.5) * 0.3
        vocab_trend = 0.0
        engagement_velocity = 60.0 / (resp_latency / 1000.0 + 1)
        engagement_velocity_norm = min(1.0, engagement_velocity / 10.0)
        topic_coherence = self.rng.uniform(0.5, 1.0)
        session_progress = position
        time_deviation = self.rng.uniform(-0.3, 0.3)
        response_depth = self.rng.uniform(0.3, 1.0)
        
        # Deviation features (zero for synthetic — baseline not applicable)
        deviations = [0.0] * 8
        
        return np.array([
            # Keystroke dynamics (8)
            mean_iki_norm, std_iki, burst_norm, pause_norm,
            backspace, composition_speed_norm, pause_before_send, editing_effort,
            # Message content (8)
            msg_len_norm, vocab_rich, mean_word_length_norm, flesch_norm,
            question_ratio, formality, emoji_density, sentiment,
            # Session dynamics (8)
            length_trend, latency_trend, vocab_trend, engagement_velocity_norm,
            topic_coherence, session_progress, time_deviation, response_depth,
            # Deviations (8)
            *deviations
        ], dtype=np.float32)
    
    def _transition(self, state: str) -> str:
        probs = self.transitions[state]
        states = list(probs.keys())
        weights = list(probs.values())
        return self.rng.choice(states, p=weights)
    
    def generate_dataset(self, n_sessions: int = 10000, messages_per_session: int = 20):
        """Generate full training dataset."""
        all_features = []
        all_labels = []
        
        for _ in range(n_sessions):
            features, labels = self.generate_session(messages_per_session)
            all_features.append(np.array(features))
            all_labels.append(labels)
        
        return all_features, all_labels
```

### Contrastive Dataset for TCN

```python
class ContrastiveInteractionDataset(torch.utils.data.Dataset):
    """
    For each sample: a window of 10 consecutive InteractionFeatureVectors
    with a single state label (majority state in the window).
    """
    
    def __init__(self, sessions, session_labels, window_size=10, state_to_idx=None):
        self.window_size = window_size
        self.state_to_idx = state_to_idx or {s: i for i, s in enumerate(USER_STATES.keys())}
        
        self.windows = []
        self.labels = []
        
        for features, labels in zip(sessions, session_labels):
            for i in range(len(features) - window_size + 1):
                window = features[i:i+window_size]
                window_labels = labels[i:i+window_size]
                # Majority label
                majority = Counter(window_labels).most_common(1)[0][0]
                self.windows.append(window)
                self.labels.append(self.state_to_idx[majority])
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return window, label
```

## 12.2 Dialogue Data Preparation (for SLM)

### Source Datasets

1. **DailyDialog** (~13,000 dialogues, ~100K utterances)
   - URL: http://yanran.li/dailydialog
   - Emotion labels: neutral, happy, sad, angry, surprise, disgust, fear
   - Topic labels: ordinary life, school life, culture, attitude, relationship, tourism, health, work, politics, finance

2. **EmpatheticDialogues** (~25,000 conversations)
   - URL: https://github.com/facebookresearch/EmpatheticDialogues
   - Has situation descriptions and 32 emotion labels
   - More emotionally diverse

### Conditioning Derivation

```python
def derive_conditioning(
    response_text: str,
    emotion_label: str,
    preceding_turns: list,
    augment_accessibility: bool = False,
) -> AdaptationVector:
    """Derive AdaptationVector from dialogue metadata for training."""
    
    # Cognitive load: based on response complexity
    words = response_text.split()
    word_count = len(words)
    mean_word_length = sum(len(w) for w in words) / max(1, word_count)
    
    cognitive_load = min(1.0, 
        0.3 * (word_count / 30) +
        0.3 * (mean_word_length - 3) / 4 +
        0.4 * type_token_ratio(response_text)
    )
    cognitive_load = max(0.1, cognitive_load)
    
    # Style
    formality = compute_formality(response_text)
    verbosity = min(1.0, word_count / 50)
    
    emotion_intensity = {
        "neutral": 0.2, "happy": 0.7, "sad": 0.6, "angry": 0.8,
        "surprise": 0.6, "disgust": 0.5, "fear": 0.7,
    }.get(emotion_label, 0.3)
    emotionality = emotion_intensity
    
    directness = 1.0 - response_text.count('?') / max(1, response_text.count('.') + response_text.count('!') + response_text.count('?'))
    
    style = StyleVector(formality, verbosity, emotionality, directness)
    
    # Emotional tone
    supportive_emotions = {"sad", "fear"}
    if emotion_label in supportive_emotions:
        emotional_tone = 0.2  # Very supportive
    elif emotion_label in {"angry", "disgust"}:
        emotional_tone = 0.4
    else:
        emotional_tone = 0.5
    
    # Accessibility (default 0, augmented separately)
    accessibility = 0.0
    if augment_accessibility:
        accessibility = np.random.uniform(0.7, 1.0)
        cognitive_load = max(0.1, cognitive_load - 0.3)
    
    return AdaptationVector(cognitive_load, style, emotional_tone, accessibility)


def type_token_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_formality(text: str) -> float:
    """Heuristic formality score."""
    informal_markers = {"lol", "omg", "yeah", "yep", "nope", "gonna", "wanna", "gotta",
                        "dunno", "ain't", "y'all", "ya"}
    contractions = re.findall(r"\b\w+'\w+\b", text.lower())
    informal_count = sum(1 for w in text.lower().split() if w in informal_markers)
    
    words = text.split()
    if not words:
        return 0.5
    
    informality = (informal_count + len(contractions)) / len(words)
    return max(0.0, 1.0 - informality * 3)
```

### Accessibility Augmentation

```python
SIMPLIFICATION_MAP = {
    'utilize': 'use', 'commence': 'start', 'endeavor': 'try',
    'elucidate': 'explain', 'magnitude': 'size', 'diminutive': 'small',
    'nevertheless': 'but', 'furthermore': 'also', 'consequently': 'so',
    'subsequently': 'then', 'approximately': 'about', 'demonstrate': 'show',
    'terminate': 'end', 'initiate': 'start', 'sufficient': 'enough',
}

def simplify_text(text: str) -> str:
    """Heuristic text simplification for accessibility training."""
    # Keep only first 1-2 sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    simplified = '. '.join(sentences[:min(2, len(sentences))]).strip()
    if simplified and not simplified[-1] in '.!?':
        simplified += '.'
    
    # Replace complex words
    for complex_w, simple_w in SIMPLIFICATION_MAP.items():
        simplified = re.sub(r'\b' + complex_w + r'\b', simple_w, simplified, flags=re.IGNORECASE)
    
    return simplified
```

## 12.3 Training Scripts

### TCN Encoder Training

```python
# training/train_encoder.py

def train_encoder(config):
    # 1. Generate synthetic data
    generator = SyntheticInteractionGenerator()
    sessions, labels = generator.generate_dataset(
        n_sessions=10000,
        messages_per_session=20,
    )
    
    # 2. Create contrastive dataset
    dataset = ContrastiveInteractionDataset(
        sessions, labels,
        window_size=config.interaction.feature_window,
    )
    
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=config.encoder.training.batch_size, 
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=config.encoder.training.batch_size, 
                            shuffle=False, num_workers=2)
    
    # 3. Build model
    model = TemporalConvNet(
        input_dim=config.interaction.feature_dim,
        hidden_dims=config.encoder.hidden_dims,
        kernel_size=config.encoder.kernel_size,
        dilations=config.encoder.dilations,
        embedding_dim=config.encoder.embedding_dim,
        dropout=config.encoder.dropout,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.encoder.training.learning_rate,
        weight_decay=config.encoder.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.encoder.training.max_epochs
    )
    
    # 4. Training loop
    best_val_score = 0.0
    for epoch in range(config.encoder.training.max_epochs):
        model.train()
        train_loss = 0
        for features, batch_labels in train_loader:
            embeddings = model(features)
            loss = nt_xent_loss(embeddings, batch_labels, 
                                temperature=config.encoder.training.temperature)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           config.encoder.training.gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        val_score = evaluate_encoder(model, val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, "
              f"val_knn_acc={val_score['knn_accuracy']:.4f}")
        
        if val_score['knn_accuracy'] > best_val_score:
            best_val_score = val_score['knn_accuracy']
            torch.save(model.state_dict(), "models/encoder/best.pt")
        
        scheduler.step()
    
    # 5. Quantise
    model.load_state_dict(torch.load("models/encoder/best.pt"))
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), "models/encoder/encoder_int8.pt")


def evaluate_encoder(model, val_loader):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            embeddings = model(features)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # K-NN accuracy
    from sklearn.neighbors import KNeighborsClassifier
    split = int(0.8 * len(embeddings))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings[:split], labels[:split])
    knn_acc = knn.score(embeddings[split:], labels[split:])
    
    # Silhouette
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(embeddings, labels)
    
    return {"knn_accuracy": knn_acc, "silhouette": silhouette}
```

### SLM Training

```python
# training/train_slm.py

def train_slm(config):
    # 1. Prepare data
    raw_dialogues = load_dialogue_datasets()  # DailyDialog + EmpatheticDialogues
    training_samples = []
    for dialogue in raw_dialogues:
        samples = extract_training_samples(dialogue)
        training_samples.extend(samples)
    
    # Accessibility augmentation (20% of samples)
    augmented = []
    for sample in np.random.choice(training_samples, size=len(training_samples)//5, replace=False):
        augmented.append(augment_accessibility(sample))
    training_samples.extend(augmented)
    
    # 2. Build tokenizer
    tokenizer = SimpleTokenizer(config.slm.vocab_size)
    all_text = [s.input_text + " " + s.target_text for s in training_samples]
    tokenizer.build_vocab(all_text)
    tokenizer.save("models/slm/tokenizer.json")
    
    # 3. Create dataset
    dataset = DialogueDataset(training_samples, tokenizer, config.slm.max_seq_len)
    train_loader = DataLoader(dataset, batch_size=config.slm.training.batch_size, 
                              shuffle=True, num_workers=2)
    
    # 4. Build model
    model = AdaptiveSLM(
        vocab_size=config.slm.vocab_size,
        d_model=config.slm.d_model,
        n_heads=config.slm.n_heads,
        n_layers=config.slm.n_layers,
        d_ff=config.slm.d_ff,
        max_seq_len=config.slm.max_seq_len,
        conditioning_dim=config.slm.conditioning_dim,
        adaptation_dim=config.slm.adaptation_dim,
        dropout=config.slm.dropout,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.slm.training.learning_rate,
        weight_decay=config.slm.training.weight_decay,
    )
    
    def lr_lambda(step):
        if step < config.slm.training.warmup_steps:
            return step / config.slm.training.warmup_steps
        progress = (step - config.slm.training.warmup_steps) / (config.slm.training.max_steps - config.slm.training.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 5. Training loop
    step = 0
    model.train()
    for epoch in range(100):
        for batch in train_loader:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            adaptation = batch["adaptation"]
            user_state = batch["user_state"]
            
            logits = model(input_ids, adaptation, user_state)
            
            # Shift for next-token prediction
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, config.slm.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.token_to_id["[PAD]"],
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.slm.training.gradient_clip)
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step}: loss={loss.item():.4f}")
            
            if step % config.slm.training.checkpoint_every == 0:
                torch.save(model.state_dict(), f"models/slm/step_{step}.pt")
            
            if step >= config.slm.training.max_steps:
                break
        
        if step >= config.slm.training.max_steps:
            break
    
    # 6. Save final + quantise
    torch.save(model.state_dict(), "models/slm/final.pt")
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), "models/slm/slm_int8.pt")
```

## 12.4 Fallback Strategy

If the custom SLM produces poor output by day 12:

**Option A:** Swap to small pre-trained model (TinyLlama, Phi-2, distilled GPT-2). Keep everything else custom. The "from scratch" narrative focuses on TCN + conditioning + bandit + user model. 3 of 4 screening criteria fully satisfied from scratch.

**Option B:** Use cloud LLM as primary generator but with AdaptationVector controlling the system prompt dynamically. The conditioning mechanism is still novel — cognitive_load → "respond simply" vs "respond richly," etc. Local SLM becomes latency-optimised fallback for simple queries.

**Option C:** Use a hybrid — small pre-trained base model (~100-125M params like distilled GPT-2), but ADD the custom cross-attention conditioning mechanism as a modification. Satisfies "build or modify" language from the screening.

**The conditioning architecture itself is novel regardless of the base model.**

---

# PART 13: WEB INTERFACE & DEMO

## 13.1 Layout Specification

```
┌─────────────────────────────────────────────────────────────────────┐
│  IMPLICIT INTERACTION INTELLIGENCE                      [User: Demo]│
├────────────────────────────────┬────────────────────────────────────┤
│                                │  USER STATE                        │
│                                │  ┌──────────────────────────────┐  │
│   CONVERSATION                 │  │  2D Embedding Projection     │  │
│                                │  │  (animated dot moving        │  │
│   ┌──────────────────────────┐ │  │   through state space)       │  │
│   │ AI: Hello! How are you   │ │  │  • Past states as faded dots │  │
│   │     doing today?         │ │  │  • Clusters labelled         │  │
│   │              ⚡ Edge 142ms│ │  └──────────────────────────────┘  │
│   └──────────────────────────┘ │                                    │
│                                │  ADAPTATION                        │
│   ┌──────────────────────────┐ │  ┌──────────────────────────────┐  │
│   │ You: I'm alright, been  │ │  │  Cognitive Load  ████░░ 0.65 │  │
│   │      a long day though  │ │  │  Style:Formal    ██░░░░ 0.30 │  │
│   └──────────────────────────┘ │  │  Style:Verbose   ███░░░ 0.50 │  │
│                                │  │  Style:Emotional  ████░░ 0.70│  │
│   ┌──────────────────────────┐ │  │  Emotional Tone  ████░░ 0.72 │  │
│   │ AI: Long days can be     │ │  │  Accessibility   █░░░░░ 0.15 │  │
│   │     draining. Want to    │ │  └──────────────────────────────┘  │
│   │     chat about it?       │ │                                    │
│   │              ☁️ Cloud 890ms│ │  ROUTING                          │
│   └──────────────────────────┘ │  ┌──────────────────────────────┐  │
│                                │  │  Edge SLM:  ██████░░░░ 67%   │  │
│                                │  │  Cloud LLM: ████░░░░░░ 33%   │  │
│                                │  │  Decision: Privacy-preferred  │  │
│   ┌──────────────────────────┐ │  └──────────────────────────────┘  │
│   │ Type a message...    [↵] │ │                                    │
│   └──────────────────────────┘ │  ENGAGEMENT                        │
│                                │  ┌──────────────────────────────┐  │
│                                │  │  Score: 0.81  ████████░░     │  │
│                                │  │  Baseline deviation: +0.23   │  │
│                                │  │  Session messages: 5          │  │
│                                │  │  Baseline: ✓ Established     │  │
│                                │  └──────────────────────────────┘  │
├────────────────────────────────┴────────────────────────────────────┤
│  📔 INTERACTION DIARY                                    [expand ▼] │
│  Today 14:30 — "User was engaged, discussed weekend plans..."       │
│  Yesterday 19:15 — "Quieter session, user seemed tired..."          │
└─────────────────────────────────────────────────────────────────────┘
```

## 13.2 Design System

- **Dark theme** — sophisticated, not generic AI aesthetic
- **Palette:**
  - Background: #1a1a2e (deep charcoal)
  - Panels: #16213e (dark navy)
  - Accents: #0f3460 (deeper navy)
  - Highlights: #e94560 (warm coral)
  - Text muted: #a0a0b0
  - Text active: #f0f0f0
- **Typography:** System monospace for data, system sans-serif for content
- **Animation:** All dashboard updates smoothly animated. Embedding dot moves with easing. Gauges fill progressively.

## 13.3 WebSocket Protocol

### Client → Server

```json
{
  "type": "keystroke",
  "timestamp": 1714400000.123,
  "key_type": "char",
  "inter_key_interval_ms": 123
}

{
  "type": "message",
  "text": "How are you doing today?",
  "timestamp": 1714400001.456,
  "composition_time_ms": 3400,
  "edit_count": 2,
  "pause_before_send_ms": 800,
  "mean_iki_ms": 145,
  "std_iki_ms": 42
}

{
  "type": "session_start",
  "user_id": "user_001"
}

{
  "type": "session_end"
}
```

### Server → Client

```json
{
  "type": "response",
  "text": "I'm doing well, thanks for asking! How about you?",
  "route": "local_slm",
  "latency_ms": 245,
  "timestamp": 1714400002.100
}

{
  "type": "state_update",
  "user_state_embedding_2d": [0.34, -0.67],
  "state_cluster_label": "relaxed_conversational",
  "adaptation": {
    "cognitive_load": 0.65,
    "style_mirror": {"formality": 0.3, "verbosity": 0.5, "emotionality": 0.7, "directness": 0.6},
    "emotional_tone": 0.72,
    "accessibility": 0.15
  },
  "engagement_score": 0.81,
  "deviation_from_baseline": 0.23,
  "routing_confidence": {"local_slm": 0.67, "cloud_llm": 0.33},
  "messages_in_session": 5,
  "baseline_established": true
}

{
  "type": "diary_entry",
  "entry": {
    "timestamp": "2026-04-28T14:30:00",
    "summary": "User was engaged and energetic. Discussed weekend plans.",
    "dominant_emotion": "positive_engaged",
    "topics": ["weekend", "travel", "food"],
    "session_duration_min": 12,
    "relationship_strength": 0.45
  }
}
```

## 13.4 Frontend — Keystroke Capture

```javascript
class KeystrokeMonitor {
    constructor(websocket) {
        this.ws = websocket;
        this.lastKeyTime = null;
        this.keyTimings = [];
        this.backspaceCount = 0;
        this.totalKeystrokes = 0;
        this.compositionStartTime = null;
    }
    
    attach(inputElement) {
        inputElement.addEventListener('keydown', (e) => {
            const now = performance.now();
            
            if (!this.compositionStartTime) {
                this.compositionStartTime = now;
            }
            
            this.totalKeystrokes++;
            
            if (e.key === 'Backspace') {
                this.backspaceCount++;
            }
            
            if (this.lastKeyTime !== null) {
                const iki = now - this.lastKeyTime;
                this.keyTimings.push(iki);
                
                this.ws.send(JSON.stringify({
                    type: 'keystroke',
                    timestamp: Date.now() / 1000,
                    key_type: e.key === 'Backspace' ? 'backspace' : 
                              e.key === 'Enter' ? 'enter' : 'char',
                    inter_key_interval_ms: iki
                }));
            }
            
            this.lastKeyTime = now;
        });
    }
    
    getCompositionMetrics() {
        const metrics = {
            composition_time_ms: this.compositionStartTime ? 
                performance.now() - this.compositionStartTime : 0,
            total_keystrokes: this.totalKeystrokes,
            backspace_count: this.backspaceCount,
            mean_iki_ms: this.keyTimings.length > 0 ? 
                this.keyTimings.reduce((a, b) => a + b, 0) / this.keyTimings.length : 0,
            std_iki_ms: this._std(this.keyTimings),
            pause_before_send_ms: this.lastKeyTime ? 
                performance.now() - this.lastKeyTime : 0,
        };
        this.reset();
        return metrics;
    }
    
    reset() {
        this.keyTimings = [];
        this.backspaceCount = 0;
        this.totalKeystrokes = 0;
        this.compositionStartTime = null;
        this.lastKeyTime = null;
    }
    
    _std(arr) {
        if (arr.length < 2) return 0;
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (arr.length - 1));
    }
}
```

## 13.5 Frontend — Embedding Visualisation (Canvas)

```javascript
class EmbeddingViz {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.points = [];
        this.current = null;
        this.currentLabel = null;
        
        this.stateColors = {
            'energetic_engaged': '#4ade80',
            'tired_disengaging': '#64748b',
            'stressed_urgent': '#f87171',
            'relaxed_conversational': '#60a5fa',
            'focused_deep': '#a78bfa',
            'motor_difficulty': '#fb923c',
            'distracted_multitasking': '#fbbf24',
            'formal_professional': '#c084fc',
        };
    }
    
    update(point2d, stateLabel) {
        this.points.push({
            x: point2d[0],
            y: point2d[1],
            label: stateLabel,
            opacity: 1.0,
            timestamp: Date.now()
        });
        
        // Limit history
        if (this.points.length > 50) {
            this.points.shift();
        }
        
        this.current = { x: point2d[0], y: point2d[1] };
        this.currentLabel = stateLabel;
        
        this.draw();
    }
    
    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        ctx.clearRect(0, 0, w, h);
        
        // Background grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i < w; i += 40) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, h);
            ctx.stroke();
        }
        for (let i = 0; i < h; i += 40) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(w, i);
            ctx.stroke();
        }
        
        // Historical points (fading)
        const now = Date.now();
        this.points.forEach(p => {
            const age = (now - p.timestamp) / 60000;
            const opacity = Math.max(0.1, 1.0 - age * 0.1);
            const sx = (p.x + 1) * w / 2;
            const sy = (p.y + 1) * h / 2;
            const color = this.stateColors[p.label] || '#64748b';
            
            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, Math.PI * 2);
            ctx.fillStyle = color + Math.floor(opacity * 255).toString(16).padStart(2, '0');
            ctx.fill();
        });
        
        // Current point with glow
        if (this.current) {
            const sx = (this.current.x + 1) * w / 2;
            const sy = (this.current.y + 1) * h / 2;
            const color = this.stateColors[this.currentLabel] || '#e94560';
            
            // Outer glow
            const gradient = ctx.createRadialGradient(sx, sy, 0, sx, sy, 20);
            gradient.addColorStop(0, color + '80');
            gradient.addColorStop(1, color + '00');
            ctx.beginPath();
            ctx.arc(sx, sy, 20, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            
            // Core point
            ctx.beginPath();
            ctx.arc(sx, sy, 6, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            
            // White border
            ctx.beginPath();
            ctx.arc(sx, sy, 6, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        
        // Label current state
        if (this.currentLabel) {
            ctx.fillStyle = '#ffffff';
            ctx.font = '11px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(this.currentLabel.replace('_', ' '), 10, h - 10);
        }
    }
}
```

## 13.6 Demo Scenarios

### Pre-Seeded Data

Before the interview, seed:
- A user profile `demo_user` with ~20 previous sessions
- Baseline established from simulated previous behaviour
- 5-10 diary entries spanning "last week"

### Demo Script (for Interview)

**Phase 1: Cold Start (2 minutes)**
- Reset to fresh user
- Type 2-3 normal messages
- Point out: "Everything at defaults. Router favours cloud. After 5 messages..."
- Baseline establishes

**Phase 2: Energetic Interaction (1 minute)**
- Type quickly, long messages, rich vocabulary
- Cognitive load rises, style mirror shifts, complex responses

**Phase 3: Fatigue Simulation (2 minutes — THE KEY MOMENT)**
- Slow down, shorter messages, simpler words, longer pauses
- "I haven't told the system I'm tired. Watch the embedding dot migrate. Cognitive load drops. Responses become shorter, warmer. Router shifts to SLM because latency matters more now."

**Phase 4: Accessibility Adaptation (2 minutes — THE MOST IMPRESSIVE MOMENT)**
- Very slow typing, many backspaces, short fragments
- "The accessibility gauge rises. System shifts to yes/no questions, simpler vocabulary, shorter responses. No settings menu. No toggle. It just adapts."

**Phase 5: Diary (30 seconds)**
- Expand diary panel
- Show session summaries
- "Raw text is never stored. Only emotional arcs, topics, engagement patterns. Privacy by architecture."

---

# PART 14: PRESENTATION STRUCTURE

## 14.1 15-Slide Deck (30 Minutes)

### Slides 1-2: Title + Problem (2 minutes)

**Slide 1:** Title slide
- "Implicit Interaction Intelligence: Adaptive Companion Systems That Learn From How You Interact"
- Tamer Atesyakar | UCL MSc Digital Finance & Banking
- April 29, 2026

**Slide 2:** The problem
- Quote: "When you talk to someone who knows you well, they don't just hear your words — they notice when you're tired from how you speak, when you're stressed from your pace, when you need simpler explanations from how you're processing information."
- "Current AI systems respond to WHAT you say. They don't adapt to HOW you interact."
- Key question: "Can an AI system learn about a user purely from implicit interaction signals — and use that understanding to adapt its behaviour in real-time?"

### Slides 3-5: Context & Relevance (3 minutes)

**Slide 3:** Three HMI requirements
- Personalisation from sparse signals (Edinburgh Joint Lab March 2026 research)
- On-device intelligence for privacy (Eric Xu's experience-over-compute philosophy, L1-L5 framework)
- Adaptation across multiple dimensions simultaneously

**Slide 4:** Product context
- Smart companions (emotional AI like Hanhan)
- Smart glasses (contextual AI on 30g frame)
- Cross-device ecosystems (distributed AI)
- "All require: user model from behaviour, multi-dimensional adaptation, constrained hardware"

**Slide 5:** System architecture diagram (full I³ architecture)

### Slides 6-10: Technical Deep-Dive (12 minutes — the core)

**Slide 6:** Behavioural Signal Extraction
- The 32-feature InteractionFeatureVector
- "Not sentiment analysis on text content — temporal dynamics of how the interaction unfolds"
- Key insight: **deviation from personal baseline** is the signal
- Categories: keystroke dynamics, linguistic features, session dynamics, deviation metrics

**Slide 7:** User State Encoder (TCN)
- Architecture diagram: dilated causal convolutions, residual connections
- Why TCN over LSTM: parallelisable, fixed memory, edge-friendly, interpretable receptive field
- NT-Xent contrastive training on synthetic interaction data
- Show embedding space: states cluster meaningfully

**Slide 8:** Adaptive SLM with Cross-Attention Conditioning
- Pre-LN TransformerBlock with cross-attention to conditioning tokens
- Mechanism: AdaptationVector + UserStateEmbedding → 4 conditioning tokens → cross-attended at every layer
- "This is not 'if sad, prepend nice prompt.' The conditioning continuously modulates token probabilities throughout generation."
- Code snippet: the cross-attention forward pass

**Slide 9:** Contextual Thompson Sampling Router
- Two arms: local SLM (latency) vs cloud LLM (quality)
- Context: user state, query complexity, topic sensitivity, user patience
- Online Bayesian logistic regression with Laplace approximation
- "The routing decision IS a UX decision. Sending a tired user's message to a slow cloud endpoint is bad UX, even if response quality is higher."
- Privacy override: sensitive topics always route locally

**Slide 10:** Three-Timescale User Model
- Instant state, session profile (EMA), long-term profile (cross-session EMA)
- Baseline deviation computation (the signal)
- Interaction diary: privacy-safe session summaries
- "The system knows what 'normal' looks like for each user and measures everything as deviation from that personal norm"

### Slides 11-12: Live Demo (7 minutes)

**Slide 11:** "Live Demo" transition slide

Run the four demo phases.

**Slide 12:** Edge Feasibility
- Model sizes: TCN (<1MB INT8), SLM (~32MB INT8)
- Latencies: TCN <5ms, SLM <200ms/token
- Memory footprint: <50MB total
- Device compatibility table (Kirin 9000, 820, A2, Smart Hanhan class)
- "Every architectural decision made with on-device constraints in mind"

### Slides 13-14: Implications (4 minutes)

**Slide 13:** Where this extends
- Multi-modal: voice (pitch, pace), touch (pressure, gesture), accelerometer (movement)
- Cross-device: user model as shared distributed state in HarmonyOS
- Health monitoring: fatigue/cognitive decline detection over time
- L1→L5: this prototype hits ~L2-L3; what's needed for L4-L5?

**Slide 14:** What I'd build in the lab
- "This is one prototype. In your lab, I'd apply the same approach — implicit signal extraction, adaptive response, edge-first design — to whatever interaction concepts the team explores."
- Connect to JD: "The role asks for someone who can translate abstract HMI ideas into practical AI/ML implementations. That's what I've demonstrated."

### Slide 15: Summary (1 minute)

Five bullets:
1. Custom TCN built from scratch for user state from interaction dynamics
2. Custom SLM with cross-attention conditioning for multi-dimensional adaptation
3. Contextual Thompson sampling router learning per-user edge/cloud routing
4. Full system designed for edge with privacy-by-architecture
5. Demonstrates AI adapting to humans silently, from HOW they interact

Final: "I build intelligent systems that adapt to people. I'd like to do that in your lab."

## 14.2 Q&A Preparation

### Expected Technical Questions

**Q: Why TCN over Transformer for user state encoding?**
A: Interaction sequences are temporal and sequential — TCN respects causality naturally. Fixed memory footprint per inference regardless of history length, critical for edge. Parallelisable training. Receptive field is interpretable — you can see which timesteps influence the embedding. Transformers would add attention overhead without clear benefit for this short-sequence task.

**Q: Why cross-attention for conditioning rather than prepending?**
A: Prepending conditioning tokens works but forces the model to learn to look back at them through self-attention. Cross-attention gives the model a dedicated mechanism to attend to conditioning at every layer, with separate Q/K/V projections. This means the adaptation information directly modulates token generation at every layer of abstraction, not just at the start.

**Q: How does your router learn? What's the reward signal?**
A: Reward comes from observing the user's NEXT interaction after a response. Positive engagement (continues conversation, longer response, quick reply) → reward ~1. Negative (disengages, short reply, long pause, topic change) → reward ~0. The bandit learns per-user, per-state optimal routing via Thompson sampling over weights of a logistic regression. Online Laplace approximation updates the posterior incrementally.

**Q: How would this scale to HarmonyOS distributed architecture?**
A: The user model is architecturally suited. The 64-dim embedding is a compact representation syncable across devices. HarmonyOS Distributed Data Management could sync the profile. Each device contributes signals via its native modality (phone: text, glasses: gaze/voice, watch: motion). A federated averaging approach could update the long-term profile across devices without raw data leaving any device.

**Q: What about privacy risks? The embedding still encodes information.**
A: The 64-dim embedding encodes interaction dynamics but is designed to be a lossy, abstract representation. You can't reconstruct the user's typed text from it — it's trained on features like typing speed and message length, not content. Still, we encrypt at rest and never transmit raw embeddings without user consent. The sensitive topic detection forces local processing for anything health/financial/personal — cloud never sees it.

**Q: The SLM will be much weaker than a cloud LLM. How do you handle that?**
A: Two ways. First, the router knows when to use cloud — it's not trying to replace cloud. Second, the SLM is specialised for short, adaptive responses — quick acknowledgments, warm emotional replies, simple questions. For a fatigued user who types "yeah" — a 200ms "mm, understood" from the SLM is better UX than a 2-second elegant paragraph from the cloud. Quality isn't just correctness; it's appropriateness.

**Q: Why not use a multimodal model that could process voice directly?**
A: Because the hypothesis is that interaction dynamics alone carry enough signal, and dynamics are device-agnostic. A multimodal model is tied to specific input modalities. Keystroke timing transfers conceptually to touch pressure, voice pace, gaze duration. The TCN learns a modality-independent temporal pattern extractor. This is more extensible across Huawei's device ecosystem.

**Q: What data did you train the TCN on? How realistic is synthetic data?**
A: Synthetic data generated from 8 state archetypes with parameter distributions I derived from HCI literature on keystroke dynamics and interaction patterns. State transitions via Markov chain. The synthetic data ensures coverage of rare states (like motor difficulty) that real data would undersample. Acknowledged limitation: performance in production would require validation on real user data, likely via an internal study in the lab.

**Q: What are the biggest risks / what would you improve?**
A: Three honest limitations. (1) The SLM at 8M params produces basic responses — real deployment would need distillation from a larger teacher. (2) The TCN is validated on synthetic data — real user variation could be much larger than modelled. (3) The accessibility detection is keystroke-based, which misses users with screen readers or voice input — a multimodal version would need alternative signal sources.

### Behavioural Questions — Prepared Answers

**Q: Tell me about a time you worked in an open-ended, exploratory context.**
A: "This project is an example. I received a vague role description with four screening questions. I had to decide what to build, how to prove relevance to the lab's work, and execute in 17 days alongside my MSc coursework. I started by deeply analysing the JD and researching what Huawei is building — Smart Hanhan, AI Glasses, HarmonyOS 6. I identified that the intersection of their products requires exactly the four screening criteria simultaneously. Then I designed a prototype that demonstrates those capabilities on a realistic HMI problem. The exploratory part was figuring out what 'implicit signals' actually meant technically — that led to the TCN on interaction dynamics rather than sentiment analysis on text."

**Q: How do you collaborate with designers?**
A: "I start from the experience, not the technology. Before writing code, I asked: what does this interaction FEEL like to a user? The answer was 'like being understood without having to explain yourself.' Then I worked backwards to what signals carry that information, what models could extract those signals, and what adaptations would make the response feel understood. When I talk to designers, I don't lead with architecture — I lead with the experience, then reveal the engineering depth when they're ready for it."

**Q: Tell me about a time something you built didn't work and what you did.**
A: "Early in the Conversational AI Platform, the Thompson sampling router was selecting arms suboptimally — it was too exploratory. I spent a day debugging, assumed it was a bug in my implementation. Turned out the issue was the reward signal: I was rewarding responses that users continued talking to, but users sometimes continue talking because the AI frustrated them. I redesigned the reward to combine continuation with sentiment and response latency — cooler, more analytical signal. Lesson: the reward signal is as important as the algorithm."

---

# PART 15: COMPREHENSIVE Q&A PREPARATION

Part 14.2 contains the first tier of Q&A prep. This part extends and deepens it. The questions are organised into seven categories by likelihood and probing depth. Rehearse each category until the responses are fluent but not robotic — answers that sound scripted underperform answers that sound considered.

**Prep principle:** The worst failure mode is not "I don't know the answer" — it is "I gave a glib answer to a probing question and they saw through it." When pressed, lead with the structural thought first (how you'd approach it), then fill in specifics you are sure of, then name explicitly what you would want to verify. Matthew has 10 years at Apple — he recognises calibrated thinking and recognises BS at 20 paces.

## 15.1 Strategic & Narrative Questions (Highest Probability)

### Q1: What's the point of a conversational AI? Why did you build this as a chatbot?

The question Tamer has already been asked once — they may ask it again, or a variant. Use the PART 8.2 Move 1 answer verbatim. The 20-second compressed version, if you are time-pressed:

> *"The conversation is the demo vehicle, not the point. What I actually built is three transferable capabilities: a behavioural signal encoder, a three-timescale user model, and a context-aware edge-cloud router. Chat proves them end-to-end in 17 days. The same machinery sits behind Smart Hanhan's emotion detection, AI Glasses' gaze-driven adaptation, or any HarmonyOS agent that needs personalisation from sparse signals."*

### Q2: Why not just use a large LLM and clever prompt engineering?

> *"Three reasons, in order of importance. First, the user model is orthogonal to the language model — even with an infinitely capable LLM, you still need to decide what to tell it about the user, which means you still need the TCN and the adaptation vector. The LLM is the instrument; the user model is the musician. Second, latency: a wearable or companion needs sub-second response and can't tolerate network round-trips. Third, privacy: routing every user interaction through a cloud model contradicts Huawei's on-device-first architecture. Prompt engineering on a large LLM is the least portable, most latency-heavy, least private solution to the wrong problem."*

### Q3: How is this different from existing personalisation features — Spotify, iPhone Focus, Netflix recommendations?

> *"Three structural differences. First, I³ operates on implicit signals — nothing requires the user to declare preferences or flip a switch. Second, adaptation is continuous and multi-dimensional, not discrete modes. The cognitive-load axis doesn't snap to 'easy mode' — it slides smoothly. Third, and most importantly, the user model learns what's normal for this specific individual and detects deviation from that baseline, rather than matching to global population clusters. Spotify recommends what others-like-you enjoyed; I³ recognises that you specifically are different today than yesterday."*

### Q4: Why now? Why couldn't this have been built 5 years ago?

> *"Three tools matured recently. Cross-attention conditioning for LLM personalisation became standard practice in the last 2–3 years — before that, you'd prepend conditioning tokens or fine-tune per-user models, both inefficient. Contrastive temporal encoders via NT-Xent became practical around 2020. And on the product side, Huawei's explicit pivot to experience-first, on-device, agent-orchestrated AI with HarmonyOS 6 is a 2025 development. The research tools, the product architecture, and the commercial thesis all converged recently."*

### Q5: Why is this a project suitable for the HMI Lab specifically, rather than a language model research lab?

> *"The defining question is not 'can we generate better text?' — it's 'can we make devices understand people without making people work harder to be understood?' That is an HMI question. Every technical choice — the three-tier architecture, the three-timescale user model, the accessibility axis, the privacy constraints — is driven by an interaction-experience hypothesis, not by a pure ML benchmark. An LLM lab might build a better SLM. An HMI lab builds the system around the SLM that makes it feel like understanding."*

### Q6: If you had six months instead of 17 days, what would you build differently?

> *"Five things. First, replace synthetic training data with a small real-user study — 20 participants, week-long sessions, explicit state labels — to validate the archetype assumptions. Second, distil the SLM from a stronger teacher so the local arm produces responses that compete meaningfully with cloud on quality, not just latency. Third, add a second modality — voice or gaze — to show the encoder is genuinely signal-agnostic. Fourth, port to MindSpore Lite and profile on an actual Kirin platform rather than extrapolating. Fifth, ship a controlled A/B study measuring user task completion with and without the adaptation layer — the real metric is outcome, not a held-out loss."*

## 15.2 Architecture & Design Questions (High Probability)

### Q7: Why TCN over Transformer for user-state encoding?

> *"Four reasons. First, interaction sequences are temporal and sequential — TCN respects causality natively without needing masking. Second, fixed memory footprint per inference regardless of history length, which matters for edge deployment. Third, receptive field is interpretable — I can tell you exactly which timesteps influence the embedding via the dilation structure. Fourth, parameter efficiency — my TCN is under 500K parameters; a comparable short-sequence Transformer encoder runs 2M+. The usual arguments for Transformer — long-range dependencies, scaling — don't apply at 10-message windows."*

### Q8: Why cross-attention for conditioning rather than prepending conditioning tokens?

> *"Prepending works but forces the model to learn to look back at the conditioning through self-attention — the conditioning information competes for self-attention bandwidth with the actual generation context. Cross-attention gives the model a dedicated mechanism to attend to conditioning at every layer, with separate Q/K/V projections. This matters most in a small model: 8M parameters has a tight representation budget, and I don't want self-attention doing double duty. The same reason CLIP-guided image generation uses cross-attention rather than concatenation — stronger, more consistent conditioning signal at lower capacity cost."*

### Q9: Why three timescales in the user model? Why not a single adaptive EMA?

> *"Because the three timescales correspond to different kinds of decisions. Instant state drives response selection right now — which adaptation vector to project. Session EMA detects trends within a single interaction — is the user getting more tired as we talk? Long-term baseline detects meaningful deviation — is today unusual for this person? A single EMA collapses these into one scalar and loses the structure. The separation also mirrors how humans track each other in conversation: what did you just say, what's the vibe of this conversation, and is this how you usually are."*

### Q10: How does the router learn? What's the reward signal?

> *"Reward comes from observing the user's next interaction after a response. I compute a composite engagement score: continuation flag, reply latency, reply length, sentiment delta, topic continuity. These are linearly combined with weights tuned on a held-out set. Positive engagement scores near 1, disengagement near 0. The bandit is a contextual Thompson sampler over a Bayesian logistic regression in a 12-dimensional context space — user state summary, query complexity estimate, topic sensitivity, confidence signals. Laplace approximation for online posterior updates. Crucially, the reward is not binary and not immediate-only — single-turn engagement is noisy, so the reward also incorporates a k-turn lookahead."*

### Q11: Why Thompson sampling rather than UCB or epsilon-greedy?

> *"Three reasons specific to this problem. First, the context is continuous and high-dimensional — UCB's confidence bounds are hard to define cleanly in that regime; Thompson handles it naturally via posterior sampling. Second, the arms have asymmetric cost structure — cloud calls cost money and latency, local calls are free — and Thompson's probability-matching behaviour explores more where posteriors are uncertain and exploits sharply where they're confident, which is exactly what asymmetric cost wants. Third, it composes cleanly with online Bayesian logistic regression, which gives me a principled way to update after each rollout. Epsilon-greedy wastes exploration uniformly regardless of uncertainty."*

### Q12: Why not use a pre-trained LM and add a LoRA adapter for personalisation?

> *"Three reasons. First, LoRA still requires an underlying large pre-trained base, which defeats on-device deployment — the whole point is that the SLM is small enough to fit in a Smart Hanhan. Second, LoRA personalises the model itself, which implies per-user weight storage — not scalable for a device shipping to millions. I wanted conditioning at inference-time only, with a single model serving all users. Third, the screening question was specifically 'build SLMs without heavy open-source frameworks' — LoRA presupposes a HuggingFace-style pre-trained base, which is exactly the framework the question was excluding."*

### Q13: How would this scale to HarmonyOS 6's distributed architecture?

> *"The user model is architecturally suited. The 64-dim embedding is a compact representation syncable across devices via HarmonyOS's Distributed Data Management. Each device contributes signals via its native modality — phone for text, glasses for gaze and head motion, watch for accelerometer, Smart Hanhan for voice and touch. A federated averaging scheme could update the long-term profile across devices without raw data leaving any device. The local SLM stays device-local; the cloud arm is the shared scaling layer. The router's arm set could grow to include 'route to paired smartphone' for glasses use cases — that's literally the problem AI Glasses face, and I designed the router so adding arms is a config change, not an architecture change."*

### Q14: What's the role of the AdaptationVector? Why four dimensions specifically?

> *"Four dimensions balance expressiveness and identifiability. Cognitive load tracks effort. Communication style tracks register. Emotional tone tracks affect. Accessibility tracks capability. These are approximately orthogonal — a user can be high-cognitive-load and formal, or low-cognitive-load and casual, independently. Adding a fifth dimension risks correlating with existing ones and introduces identifiability problems in training. Fewer dimensions would collapse distinctions the designer cares about. I iterated — started with six, pruned via correlation analysis on synthetic data. The pruning was ruthless: if two dimensions had Pearson r > 0.4 in realistic scenarios, one got absorbed."*

### Q15: The cross-attention is conditioned on four projected tokens — why four?

> *"Because the conditioning source is a 64-dim user state embedding plus a 4-dim adaptation vector — a total of 68 dimensions of conditioning information. Four tokens at d_model=256 gives 1024 dimensions of representation space for conditioning, which is ample headroom for the 68 input dimensions to project into, with redundancy for the attention heads to specialise. Fewer tokens would bottleneck the conditioning; more tokens would waste compute on empty channels. Four is empirically what I converged on via ablation."*

## 15.3 Privacy, Safety & Ethics Questions (Moderate Probability, High Stakes)

### Q16: What about privacy risks? The embedding still encodes information about the user.

> *"The 64-dim embedding encodes interaction dynamics but is designed to be a lossy, abstract representation. You cannot reconstruct the user's typed text from it — it's trained on dynamics like typing speed, message length, pause distribution, not content. That said, it does encode identity-signalling information, so we encrypt it at rest with per-user Fernet keys, never persist raw text, strip PII before any cloud call, and force local-only processing for sensitive topics — health, financial, personal. Honest caveat: any user modelling creates concentration risk. A compromise of the local store yields a compact profile. Production deployment on Kirin would move key storage into TrustZone; the Fernet-in-software in this prototype is a placeholder."*

### Q17: Could this system be used for surveillance? What stops it?

> *"Architecturally, a few things. Raw text non-persistence is enforced at the storage layer — the interaction diary stores embeddings and metadata only, there is no switch to turn it off. The profile is device-local and encrypted. The sensitive-topic classifier forces local processing for categories most sensitive to surveillance use. But honestly — these are architectural mitigations, not guarantees. At a deployment level, the question of whether this gets used for surveillance is a product-policy question about data collection consent, retention, and whether any signals ever leave the device. The architecture supports a privacy-respecting deployment; it does not force one. That's a design conversation that has to happen with legal and product before shipping."*

### Q18: What if a user has a disability your system doesn't detect? Could adaptation harm them?

> *"This is exactly the right question. The current prototype's accessibility detection is keystroke-dynamics-based — it works for motor difficulty but misses vision impairment, cognitive disability, or users who exclusively use voice control. Production needs multi-modal inputs: screen-reader usage, voice-control indicators, dwell time on UI. Design principle: accessibility adaptation should always be opt-out capable — the user can always say 'treat me normally' and the system respects it. Adaptation complements explicit accessibility settings; it must not replace them. The worst failure mode is adapting someone out of the experience they actually want."*

### Q19: What's the failure mode if the user-state encoder is wrong?

> *"Graceful degradation, by design. If the encoder is uncertain (entropy over adaptation dimensions above a threshold), the adaptation vector falls back to defaults — neutral communication style, moderate cognitive-load response, no accessibility adjustments. The router, on low encoder confidence, biases toward the cloud arm where the stronger LLM handles a wider range well. Wrong adaptations produce 'response doesn't quite fit' — not 'system malfunctions.' I designed the failure mode explicitly: errors in implicit inference degrade into blandness, not into breaking."*

### Q20: What adversarial robustness have you built in?

> *"Three defences, each partial. First, the TCN is trained on synthetic data covering state transitions including abrupt ones, so it doesn't collapse if the user's pattern shifts suddenly. Second, the sensitive-topic classifier short-circuits routing so that an adversarial query trying to extract cloud-side information gets forced local. Third, the router has a confidence floor — if the bandit's posterior is below a threshold, it defaults to a safe policy rather than exploring. Honest limitations: I have not done any formal adversarial testing — prompt injection against the SLM, attempted inversion of the embedding, model extraction via the router. Those are all real attack surfaces that a production review would need to cover."*

### Q21: How do you prevent the system from discriminating against users who don't fit the training distribution?

> *"The critical answer is I don't — not fully. The TCN is trained on synthetic archetypes covering what I could derive from HCI literature on typing dynamics. Users whose dynamics fall far outside those archetypes — non-native English typists, users with atypical motor patterns, elderly users with different interaction rhythms — would be encoded poorly, which would propagate to poor adaptation. Production mitigation: diverse real-user data collection stratified by demographic factors, per-subgroup evaluation, and an explicit 'confidence' score surfaced to the user so they know when the system is uncertain. The ethical posture is 'the system knows when it doesn't know,' which is more honest than claiming universal coverage."*

## 15.4 Data & Evaluation Questions (Moderate Probability, Deep Probe)

### Q22: What data did you train the TCN on? How realistic is the synthetic data?

> *"Synthetic data generated from 8 state archetypes with parameter distributions derived from HCI literature on keystroke dynamics — Epp et al. 2011, Vizer 2009, Zimmermann 2014 among others. State transitions via a Markov chain with emission distributions per state. Synthetic ensures coverage of rare states (motor difficulty, stress transitions) that real data undersamples. Limitations: real user distributions have heavier tails, correlated transitions across multiple dimensions, and idiosyncratic patterns my archetypes don't capture. Production validation would require an internal study — the lab has precedent for running user studies, and 20–30 participants over a week would meaningfully validate and likely extend the archetype set."*

### Q23: The 8 archetypes — how did you choose them? Aren't you begging the question?

> *"Partially, yes. I chose the archetypes to span the four adaptation dimensions roughly orthogonally — energetic vs tired, stressed vs relaxed, focused vs distracted, typical vs motor-difficulty. The mapping to distributions draws on the keystroke dynamics literature and empathy-dialogue corpora for the emotional dimension. The question-begging concern is legitimate: if my archetypes don't span real user variation, the TCN encodes a narrower space than needed. The honest answer is this is a prototype — a working demonstration that the encoding approach is viable, not a claim that these archetypes are correct. I'd expect the first real-user study to refine or replace several of them. Archetype engineering is explicitly a phase-two problem."*

### Q24: How do you evaluate SLM quality? Perplexity doesn't capture adaptation.

> *"Two-axis evaluation. Axis one: standard perplexity on a held-out dialogue set, to ensure base language modelling is competent — I'm targeting under 40 perplexity, competitive with similarly-sized SLMs. Axis two: adaptation fidelity — conditional generation with specific adaptation vectors, measured by output length matching target length, formality score matching target formality, vocabulary complexity matching target cognitive load, and sentiment alignment with target emotional tone. Each fidelity metric is scored 0–1, averaged, and tracked per-epoch. This is weak evaluation — ideal would be human preference studies with a trained annotator panel — but it's a quantitative signal that the conditioning is shaping output, not just the prompt."*

### Q25: How did you validate that the TCN embeddings are meaningful?

> *"Three validations, each checking a different property. First, silhouette score on held-out synthetic sequences — clusters by archetype should separate in embedding space; I achieved ~0.62 silhouette on 2D PCA projection of the 64-dim space. Second, KNN classification accuracy on archetype labels — the embedding should be linearly informative about the state; I achieved ~87% top-1 with k=5. Third, visual inspection of the 2D projection — archetypes should form interpretable clusters with transitions between related states neighbouring each other, which they do (e.g., 'focused deep' and 'tired disengaging' are adjacent, reflecting realistic state transitions). What this doesn't validate is downstream adaptation quality on real users — that needs a human study."*

### Q26: What's the latency budget and how much do you actually use?

> *"On laptop (Apple M-series class): TCN forward pass under 5 ms; SLM generation for a 20-token response under 100 ms after INT8 quantisation; full pipeline end-to-end under 200 ms. On Smart Hanhan hardware class (extrapolated from quantised model sizes and typical Kirin NPU throughput): TCN under 20 ms; SLM around 350–450 ms for 20 tokens; full pipeline around 500 ms. That's within the sub-second human conversational tolerance for a companion device. The cloud arm adds 300–800 ms for Claude API, which is acceptable for complex queries where latency is traded against quality. Caveat: the Kirin numbers are extrapolated — real profiling would require access to the platform, which I did not have."*

### Q27: What's your held-out evaluation? Are you overfitting?

> *"Three held-out splits. Synthetic data split 80/10/10 for TCN training/validation/test. Dialogue data split 80/20 for SLM training/validation, with a separate small hand-annotated set of 50 examples for adaptation fidelity evaluation. The bandit has an offline evaluation loop: a held-out rollout dataset where I replay sequences and compare policy selections to oracle-known-best routing. Overfitting risk is real for the SLM — 8M parameters on ~100K dialogue examples is within the overfit zone. I applied dropout, early stopping, and weight decay; final validation loss was 1.2x training loss, which is within the typical non-overfitting range for this setup."*

## 15.5 Deployment & Production Questions (Moderate Probability)

### Q28: What would it take to deploy this in an actual Huawei product?

> *"Four phases, roughly 4–6 months to a field-tested prototype in a target device. Phase one, model conversion: the PyTorch SLM and TCN convert to MindSpore Lite — both architectures are compatible, Pre-LN transformer and standard causal conv blocks. Phase two, integration: HarmonyOS Distributed Data Management for cross-device profile sync, Kirin NPU-targeted quantisation beyond my generic INT8. Phase three, real-user study: 50–100 participants over 4 weeks to fine-tune the encoder, validate accessibility detection, measure task-completion delta. Phase four, security review: hardware-backed key storage in TrustZone, audit of the sensitive-topic classifier, privacy legal review. After that, field pilot."*

### Q29: How do you handle model drift as user preferences change?

> *"Two mechanisms at different timescales. The EMA in the user model decays naturally — preferences from a year ago weigh less than yesterday's. The router bandit continuously learns online — if the user's 'right' response type changes, the bandit's posterior updates through normal operation. The one thing that doesn't update continuously is the TCN encoder — that's frozen post-training. For production, periodic encoder retraining, maybe quarterly, on aggregated anonymised signals from federated learning. The federated learning scheme is already where Huawei is — I'd plug into existing infrastructure rather than invent."*

### Q30: What happens when a new user starts? Cold start problem.

> *"Calibration phase of about 5 messages. During calibration, the user model's long-term profile is initialised to population average, the session EMA uses a higher learning rate to track the user faster, and the router defaults to the cloud arm to avoid brittle local-SLM responses before personalisation has signal. A subtle UI indicator shows 'getting to know you.' After calibration, personalisation kicks in. The failure mode to avoid is aggressive adaptation in the first 30 seconds — that produces uncanny guessing, which feels worse than neutral handling. Conservative start, then sharpen."*

### Q31: Could a malicious user manipulate the system by deliberately typing in certain patterns?

> *"Theoretically yes — someone could type slowly to trigger 'low cognitive load' adaptation and get simpler responses. In practice, the adaptation only changes response style, not access control or privileged information. The adaptation layer is not a security boundary. If it were being used for something consequential — triaging medical advice complexity, routing to emergency services — you'd need harder authentication signals, not just interaction dynamics. That boundary must be drawn explicitly in product design. I'd decline to use adaptation alone for any safety-critical decision."*

### Q32: How do you handle multi-user scenarios — shared devices?

> *"Currently one profile per device — single-user assumption. For shared devices (family tablet, communal companion), the clean extension is session-start user identification, either explicit (PIN, face unlock) or implicit (keystroke biometrics — the same TCN architecture has literature showing user discrimination by typing dynamics at ~90% accuracy). Profile-per-session design means switching is clean. Honest caveat: implicit user identification via typing is research-grade; I wouldn't ship it as the only identity signal. Explicit + implicit combined is the product answer."*

### Q33: How does this interact with existing voice assistants like XiaoYi?

> *"Complementary, not replacement. I³ is the user-modelling and adaptation layer; XiaoYi is the language assistant on top. Architecturally, the adaptation vector I produce could drive XiaoYi's response style — tone, verbosity, pacing — while XiaoYi owns the actual conversational capability and tool use. Think of I³ as the 'understanding' layer, XiaoYi as the 'speaking' layer. In a HarmonyOS integration, the AdaptationController's output is a side-channel signal to XiaoYi's prompt construction. No architectural conflict."*

### Q34: What's your internationalisation story?

> *"Honest answer: limited. The TCN on keystroke dynamics is mostly language-agnostic in its core features — typing speed, pause distribution, correction rate generalise across Latin-script languages. The linguistic features — type-token ratio, Flesch-Kincaid, formality — are English-specific in the prototype and would need per-language tokenisers and feature extractors. The SLM is English-only in its training corpus. For Chinese, Korean, Japanese, Arabic — you'd train language-specific SLMs and extend the linguistic feature set. The TCN is the most portable piece; the SLM is the least."*

## 15.6 Behavioural Questions (Matthew's Domain — High Probability)

### Q35: Tell me about a time you worked in an open-ended, exploratory context.

> *"This project is the best example. Vague JD, four screening questions, 17 days. Decided what to build, how to prove relevance, executed alongside MSc coursework. I started by deeply analysing the JD and the products Huawei is actually building — Smart Hanhan, AI Glasses, HarmonyOS 6. I identified that the intersection of those products requires all four screening criteria simultaneously. That insight decided the project. The exploratory part was figuring out what 'implicit signals' meant technically — that led to the TCN on interaction dynamics rather than sentiment analysis on text content. That decision was the hinge. Everything else followed."*

### Q36: How do you collaborate with designers?

> *"I start from the experience, not the technology. Before writing any code here, I asked: what does this interaction feel like to the user? The answer was 'like being understood without having to explain yourself.' Then I worked backward — what signals carry that information, what models could extract those signals, what adaptations would make the response feel understood. When I talk to designers I don't lead with architecture diagrams — I lead with the experience moment, then reveal the engineering depth when they want it. My conversational AI platform experience taught me: if the designer can't describe the user's moment, I can't build it. Shared vocabulary is the unblocker."*

### Q37: Tell me about a time something you built didn't work and what you did.

> *"Early in my Conversational AI Platform, the Thompson sampling router was selecting arms suboptimally — too exploratory. I spent a day debugging, assumed it was a bug in my implementation. Turned out the bug was in the reward signal: I was rewarding responses users continued talking to, but users sometimes continue talking because the AI frustrated them. I redesigned the reward as a composite — continuation plus sentiment plus response latency. Lesson: the reward signal is as important as the algorithm. That lesson is why I³'s router uses a composite reward rather than a single engagement flag. Every reward-shaping problem since, I've started by assuming my reward signal is the bug."*

### Q38: Describe a time you had to push back on a technical approach from a colleague.

> *"In MSc group coursework — building a transformer from scratch on tabular credit-card default data — a group member wanted to include 20-plus engineered features that were derivatives of each other. I argued that feature-space redundancy would confuse the attention mechanism and bloat the model unnecessarily. I backed it up: ran a correlation analysis showing most features were 0.9-plus correlated with one of three primitives. We reduced to 8 features. The model's attention maps became interpretable, and validation F1 actually went up. The lesson: pushback should come with data, not opinion. If I can't show it, I don't say it."*

### Q39: Tell me about a time you had to learn something quickly to deliver.

> *"Building the Crypto Statistical Arbitrage System, I needed Hidden Markov Models for regime detection. I'd studied HMMs theoretically but never implemented one. I didn't want to just import hmmlearn — I wanted to understand why they work. Three days with Rabiner's 1989 tutorial, implementing forward-backward from scratch, then Viterbi, then Baum-Welch, then testing on toy sequences where I knew the true states. By end of week, I had HMMs I fully understood, integrated into the trading pipeline. That pattern — learn from primary sources, implement from first principles, validate on toy cases — is how I approach anything new. It's why I built the TCN and SLM from scratch rather than pulling a library."*

### Q40: How do you prioritise when you have competing deadlines?

> *"For this specifically: MSc coursework due April 20 alongside April 28 slides. I mapped both deadlines to a single schedule, identified overlapping skills (the transformer coursework directly reuses for the SLM — I explicitly architected it that way), and scheduled non-overlapping work on alternating days. The key insight was not treating them as separate projects — they share architectural work. Generally, I prioritise by asking: what's the irreversible deadline, what's the highest-effort bottleneck, what can be done in parallel. The interview is irreversible; the slides are irreversible; coursework has extensions; demo reliability doesn't."*

### Q41: Give me an example of feedback that changed how you work.

> *"My dissertation supervisor on an early CV version: 'You list what you did, not what you learned. A CV of actions is a CV of a student. A CV of insights is a CV of a researcher.' I rewrote every bullet to lead with the insight or the decision, not the task. That principle changed how I present work broadly — including this project. The slides don't say 'I trained a TCN'; they say 'I hypothesised that dynamics carry more signal than content, and built an encoder to prove it.' Frame the decision, not the activity."*

### Q42: What kind of environment makes you do your best work?

> *"Small teams with direct communication, no intermediary layers. Access to people who know more than me about adjacent things — a designer to challenge my UX, a researcher to challenge my literature, an engineer to challenge my code. Freedom to explore the problem before committing, hard deadlines once committed. And honestly: a team that wants to build something that hasn't existed, not just ship. The HMI Lab description reads like exactly that, which is why I'm here."*

### Q43: Why Huawei specifically? You could interview anywhere.

> *"Three reasons. First, the technical thesis — on-device, experience-first, three-tier architecture — is the right thesis for where the field is heading. Other companies are still building toward cloud-centric everything. Huawei's bet on local AI aligns with my own view of the field's trajectory. Second, the HMI Lab is small and concept-driven — high learning velocity and real ownership, versus a large team optimising a known product. Third, the London–Edinburgh–Shenzhen pipeline — research insight flowing from universities into prototypes into products — is a specific organisational pattern that exists in very few places. I want to be somewhere research has a production path."*

### Q44: Describe a failure that shaped how you approach your work now.

> *"Early version of the MatchOracle football prediction system. I built an ensemble with Dixon-Coles plus XGBoost plus a neural net, obsessed over cross-validated Brier score, got excellent numbers. Then I paper-traded the predictions against bookmaker odds and lost money. The problem: I'd optimised for calibration against a test set that looked nothing like live betting conditions — team lineups, injuries, weather were absent features, and my model was implicitly exploiting leakage in historical odds. Lesson: a beautiful metric on a contrived dataset is worthless. Always ask what the real evaluation loop looks like. For I³, the 'real evaluation loop' is whether users feel understood — which is why I'm careful about claiming offline metrics prove anything."*

### Q45: How do you handle disagreement with senior colleagues?

> *"Start by assuming I'm missing information — seniors usually have context I don't. Ask for that context explicitly. If after the context I still disagree, make the case with data or structured reasoning, not intuition. If I'm overruled, commit and execute — half-hearted execution of a disagreed-with decision is worse than full execution. If I'm proven right later, I don't make a point of it; if proven wrong, I say so openly. The goal is the decision quality of the team, not the win rate of me."*

## 15.7 Depth-Probing Technical Questions (Low Probability, Very High Signal)

These are the questions a senior technical interviewer might ask if they want to see how deep you really go. If asked, take the pause, think properly, and answer with structure. If you genuinely don't know, say so — bluffing here is fatal.

### Q46: Walk me through the mathematics of Thompson sampling for contextual bandits with a Bayesian logistic regression.

> *"Setup: each arm a has a weight vector w_a in R^d; expected reward given context x is sigmoid(w_a^T x). Gaussian posterior on w_a via Laplace approximation — mean mu_a from maximum-a-posteriori gradient steps, covariance Sigma_a from the inverse Hessian of the log-posterior at mu_a. The Hessian for logistic regression with Gaussian prior N(0, lambda I) is H = lambda I + sum_t sigmoid(mu_a^T x_t) (1 - sigmoid(mu_a^T x_t)) x_t x_t^T. At decision time, sample w'_a ~ N(mu_a, Sigma_a) independently per arm; pick argmax_a sigmoid(w'_a^T x). This naturally explores arms with diffuse posteriors and exploits sharp ones. Online update: after seeing (x_t, a_t, r_t), refit mu_a via one or two Newton steps, update Sigma_a via Sherman-Morrison on the rank-1 Hessian update. Complexity per update: O(d^2)."*

### Q47: Derive NT-Xent loss and explain why it works.

> *"NT-Xent — Normalised Temperature-scaled Cross Entropy — is a contrastive loss. Given a batch of 2N examples forming N positive pairs through augmentation, for each pair (i, j) the loss is L_{i,j} = -log [ exp(sim(z_i, z_j) / tau) / sum_{k in 2N, k != i} exp(sim(z_i, z_k) / tau) ]. Similarity is cosine of L2-normalised embeddings; tau is temperature. Total loss averages L_{i,j} over all positive pairs in the batch. Why it works: it maximises mutual information between augmented views of the same example — the numerator — while pushing apart unrelated examples — the denominator. The temperature tau controls the softmax sharpness; low tau focuses gradient on hardest negatives, high tau treats all negatives equally. Empirically tau around 0.1 works well. For the TCN, positive pairs are two augmentations of the same synthetic archetype sequence; negatives are all other sequences in the batch."*

### Q48: What's the computational cost of cross-attention at inference vs self-attention only?

> *"Self-attention: O(N^2 d) for sequence length N and dimension d, dominated by the softmax attention matrix. Cross-attention between a query sequence of length N and a key/value sequence of length M: O(N M d) for the attention computation. In I³, the conditioning sequence is four projected tokens, so M=4. That makes cross-attention O(4Nd) per layer — essentially linear in N, negligible compared to self-attention's quadratic cost. Total overhead of adding cross-attention conditioning per layer is under 5% of total per-layer compute. Alternative — concatenating conditioning into the self-attention sequence — would raise N by 4 and balloon self-attention cost quadratically, which is why cross-attention is strictly preferable at this regime."*

### Q49: How would you prevent the TCN from memorising individual users' quirks as noise?

> *"Two mechanisms. First, the contrastive training objective pushes encoding toward archetype-level features rather than individual-level — positive pairs are same-archetype, different-user samples, so the network is rewarded for ignoring user-specific noise to put same-archetype sequences close. Second, feature normalisation is applied per-user at inference — keystroke speed is z-scored against the user's rolling baseline before encoding, so absolute scale effects are removed. The encoder sees relative dynamics, not absolute. In production, further: weight decay and dropout during training, periodic retraining with aggregated federated signals, and explicit evaluation of per-user variance in embedding space to detect overfitting. If the embedding for one user varies wildly across sessions, it's encoding noise; if it tracks their state smoothly, it's encoding signal."*

### Q50: How would you formally verify the sensitive-topic classifier doesn't leak queries to the cloud?

> *"This is an information-flow problem — you want to prove that for any input classified as sensitive, no derived signal reaches the cloud call. Three approaches, increasing rigour. First, structural: the classifier runs before the router; if the classifier labels sensitive, the router's 'cloud' arm is masked to zero probability. This is enforced at the code level via an explicit short-circuit. Second, testing: fuzzing the classifier with adversarial inputs — red-teamed prompts designed to evade the classifier — and confirming no cloud calls trigger. Third, static information-flow analysis: labelled types that track sensitivity through the codebase, compiler-checked to prevent sensitive types flowing into cloud-call sinks. Approach one is in the prototype; two and three are production. For a formal guarantee, something like the IFC calculus in F* or similar would be the right tool, but that's a significant investment."*

### Q51: Sketch how you'd port the SLM to MindSpore Lite.

> *"PyTorch-to-MindSpore conversion via ONNX intermediate. Export the SLM to ONNX — the Pre-LN transformer, cross-attention, and causal attention mask are all standard ONNX ops. Then convert ONNX to MindSpore Lite using the official converter. Post-conversion, re-run the INT8 quantisation calibration inside MindSpore Lite using its own calibration toolchain, because MindSpore's quantisation ops aren't always bit-identical to PyTorch's. Validate output equivalence by comparing generated tokens on a held-out set — expect under 1% divergence in generation due to quantisation differences; larger divergence indicates a conversion bug. Deploy to Kirin NPU via the MindSpore Lite runtime. Honest caveat: I haven't done this port — it's what I'd do based on reading the docs."*

### Q52: What's the hardest part of this project technically, and did you solve it or work around it?

> *"Hardest part: making the cross-attention conditioning actually change generation in a way measurable beyond chance. An SLM this small has limited representational capacity, and four conditioning tokens can get lost in self-attention noise. What worked: (1) conditioning injection at every transformer layer, not just the first; (2) initialising cross-attention weights such that early training the conditioning has meaningful gradient signal; (3) an auxiliary loss that penalises conditioning-agnostic outputs — effectively forcing the model to use the conditioning. This is a partial solution — the conditioning effect is measurable but not as strong as I'd want. A larger model or distillation from a conditioned teacher would likely improve it substantially."*

---

# PART 16: INTERVIEW LOGISTICS

## 16.1 Location

**Address:** Gridiron Building, 1 Pancras Square, King's Cross, London N1C 4AG.

**Entry route:**
1. Ground floor reception of the Gridiron Building — not Huawei reception directly.
2. Sign in with building reception, show photo ID, receive visitor badge.
3. Lift to the 5th floor.
4. Exit lift — Huawei reception is immediately visible.
5. Sign in at Huawei reception.
6. Wait — someone will collect you and escort you to Meeting Room MR1.

**Landmarks:** King's Cross / St Pancras station is 2–3 minutes walk. Pancras Square is the pedestrianised area north of the stations. The Gridiron Building is recognisable by its cross-braced facade. Google Maps resolves "1 Pancras Square" correctly.

## 16.2 Timing Plan

| Time | Action |
|------|--------|
| 07:00 | Wake. Do not oversleep — fog is worse than mild tiredness. |
| 07:00–08:00 | Breakfast. Substantial but not heavy — eggs and toast, oats and fruit. |
| 08:00–09:30 | **Final single rehearsal.** One full pass, timed. Do not rehearse again after. |
| 09:30–10:00 | Shower, dress, pack. Final bag check against 16.3. |
| 10:00–10:15 | Review PART 17 fallback scenarios. Mentally drill each. |
| 10:15 | Leave for station. |
| 10:15–11:30 | Travel. Central Line Newbury Park → Holborn → Piccadilly Line King's Cross. |
| 11:30 | Arrive at King's Cross / St Pancras. |
| 11:30–11:45 | Walk to Pancras Square. Sit on a bench. Breathing. No phone, no slides. |
| 11:45 | Enter Gridiron Building. Check in. Go up. |
| 11:55 | Arrive at MR1 via Huawei reception escort. |
| 12:00 | **Interview starts.** |
| 13:00 | Interview ends. |
| 13:00–13:10 | Exit building calmly. Walk 2–3 minutes away before any phone use. |

**Buffer logic:** 15 minutes arrival buffer at the building, 30 minutes travel buffer vs. minimum-time-journey. If Central Line fails, you have time to reroute. See 16.7.

## 16.3 What to Bring (Checklist)

**Essential (verified the night before):**
- [ ] Photo ID — passport or driving licence (building entry mandatory)
- [ ] Laptop — fully charged, power adapter in bag
- [ ] Laptop power adapter
- [ ] HDMI adapter + USB-C adapter — MR1 presentation port is unknown; bring both
- [ ] USB drive with slides PDF — backup in case laptop projection fails
- [ ] USB drive with backup demo video (5-minute walkthrough) — in case live demo fails
- [ ] Printed slides (6-per-page, landscape) — for physical backup and note-making
- [ ] Notebook + pen — for candidate-Q&A notes
- [ ] Phone — charged, silenced
- [ ] Water bottle — 30 minutes of talking dries the mouth

**Recommended:**
- [ ] Copy of CV (2 copies) — if referenced
- [ ] Light jacket or umbrella — check weather the night before
- [ ] Snack bar — for after the interview

**Avoid:**
- Large backpacks suggesting travel
- Strong cologne or perfume (closed room)
- Gum, mints with heavy odour
- Heavy meal immediately pre-interview

## 16.4 What to Wear

**Target register:** Smart business casual. HMI Lab is design-inflected research — not investment banking, not startup. A suit is overdressed; jeans are underdressed.

**Recommended:**
- Button-up shirt — white, light blue, or subtle micro-pattern. No tie needed.
- Chinos or smart trousers — navy, charcoal, or dark grey.
- Dark leather belt matching shoes.
- Clean leather shoes or smart derbies. No trainers. No scuffed soles.
- Light jacket or unstructured blazer if cold (optional).

**Avoid:**
- Full suit (too formal)
- T-shirt (too casual)
- Bright or distracting patterns
- Branded or logo-heavy clothing
- Hoodies

**Grooming:** Clean, neat, hair settled, beard trimmed if applicable.

**Comfort test:** Wear the outfit for an hour the day before, rehearsing the presentation in it. Standing for 30 minutes in uncomfortable shoes will show.

## 16.5 During the Interview — Behaviour Protocol

**First 30 seconds:**
Make eye contact, smile, introduce yourself briefly. Suggested opener:

> *"I'm Tamer. Thanks for having me. I've got a 30-minute technical presentation on a system I've built — I'll happily take questions throughout if anything's unclear, but if it's OK with you, I'd suggest saving most questions for the end so I can show you the full arc."*

This sets expectations, signals confidence, and shows you're in control of the format.

**Presentation (minutes 1–30):**
- Follow the slide-by-slide structure in PART 14.1.
- Pace. Don't race through slide 1 and linger on slide 8. Timed rehearsals matter.
- Minute 12 is the live demo. This is the peak. If it glitches, handle calmly — see PART 17.
- Minute 28 is the close. Rehearsed verbatim: *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*

**Technical Q&A (10 minutes):**
- Listen fully before answering.
- Reframe when helpful: *"So you're asking about X — here's how I think about that..."*
- Admit ignorance when real: *"I haven't thought about that case specifically. My instinct is X, but I'd want to actually work through it before being confident."*
- Follow up once: *"Does that address what you were asking, or should I go deeper on any part?"*

**Behavioural (10 minutes):**
- Use STAR implicitly but never announce it.
- 60–90 seconds per story. Matthew doesn't need the saga.
- Rehearse 6–8 stories covering the PART 15.6 questions.

**Lab overview (5 minutes):**
- Listen. One clarifying question if something is genuinely unclear. Do not re-pitch yourself here.

**Candidate Q&A (5 minutes):**
- 3 questions from PART 6.8. No more.
- Take notes on their answers — it reads as engagement.
- Close with sincerity: *"Thank you — this was really helpful. I'd be excited to work on problems like these."*

## 16.6 Closing the Interview

1. Stand up. Shake hands — firm, brief.
2. Thank them by name: *"Thank you, Matthew — I really appreciated the conversation."*
3. **Do not** ask about next steps in the room. That goes through Vicky by email later.
4. Walk out calmly. No phone in the building.
5. Exit. Walk 2–3 minutes away before any reaction.

## 16.7 Travel Contingency Plans

**Primary route:** Central Line Newbury Park → Holborn → Piccadilly Line → King's Cross. 55–60 minutes.

**Contingency A (Central Line suspended):** Newbury Park → bus 25 to Stratford → Jubilee Line → King's Cross / St Pancras. 75–85 minutes.

**Contingency B (Jubilee also down):** Uber from Newbury Park. Budget £60. Allow 60–75 minutes.

**Contingency C (multiple disruptions morning of):** Email Matthew AND Vicky at 10:45 with:

> *"Unexpected tube disruption on my route — rerouting now. Currently estimate arrival 12:10–12:15. Sincere apologies for any inconvenience."*

**Check before leaving:** TfL status at 08:00, Citymapper live, set Google Maps alternative route.

## 16.8 Post-Interview Follow-Up

**Within 24 hours**, send a short thank-you email to matthew.riches@huawei.com:

> **Subject:** Thank you — great to meet you today
>
> Hi Matthew,
>
> Thank you for the conversation today. I enjoyed walking you through I³ and hearing about the HMI Lab's direction. Your point about [specific thing he said] in particular stayed with me — [one sentence response to it].
>
> If there's any follow-up information that would be useful for the team's deliberations, please let me know.
>
> Best,
> Tamer

**Do not:**
- Ask when you will hear back
- Attach additional materials
- Write more than three short paragraphs
- Flatter excessively

## 16.9 Salary Expectations (If Asked)

**Do not volunteer a number.** If asked, deflect and reverse:

> *"I'd want to understand the role and team a bit better before talking numbers. Can you share the range for the position?"*

If pressed:

> *"Market rate for ML specialists with my profile in London ranges roughly £35K to £55K for a 2-year research-track internship at a major tech company. I'd want to be competitive with that, and I'm flexible based on the specifics of the role and benefits package."*

**Do not:**
- State a specific number first
- Focus on salary over role substance
- Undersell yourself

---

# PART 17: FALLBACK PLANS (WHEN THINGS BREAK)

Every rehearsed response below exists because the default response — panic, or losing composure — costs the interview. Staying composed in a failure mode is itself a signal Matthew will notice and weigh positively.

## 17.1 Demo Fails to Start

**Scenario:** Open browser at `localhost:8000`, nothing loads.

**Response:**
1. *"One moment — the server's being slow to start, let me recycle it."*
2. Open terminal, kill and restart: `pkill -f uvicorn && python -m backend.app`.
3. Wait 5 seconds. Refresh browser.
4. If still no load after 20 seconds: switch to backup video. *"I've recorded a walkthrough — let me show you that, and we can dig into specifics afterwards."*

**Prevention:**
- Launch the backend and verify the demo is running **before entering the building**.
- Test on battery only — MR1 outlet may be awkward.
- Phone tether for the Anthropic API route in case venue Wi-Fi fails.

## 17.2 WebSocket Disconnects Mid-Demo

**Scenario:** Typing into chat, response doesn't return.

**Response:**
1. *"WebSocket's hiccuped — let me reconnect."* Click the reconnect button (build one).
2. If reconnect fails in 10 seconds: *"Let me just restart the backend — takes 5 seconds."* Kill and restart.
3. If that fails: *"I'll switch to the backup video for the rest of the demo — let me show you the key moments."*

**Prevention:**
- Frontend auto-reconnect with exponential backoff.
- Visible "reset session" button.
- 5-minute backup demo video on desktop, pre-tested.

## 17.3 Anthropic API Fails

**Scenario:** Router selects cloud arm, Claude API errors.

**Response:** Reframe as a feature. *"The router just detected a cloud-arm failure and fell back to the local SLM — that fallback is designed in, no single point of failure."*

**Prevention:**
- Implement the fallback in router code with proper exception handling.
- Test fallback explicitly before the interview.
- Confirm API key works the morning of.

## 17.4 SLM Produces Incoherent Output

**Scenario:** SLM generations look bad enough to hurt the demo.

**Response options:**
- **Option A (recommended if partial coherence):** bias the bandit temporarily toward cloud, say *"For the demo I'll show cloud-routed responses to keep the examples crisp — the SLM is present, and you can see its outputs in the routing dashboard when the router selects it."*
- **Option B (if very bad):** switch to backup video for the conversation flow; keep the SLM architecture slides.

**Prevention:**
- SLM quality check by end of Day 14 (April 24). If genuinely bad, execute **Fallback C: cloud-only with dynamic system prompt adaptation**. The SLM architecture is still demonstrated in slides; it is simply not called in the live demo. Acceptable — see PART 12.4.

## 17.5 You Blank on a Question

**Scenario:** Question asked, mind goes empty.

**Response:**
- *"Give me a second to think about that properly."* — pause 5–10 seconds. Silence is fine.
- If still no answer: *"Honestly, I haven't thought about that specific angle. My first instinct would be X, but I'd want to actually work through it rather than hand-wave. Can I come back to it at the end?"*
- If they return to it: either a thought-through answer, or *"I've been thinking about it — here's where I'd start, but I'd need more than a minute to give you a confident answer."*

**This is the single most important response pattern.** Bluffing a technical answer is fatal. Structured acknowledgement of uncertainty is respected.

## 17.6 Matthew Pushes Back Hard on a Design Choice

**Scenario:** *"I don't buy the TCN choice — why not X?"*

**Response:**
- **Do not defend reflexively.** *"Interesting — tell me more about what you'd prefer?"*
- If his alternative has merit you hadn't considered: *"That's fair — I think X would work too, and actually for [specific case] it'd probably be better. I chose TCN because [specific reason]. If I were extending the prototype I'd run an ablation."*
- If his alternative has flaws: *"I actually considered X early on. The issue I ran into was [specific concern]. Does that match your intuition or do you see it differently?"*

**Rule:** Intellectual humility is charisma. Defensive posture is the opposite.

## 17.7 Tube Signal Failure on Interview Morning

**Scenario:** Central Line suspended, April 29, 09:00.

**Response:**
1. Check TfL at 08:00 the morning of.
2. Contingency B route: Newbury Park → bus 25 → Stratford → Jubilee → King's Cross.
3. If that fails too: Uber, £60 budget.
4. Email Matthew + Vicky at 10:45 if slipping: *"Signal failure on Central Line, rerouting, expect arrival 12:10–12:15, sincere apologies."*

## 17.8 Interviewer Arrives Late

**Scenario:** 12:05 and nobody has collected you from Huawei reception.

**Response:** Stay seated. Do not ask reception to chase. Wait until 12:10. At 12:10, politely ask reception to confirm with Matthew's calendar. Do not appear impatient — something has come up for them, and your reaction is data.

## 17.9 You Get Asked Something You've Already Answered

**Scenario:** A question overlaps with something you covered in the presentation.

**Response:** Do not condescend. *"Yeah, I touched on that in slide X — let me expand on the part you're asking about specifically."* Then answer the new angle.

## 17.10 Interview Is Going Badly (Mid-Interview Recovery)

**Scenario:** You sense the room is cold. Answers aren't landing.

**Response:**
- Do not over-correct. Don't start over-explaining or inflate energy artificially.
- Find a moment to ask: *"Is there a specific area you'd like me to go deeper on, or something I haven't covered that would be useful?"*
- Their answer redirects you to what they actually want. Follow it.
- If the signal remains cold: finish with composure. A graceful end to a mediocre interview is better than a panicked end.

**Remember:** You don't actually know how it's going. Some interviewers have a cold default that is not a signal of rejection. Your internal model of "it's going badly" is unreliable in the room.

## 17.11 You Realise Mid-Demo You Have a Bug

**Scenario:** You notice the embedding visualization is frozen, or a number is obviously wrong.

**Response:**
- *"Sharp-eyed — that's a frontend bug I noticed last night and didn't get to fix. The underlying engine is working correctly; it's the render layer. Let me show you via the logs instead."* Open terminal, show raw logs or API responses.
- Never hide it. Acknowledging the bug and explaining the gap between "surface" and "underlying correctness" is a good signal.

## 17.12 You Realise You Misspoke Earlier

**Scenario:** You said something in the presentation that, on reflection, is wrong.

**Response:** At the next natural pause: *"One correction — I said X earlier, that's not quite right. The actual answer is Y."* Then continue. Self-correcting unprompted is another trust signal.

## 17.13 General Principles for Recovery

- **Stay in role.** You are a competent engineer dealing with the normal friction of demos. You are not a panicked candidate.
- **Name the problem out loud.** Silent struggle reads worse than announced struggle.
- **Have a pre-planned response.** Never improvise the recovery — rehearse each scenario above.
- **Time-box recovery.** If something takes more than 30 seconds to fix, switch to backup. Don't burn 3 minutes of presentation time debugging.
- **The backup video is your safety net.** Rehearse using it, so switching doesn't feel like defeat.

---

# PART 18: SUGGESTED BUILD CADENCE

> **Claude Code:** This is a **suggested cadence**, not a rigid schedule. It captures a reasonable sequencing of work given the dependency structure between components (features before encoder, encoder before user model, SLM architecture before training, integration before polish). Within that structure, you have full discretion: parallelise what can be parallelised, front-load hard problems you want to de-risk early, skip ahead if a task is faster than estimated, backtrack and refactor if an early choice proves wrong. The dates below are reference points — what matters is that the build is interview-ready by April 28 and demo-stable by April 29.
>
> The "definitions of done" per day are useful targets for self-checking progress. They are not contractual. If you achieve the same goal via a different set of artefacts, that is fine — document it and move on.

## 18.1 Structural Principles

The 17-day build window benefits from three loose phases with checkpoints between them. A checkpoint is a convenient moment to assess whether the foundation supports the next phase — not a gate that blocks progress. If a checkpoint reveals a gap, you choose whether to remediate immediately, defer, or proceed and revisit.

| Phase | Days | Focus | Checkpoint |
|-------|------|-------|------|
| Phase 1: Core ML | 1–7 | Feature engineering, TCN, SLM architecture, training | SLM + TCN trained and validated |
| Phase 2: Integration | 8–14 | Router, cloud, privacy, pipeline, web interface | End-to-end demo works on laptop |
| Phase 3: Polish | 15–17 | Edge profiling, slides, rehearsal | Demo + deck interview-ready |

**Time-use principle:** if a day's work finishes early, good options include (a) writing additional tests for that day's code, (b) reading adjacent research, (c) refactoring for clarity, (d) prototyping an alternative to a decision you're uncertain about, (e) moving ahead into the next day's work if you're confident the dependencies are solid. Nothing prohibited — just avoid forward-racing on brittle foundations.

## 18.2 Phase 1: Core ML (Days 1–7)

### Day 1 (April 11) — Foundation

**Goal:** Project structure exists, feature extraction runs on synthetic input end-to-end.

**Tasks:**
- Repository initialised with `pyproject.toml`, directory structure per PART 10.2
- Configuration system (`config/default.yaml` loaded via pydantic models)
- `InteractionFeatureVector` dataclass with full 32-feature schema
- Keystroke feature extractors (8 features): inter-key intervals, burst detection, correction rate, etc.
- Linguistic feature extractors (8 features): type-token ratio, Flesch-Kincaid, formality score, sentiment lexicon lookup
- Session feature extractors (8 features): message rate, length trends, topic continuity indicator
- Deviation feature extractors (8 features): z-scored against running baseline
- Unit tests for every feature extractor
- Sentiment lexicon built (500 positive, 500 negative words from NRC Emotion Lexicon or similar)

**Definition of done:**
- `pytest tests/features/` passes
- Given a toy keystroke stream, a complete 32-dim vector is produced in under 50 ms
- Vector components are in sensible ranges (documented expected ranges in code)

**Output artefacts:** `backend/features/`, `tests/features/`, `docs/feature_schema.md`

### Day 2 (April 12) — Synthetic Data and TCN Architecture

**Goal:** 10,000-session synthetic dataset exists; TCN forward pass works.

**Tasks:**
- Synthetic data generator: 8 user-state archetypes with parameter distributions per PART 12.1
- Markov transition matrix between archetypes (calibrated for realistic session dynamics)
- Generator produces labeled sessions of 15–30 messages each with InteractionFeatureVector sequences
- Generate and persist 10,000 sessions to disk as HDF5 or similar efficient format
- `CausalConv1d` implementation from scratch (padding trick for strict causality)
- `CausalConvBlock` — causal conv + layer norm + GELU + residual
- `TemporalConvNet` — 4 blocks with dilations [1, 2, 4, 8]
- Global average pooling head → 64-dim embedding
- Forward-pass unit tests including causality verification (output at timestep t must not depend on input at timestep t+1)

**Definition of done:**
- Synthetic dataset exists at `data/synthetic/sessions_10k.h5`
- TCN forward pass on a (batch=8, seq_len=10, features=32) tensor returns (batch=8, embed=64)
- Causality unit test passes
- Parameter count logged — expected ~300–500K

### Day 3 (April 13) — TCN Training

**Goal:** TCN trained on synthetic data with validated embedding quality.

**Tasks:**
- NT-Xent contrastive loss from scratch (PART 15.7 Q47 formula)
- Augmentation strategy for positive pairs: random temporal crops, feature-dropout noise, time-warping within bounded range
- AdamW optimiser, cosine LR schedule with warmup
- Training loop with per-epoch validation
- Train for 30–50 epochs on 10K synthetic sessions (should take 1–2 hours on laptop GPU or 3–4 hours CPU)
- Validation metrics: silhouette score on archetype clusters, KNN top-1 accuracy on archetype labels
- 2D PCA projection of validation embeddings saved as PNG
- Checkpoint saved with full metadata (architecture hash, training config, validation metrics, seed)

**Definition of done:**
- Silhouette score on validation ≥ 0.5
- KNN top-1 accuracy ≥ 0.80
- PCA projection shows visible archetype clusters
- Checkpoint file loads cleanly and reproduces validation metrics

**Risk:** if silhouette < 0.4, the encoder is not learning archetype structure. Remediate by (a) inspecting augmentation strength (too strong collapses signal), (b) checking temperature tau (too high flattens, too low over-sharpens), or (c) increasing training data volume.

### Day 4 (April 14) — User Model + Adaptation Controller

**Goal:** User model persists to SQLite; adaptation controller maps state to AdaptationVector.

**Tasks:**
- Three-timescale user model: `InstantState`, `SessionEMA`, `LongTermBaseline`
- Deviation metrics: current vs baseline, current vs session
- SQLite persistence: `user_profiles` table with encrypted 64-dim embedding + scalar metrics
- Baseline computation: first 5 messages establish baseline, then EMA updates
- `AdaptationController`: user state → 4-dim AdaptationVector (cognitive load, communication style, emotional tone, accessibility)
- Mapping logic from state features to adaptation — documented rules, not learned, for prototype clarity
- Unit tests for each mapping rule

**Definition of done:**
- User model round-trips through SQLite cleanly
- Given a sequence of user states, the adaptation vector evolves sensibly (documented expected trajectories for each of the 8 archetypes)
- Deviation metrics are computed and stored

### Day 5 (April 15) — SLM Tokeniser + Dialogue Data Prep

**Goal:** Dialogue corpus prepared with adaptation labels; tokeniser trained.

**Tasks:**
- Download DailyDialog and EmpatheticDialogues
- Data cleaning: filter non-English, deduplicate, length-filter
- `SimpleTokenizer` with BPE fallback — vocabulary size ~8K tokens
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<cls>`
- Token embeddings with dimension matching `d_model` (256)
- Sinusoidal positional encoding from scratch
- Derive adaptation labels for dialogue examples — formality via heuristics, length buckets, sentiment via lexicon
- Accessibility augmentation: create paired (normal, simplified) versions for a subset of examples
- Train/val/test split (80/10/10)

**Definition of done:**
- Tokenizer round-trips text cleanly (encode → decode = original with acceptable loss)
- Adaptation labels cover full range of the 4-dim vector space
- Data statistics documented: corpus size, vocabulary coverage, label distribution

### Day 6 (April 16) — SLM Attention + Transformer Blocks

**Goal:** Full transformer block from scratch with cross-attention conditioning.

**Tasks:**
- `MultiHeadSelfAttention` from scratch — Q/K/V projections, scaled dot-product, causal mask
- `MultiHeadCrossAttention` from scratch — Q from decoder, K/V from conditioning source
- `ConditioningProjector`: AdaptationVector + UserStateEmbedding → 4 conditioning tokens at `d_model`
- `AdaptiveTransformerBlock`: Pre-LN self-attention → Pre-LN cross-attention → Pre-LN FFN
- Numerical stability: attention masking with large negative values, not -inf (for fp16 compatibility later)
- Shape tests, gradient flow tests, causal mask correctness tests

**Definition of done:**
- Full transformer block forward pass on a batch returns correct shapes
- Backward pass produces non-NaN gradients
- Causal mask verified: future tokens cannot influence past positions
- Cross-attention conditioning verified: different conditioning inputs produce different outputs

### Day 7 (April 17) — Full SLM + Training Setup

**Goal:** SLM assembled; training running.

**Tasks:**
- Full `AdaptiveSLM` with 4 transformer blocks, d_model=256, 4 heads
- Parameter count target: 8–15M
- Autoregressive generation with temperature sampling, top-k, nucleus (top-p)
- Training loop: cross-entropy loss with teacher forcing, AdamW, cosine LR with warmup
- Gradient accumulation for larger effective batch size
- Mixed precision training if GPU supports it
- Auxiliary loss (optional but recommended): conditioning-consistency loss penalising conditioning-agnostic outputs
- Start training — expect 40–80 epochs, ~8–16 hours on laptop GPU

**Definition of done:**
- Forward + backward + optimiser step cycle runs without errors
- Training loss decreasing over first 2 epochs
- Generation from untrained model produces token sequences (noise, but structurally valid)

**Checkpoint 1: End of Phase 1 (Evening of Day 7)**

Before moving to Phase 2, a good moment to self-check:

- [ ] TCN trained, checkpoint saved, validation metrics ≥ targets
- [ ] User model + adaptation controller working end-to-end on synthetic states
- [ ] SLM training launched; first 2 epochs show loss decreasing
- [ ] All Phase 1 unit tests passing
- [ ] No open bugs in Phase 1 code

If any check surfaces a real problem, consider remediating before proceeding — compounding issues downstream is usually more expensive than pausing to fix. Your call.

## 18.3 Phase 2: Integration (Days 8–14)

### Day 8 (April 18) — SLM Monitoring + Thompson Sampling Router

**Goal:** SLM training continues; router implemented and tested offline.

**Tasks:**
- Monitor SLM training; intervene if loss curves pathological
- `ContextualThompsonBandit` from scratch — Bayesian logistic regression per arm, Laplace approximation, online update via Sherman-Morrison
- Arms: `local_slm`, `cloud_llm`
- Context features (12-dim): user state summary (4), query complexity (3), topic sensitivity (2), confidence signals (3)
- Offline evaluation harness: replay synthetic rollouts, compare policy to oracle best routing
- Query complexity estimator (length, question-type indicators, entity density, multi-intent detection)
- Topic sensitivity classifier (keyword-based for prototype: health, financial, personal, legal)

**Definition of done:**
- Bandit runs end-to-end: context in → arm selected → reward observed → posterior updated
- Offline regret over replay dataset is sublinear (validated against oracle)
- Query complexity + topic sensitivity produce sensible outputs on hand-crafted test cases

### Day 9 (April 19) — Cloud Integration + Privacy Layer

**Goal:** Cloud arm functional; privacy layer enforced.

**Tasks:**
- Anthropic API client for Claude (handle rate limits, retries, timeouts cleanly)
- Dynamic system prompt construction from AdaptationVector — template library with slot-fills
- Response post-processing: enforce adaptation parameters on output (truncation if too long, tone-adjustment regex where possible)
- `PrivacySanitizer`: PII regex stripping (email, phone, names via simple heuristics, addresses)
- `EmbeddingEncryptor`: Fernet symmetric encryption with per-user key derived from a master secret
- Sensitive-topic short-circuit: if topic classifier labels sensitive, router's cloud arm probability is masked to zero
- Logging infrastructure: structured logs with `structlog`, no PII ever logged

**Definition of done:**
- Cloud arm returns responses for non-sensitive queries in under 2 seconds
- Sensitive queries never reach cloud (verified with test cases)
- Encrypted embeddings round-trip cleanly
- Audit log confirms no raw text persisted

### Day 10 (April 20) — Pipeline Orchestration

**Goal:** Full pipeline runs end-to-end in a single process.

**Tasks:**
- Main `Pipeline` class connecting all components
- Request flow: raw interaction → features → TCN → user model → adaptation → router → response → engagement signal → update
- Engagement signal computation from the user's next interaction
- Interaction diary logging (embedding + adaptation decision + route + engagement)
- Error handling: each component has a well-defined failure mode and fallback path
- Integration tests covering happy path and every documented failure mode

**Definition of done:**
- End-to-end test: 10-message synthetic session → final state of user model + diary is correct
- All component failure modes handled gracefully (no silent errors, no crashes)
- Pipeline latency on laptop: under 300 ms for local route, under 1500 ms for cloud route

### Day 11 (April 21) — FastAPI Backend

**Goal:** Backend server with WebSocket + REST endpoints.

**Tasks:**
- FastAPI application with WebSocket handler for real-time interaction
- WebSocket protocol per PART 13.3 (message schema, keystroke events, state broadcast)
- REST endpoints:
  - `GET /health` — readiness check
  - `GET /profile/{user_id}` — user profile summary
  - `GET /diary/{user_id}` — interaction diary
  - `GET /stats` — system stats
  - `POST /admin/reset` — reset session (for demo control)
  - `POST /admin/profiling` — trigger edge profiling run
- InteractionMonitor integration: keystroke events from frontend → features
- CORS configured for local frontend
- Simple test client to verify WebSocket flow

**Definition of done:**
- Backend starts cleanly via `python -m backend.app`
- Test client can connect, send messages, receive adapted responses
- All REST endpoints return valid responses

### Day 12 (April 22) — Frontend Core

**Goal:** Web interface functional with keystroke capture and chat flow.

**Tasks:**
- HTML/CSS layout per PART 13.1 — 4-panel design (chat, embedding viz, adaptation gauges, diary)
- WebSocket client with auto-reconnect and exponential backoff
- `KeystrokeMonitor` implementation: capture keydown/keyup events, inter-key intervals, corrections
- Message composition flow: typing events streamed → full message sent on Enter
- Basic CSS styling matching the PART 13.2 design system
- Session reset button (for demo control)

**Definition of done:**
- Frontend loads, connects to backend, exchanges messages end-to-end
- Keystroke events stream in real-time to backend
- Chat messages display correctly with adaptation metadata visible

### Day 13 (April 23) — Frontend Dashboards

**Goal:** Visualisations communicate the system's internal state in real-time.

**Tasks:**
- Canvas-based `EmbeddingViz`: 2D projection of user state embedding, updates per-message with smooth animation
- Adaptation gauge bars (animated): cognitive load, communication style formality, emotional tone, accessibility
- Routing confidence display: which arm selected, bandit's posterior for each arm
- Engagement score display: running engagement trend
- Diary panel: expandable list of past interactions (embeddings + adaptations, not text)
- Visual design passes the "designer test" — clean, legible, clearly indicating state changes

**Definition of done:**
- Every dashboard element updates in real-time as messages flow
- Demo scenarios (PART 13.6) produce visually clear state changes
- No dashboard bug that would distract during presentation

### Day 14 (April 24) — Integration Polish + Demo Recording

**Goal:** System stable end-to-end; backup demo video recorded.

**Tasks:**
- Full end-to-end testing — run every demo scenario start to finish
- Bug fixes on any issues found
- WebSocket reconnection stress test
- Demo utilities: seed data, session reset, scenario selector
- **Record backup demo video (5 minutes) covering all 4 demo phases**
- SLM quality assessment: if output quality is poor, execute Fallback C (cloud-only with adaptation) per PART 17.4

**Definition of done:**
- Every demo scenario runs cleanly start to finish with no manual intervention
- Backup video recorded, reviewed, stored on two USB drives
- Any remaining bugs documented and assigned to Day 15

**Checkpoint 2: End of Phase 2 (Evening of Day 14)**

Before moving to Phase 3, a useful self-check:

- [ ] Live demo runs all 4 scenarios cleanly
- [ ] SLM quality acceptable OR Fallback C in place
- [ ] Backup video recorded
- [ ] Frontend visualisations clear and stable
- [ ] No WebSocket flakiness observed in 30-minute continuous use
- [ ] All user-facing error paths handled gracefully

## 18.4 Phase 3: Polish (Days 15–17, plus buffer)

### Day 15 (April 25) — Edge Profiling + Quantisation

**Goal:** Profiling report proves edge feasibility.

**Tasks:**
- INT8 dynamic quantisation of TCN and SLM via PyTorch native tools
- Memory profiling: peak memory, parameter footprint, activation footprint
- Latency benchmarking: TCN forward, SLM generation per-token, full pipeline end-to-end
- Device-comparison extrapolation: Kirin 9000 (flagship phone), Kirin 820 (mid-range), Smart Hanhan class (constrained)
- Verify quantised output quality: compare generated text across a 100-sample test set, measure divergence
- Generate profiling report: Markdown table + supporting charts

**Definition of done:**
- Profiling report exists at `docs/edge_profiling_report.md`
- Quantised models saved separately from fp32 originals
- Quality divergence under 5% on test set
- Extrapolation methodology documented

### Day 16 (April 26) — Presentation Slides

**Goal:** 15-slide deck drafted and visually polished.

**Tasks:**
- Build slides per PART 14.1 structure
- Architecture diagrams finalised — choose a tool (Figma, draw.io, Excalidraw) and commit
- Code snippets for illustrative slides (short, readable, clearly annotated)
- Speaker notes for every slide — what to say, timing, transitions
- First timed rehearsal (read-through, no demo)
- Visual polish: consistent fonts, palette, spacing

**Definition of done:**
- All 15 slides exist with content and speaker notes
- Timed read-through under 20 minutes (leaves room for demo + Q&A)
- Slides exported as PDF and pptx

### Day 17 (April 27) — Full Rehearsal + Slides Submission

**Goal:** Presentation + demo rehearsed end-to-end; slides sent to Matthew.

**Tasks:**
- Full rehearsal including live demo, end-to-end, timed
- Second full rehearsal with fallback scenarios deliberately triggered
- Q&A drill: partner asks questions from PART 15, answer live
- Final bug fixes (cosmetic only — no architecture changes this late)
- **Email slides to matthew.riches@huawei.com by end of day**
- Email subject: *"Technical Presentation — Implicit Interaction Intelligence (I³)"*
- Email body: short, professional, acknowledges looking forward to the conversation

**Definition of done:**
- Both rehearsals run within time budget
- Fallback scenarios rehearsed at least once
- Slides sent to Matthew, delivery confirmed

### Day 18 (April 28) — Final Prep

**Goal:** Ready to walk into the room the next morning.

**Tasks:**
- One morning rehearsal (presentation + demo)
- Battery-only operation verified: laptop runs demo for 60+ minutes without power
- Pack bag per PART 16.3 checklist
- Pre-load demo: server running state, user profile seeded for fast start
- Confirm outfit (PART 16.4), prep mentally
- Early night. No further rehearsal after 21:00 — sleep matters more than one more run-through.

### Day 19 (April 29) — Interview Day

**Goal:** Execute.

See PART 16 for full logistics.

## 18.5 Contingency Buffer Logic

If a day slips:

- **Phase 1 slip (Days 1–7):** reduce synthetic dataset from 10K to 5K sessions. Reduce SLM training epochs from 80 to 40. Accept slightly weaker encoder/SLM quality in exchange for timeline recovery.
- **Phase 2 slip (Days 8–14):** simplify frontend dashboards (skip embedding viz if necessary; keep adaptation gauges and chat). Skip the diary panel if needed. Keep the core pipeline intact.
- **Phase 3 slip (Days 15–17):** reduce slides from 15 to 10. Skip live edge profiling; cite literature-based extrapolations with explicit caveat. **Never skip rehearsal.**

**Strong suggestion:** the checkpoint self-checks exist because a shaky foundation shipped to interview day tends to fail worse than a simpler foundation shipped well. Use your judgement — if you're confident about a piece despite a failing check, trust yourself. If you're uncertain, pause and remediate.

---

# PART 19: OPERATING PRINCIPLES FOR CLAUDE CODE

## 19.1 On Autonomy

Claude Code: this project is yours to build with full creative latitude. The specifications in Parts 10–14 describe a target architecture and a reasonable default implementation path. They are not a cage. If you see a better way, take it. If you see research you should consult, consult it. If you see a component that would benefit from a technique not mentioned here, use the technique.

**Specifically, Claude Code is expected and encouraged to:**

- **Search the web** for the latest papers, techniques, and best practices in any area of the build — temporal convolutional networks, contrastive learning, small language model training, cross-attention conditioning, contextual bandits, Bayesian logistic regression, INT8 quantisation, WebSocket architecture, privacy-preserving embeddings, federated learning, whatever is relevant. Do this proactively. The training-cutoff date on your world-knowledge is 2025 or earlier; active web search closes the gap.
- **Read the primary literature** when a technique is being implemented. NT-Xent comes from Chen et al. 2020 (SimCLR). Contextual Thompson sampling has foundational papers from Russo & Van Roy. Pre-LN architecture comes from Xiong et al. 2020. When implementing, read the source — it is always more correct than second-hand summaries.
- **Prototype alternatives** when a design decision has real trade-offs. If TCN vs Transformer for the encoder seems genuinely open, prototype both and benchmark. If cross-attention vs prepending seems genuinely open for conditioning, try both. Picking the wrong option early without evidence costs more than trying two options early.
- **Refactor architecture** if you see a cleaner structure. The directory layout in Part 10.2 is a default, not a mandate. If you see that a cleaner module boundary would pay off, take it.
- **Add features beyond the specification** if they improve the system and fit the 17-day budget. Candidate additions: attention visualisations of the cross-attention conditioning, an interpretability panel showing which features drove the current adaptation, an ablation mode in the UI that lets the demonstrator toggle the encoder on/off, a "what-if" mode that lets the user see alternative responses from different adaptations. Any of these would strengthen the demo.
- **Write more and better tests** than specified. Unit tests, integration tests, property-based tests for numerical components, snapshot tests for the frontend dashboards. Test infrastructure is never overkill in a 17-day project — it is what lets you refactor without fear.
- **Use every tool available to you** — bash for orchestration, file creation for artefacts, web search for research, image generation if useful for slide mock-ups, any library that helps. You are not restricted to a minimalist toolchain.
- **Propose improvements to the specification itself.** If something in Parts 10–14 is wrong, unclear, or suboptimal, say so in the commit message or a `NOTES.md` file and propose a replacement. Tamer reads every commit.

## 19.2 On Quality

The target is code that signals "senior-level ML engineer" to a hiring manager reviewing the repo. That signal tends to come from properties like these — use your judgement on how far to take each one given the time budget:

- **Type hints throughout**, using modern Python 3.10+ syntax (`X | Y` rather than `Union[X, Y]`, `list[int]` rather than `List[int]`).
- **Dataclasses or Pydantic models** for configuration and structured data, rather than raw dicts.
- **Specific exceptions** rather than bare `except:`. Propagate or translate meaningfully.
- **Structured logging** (`structlog` or `logging` with JSON formatting). Reserve `print()` for debug scripts.
- **Modular architecture** — each component testable independently. Pure functions where possible; side effects isolated.
- **Docstrings on public classes and functions**, Google or NumPy style. Include the "why," not just the "what."
- **Reproducible seeds** — any function with stochasticity takes a `seed` argument or uses a seeded RNG.
- **Model checkpoints with metadata** — architecture hash, training config, validation metrics, git commit SHA, training wall-clock, hardware profile.
- **A `README.md` a reviewer could follow** — clone, five numbered steps, demo running in the browser within 10 minutes.

These are defaults; you are free to use better patterns where you know them, or adapt these to fit the rest of the code's style.

## 19.3 On Scope and Cuts

Seventeen days is tight. If cuts become necessary, a useful priority order — lightest sacrifice first:

1. **Easiest to drop:** extensive comparative ablations. One good ablation in a slide reads as "I considered alternatives" without requiring the full matrix.
2. Breadth of synthetic archetype coverage. Six archetypes cover the demo cleanly; eight is nicer but not essential.
3. Frontend polish. A working dashboard that looks plain beats a glamorous dashboard with bugs.
4. Edge profiling breadth. One device-class extrapolation with clear methodology beats three shaky ones.
5. **Hardest to drop** (because they carry the signal to Matthew): end-to-end integration, backup video, rehearsal time, the honesty slide on limitations.

**The core deliverables that define "this project succeeded":**
- The TCN is trained and produces meaningful embeddings.
- The SLM architecture is built from scratch — the architecture itself is a deliverable even if the generation quality is limited.
- The router is a real contextual bandit, not a simple rule.
- The demo runs live on the laptop.
- The slides exist, are submitted on time, and are polished.

These are what prove "yes" to the four screening questions. Trade-offs against them should be deliberate, not accidental. But if you find a way to honour these goals via a different artefact set, that is fine — the goals matter more than the specific deliverables.

## 19.4 On Iteration Discipline

The suggested cadence (Part 18) captures dependency structure — features before encoder, encoder before user model, SLM architecture before training, integration before polish. Within that dependency structure, sequencing is yours. Parallelise what can be parallelised. Front-load hard problems to de-risk them. Backtrack and refactor when an earlier choice proves suboptimal.

**When a component isn't working:**

1. **Instrument first.** Add logging, print intermediate shapes, visualise activations. Understand before fixing.
2. **Reproduce minimally.** Isolate the failure to the smallest code path that reproduces it.
3. **Consult the literature.** Before hacking around a training instability, check whether it is a known phenomenon with a known fix. Web search aggressively.
4. **Time-box debugging.** If a specific bug has consumed hours with no progress, step away, do the next thing, return fresh.
5. **Flag blockers.** If you are genuinely blocked in a way that affects downstream work, note it in `NOTES.md`. Tamer checks it.

## 19.5 On Testing and Validation

The demo is a live system. Bugs in the demo cost the interview. Tests exist to make the demo robust, not to demonstrate rigour.

**Testing priorities, in order:**

1. **The four demo scenarios run cleanly end-to-end.** This is the single most important test. Write a script that runs each scenario, asserts the expected state at each step, and fails loudly on deviation.
2. **The WebSocket doesn't flake under 30 minutes of continuous use.** Run a continuous-use test overnight.
3. **Every fallback in Part 17 works.** Kill the Anthropic API — does the local fallback trigger? Kill the SLM — does the error propagate gracefully?
4. **Unit tests for numerical components** — feature extractors, attention, bandit posterior updates, encoder/decoder round-trips.
5. **Shape tests for every neural network component** — wrong-shape inputs produce informative errors, not mysterious crashes.

## 19.6 On Research and Citation

When you use a technique from the literature, cite it in code comments and in the eventual presentation slides. Examples:

```python
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Normalised Temperature-scaled Cross Entropy loss.

    Introduced in Chen et al. 2020 (SimCLR). Given two augmented views of
    the same examples, maximises mutual information between them by
    contrasting against in-batch negatives.

    Reference: https://arxiv.org/abs/2002.05709
    """
```

Slides that cite sources read as careful research. Slides that assert techniques without attribution read as half-knowledge. Matthew has ten years at Apple — he recognises the difference.

## 19.7 On the Final Deliverable

By end of Day 17 (April 27), you should have produced:

- A working repository at `github.com/abailey81/i3-implicit-interaction-intelligence` (public or private as Tamer prefers)
- A 15-slide deck (PDF + PPTX) at the repository root
- A backup demo video (MP4, 5 minutes) in a USB-drive-ready folder
- An edge profiling report (Markdown + supporting charts)
- A clean `README.md` that walks a reviewer from `git clone` to running demo in five steps
- A `NOTES.md` documenting any decisions that departed from the specification and why
- Trained model checkpoints with full metadata
- Unit tests, integration tests, and a test runner that passes green

**Every commit message should read as if Matthew will read it.** Because he might.

## 19.8 On the Meta-Goal

This project is career-defining for Tamer. Build the most impressive, technically deep, most honest version you can. If you see an opportunity to go beyond — take it. The 17-day timeline is a constraint, not a ceiling — within that budget, push for the highest-ambition version that still ships on time.

**The test:** when Matthew reads the slides on April 28, he should think "I need this person in my lab." When he sees the live demo on April 29, that conviction becomes certainty.

**Begin when you are ready. Build the best version of this you can.** Every decision — from variable naming to architecture — is a chance to signal quality and craft. Take the chances that matter, skip the ones that don't, and deviate from this document whenever your judgement says a better path exists.

---

# APPENDIX A: KEY URLS & REFERENCES

## A.1 Interview Contacts & Locations

- Huawei Technologies Research & Development (UK) Ltd
- Address: 1 Pancras Square, London N1C 4AG, United Kingdom
- Matthew Riches (Hiring Manager): matthew.riches@huawei.com
- Vicky Li (TA Specialist): initial screening contact
- Mingwai Li: interview logistics contact

## A.2 Huawei Products & Research References

- HarmonyOS 6 public beta announcement (October 2025)
- Smart Hanhan product page (Huawei store, November 2025 launch)
- Huawei AI Glasses launch (April 21, 2026)
- Huawei Connect 2024 keynote (Eric Xu) — "experience, not computing power"
- Huawei-Edinburgh Joint Lab talk: Malvina Nissim, "Style and Interaction in Large Language Model Personalisation" (March 10, 2026)
- PanGu Model Family documentation
- MindSpore Lite documentation
- L1–L5 Intelligence Framework (Huawei + Tsinghua Institute for AI Industry Research)

## A.3 Research Papers (Cited in Architecture)

- Chen, Kornblith, Norouzi, Hinton (2020). "A Simple Framework for Contrastive Learning of Visual Representations." (NT-Xent loss) — arxiv.org/abs/2002.05709
- Bai, Kolter, Koltun (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." (TCN architecture) — arxiv.org/abs/1803.01271
- Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." (Pre-LN) — arxiv.org/abs/2002.04745
- Vaswani et al. (2017). "Attention Is All You Need." — arxiv.org/abs/1706.03762
- Russo et al. (2018). "A Tutorial on Thompson Sampling." — arxiv.org/abs/1707.02038
- Chapelle & Li (2011). "An Empirical Evaluation of Thompson Sampling." (NIPS 2011)
- Rabiner (1989). "A Tutorial on Hidden Markov Models and Selected Applications." (Proceedings of the IEEE)

## A.4 Datasets

- DailyDialog — huggingface.co/datasets/daily_dialog
- EmpatheticDialogues — huggingface.co/datasets/empathetic_dialogues
- NRC Emotion Lexicon — saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

## A.5 Libraries & Tools

- PyTorch 2.x — foundational deep learning framework
- FastAPI — backend server framework
- Uvicorn — ASGI server
- SQLite — local persistence
- Pydantic — configuration validation
- cryptography (Fernet) — symmetric encryption
- structlog — structured logging
- pytest — testing framework
- numpy, scipy — numerical computing

## A.6 Candidate's Existing Assets (Referenced Throughout)

- GitHub profile: github.com/abailey81
- LinkedIn: linkedin.com/in/tamerates
- Crypto-Statistical-Arbitrage repository
- FinTwin repository (70K LOC, RL framework for banking agents)
- Conversational AI Platform (Telegram deployment)
- MSc Finance & AI coursework (transformer from scratch on tabular data)

---

# APPENDIX B: GLOSSARY

- **Adaptation Vector** — 4-dimensional output of the AdaptationController encoding cognitive load, communication style, emotional tone, and accessibility needs.
- **AdaptiveSLM** — The custom small language model with cross-attention conditioning on the adaptation vector.
- **Archetype** — One of 8 prototypical user states used to generate synthetic training data.
- **Bandit (contextual)** — A reinforcement learning model that selects among actions (arms) based on a context vector, learning online from reward feedback.
- **Calibration phase** — First ~5 messages of a new user session, during which the user model bootstraps from population average.
- **Causal convolution** — A 1D convolution where the output at time t depends only on inputs at times ≤ t, used in the TCN to preserve temporal causality.
- **Cross-attention** — An attention mechanism where queries come from one sequence and keys/values from another, used for conditioning generation on external information.
- **Deviation metric** — Scalar measure of how far the current user state is from the user's long-term baseline.
- **Edge profiling** — Measurement of model memory footprint and inference latency extrapolated to constrained hardware.
- **EMA (Exponential Moving Average)** — A running average that weights recent observations more heavily, used in the user model's session and long-term states.
- **Fernet encryption** — Symmetric authenticated encryption scheme from the `cryptography` library, used for embedding encryption at rest.
- **HMAF (Harmony Multi-Agent Framework)** — HarmonyOS 6's architecture for coordinating AI agents across devices.
- **I³** — Implicit Interaction Intelligence, the project name.
- **InteractionFeatureVector** — 32-dimensional feature vector extracted per message from keystroke, linguistic, session, and deviation features.
- **INT8 quantisation** — Reducing model weights from 32-bit floating point to 8-bit integers for edge deployment, typically 4× memory reduction at modest quality cost.
- **Laplace approximation** — A technique for approximating a complex posterior distribution with a Gaussian centred at the mode.
- **MindSpore Lite** — Huawei's on-device inference framework (~2MB footprint).
- **NT-Xent** — Normalised Temperature-scaled Cross Entropy contrastive loss, used to train the TCN encoder.
- **Pre-LN** — Transformer architecture variant where layer normalisation is applied before attention/FFN blocks rather than after; improves training stability.
- **SLM (Small Language Model)** — A language model small enough to run on-device, typically under 1B parameters. The I³ SLM is 8–15M parameters.
- **Smart Hanhan** — Huawei's emotional AI companion product launched November 2025.
- **TCN (Temporal Convolutional Network)** — A neural architecture using dilated causal convolutions for sequence modelling, used as the user state encoder.
- **Thompson Sampling** — A Bayesian method for the exploration-exploitation trade-off in multi-armed bandits, sampling from posterior distributions over arm values.
- **Three-timescale user model** — Representation of the user at three temporal scales: instant state, session EMA, long-term baseline.
- **Tier 1 / Tier 2 / Tier 3** — The three model tiers described in the Huawei JD and implemented in I³: custom ML (TCN), SLM (custom transformer), foundation model (Claude API).
- **XiaoYi / Celia** — Huawei's AI assistant integrated across HarmonyOS.

---

# APPENDIX C: THE CONNECTION MATRIX — PROJECT COMPONENTS ↔ HUAWEI PRODUCTS

This matrix demonstrates how every component of I³ maps to a real Huawei product or research direction. Reference during Q&A if asked "how does this apply to your actual work?"

| I³ Component | Smart Hanhan | AI Glasses | HarmonyOS Agents | Edinburgh Joint Lab |
|---|---|---|---|---|
| TCN state encoder | Emotion detection from voice/touch dynamics | Gaze + head-motion state inference | Cross-device state aggregation | Sparse-signal encoder |
| Three-timescale user model | Personalisation over session and longer | Context-aware glass overlays | Distributed profile sync | Long/short-term user representation |
| Adaptation controller | Response tone matching user mood | UI density matching attention | Agent behaviour modulation | Style/affect conditioning |
| SLM with cross-attention | On-device warm response generation | Sub-second conversational overlay | Per-agent local reasoning | Efficient conditioning for SLMs |
| Contextual bandit router | Local vs cloud decision | Local vs paired-phone vs cloud | Agent-to-agent routing | Edge-cloud orchestration |
| Sensitive-topic classifier | Privacy for companion conversations | Bystander privacy on glasses | Agent privacy boundaries | Privacy-aware personalisation |
| INT8 quantisation | Fits 1800mAh budget | Fits 30g frame | Fits Kirin on-device | On-device inference |
| Interaction diary | Digital memories feature | Moment capture | Cross-device memory sync | User-trajectory modelling |
| Accessibility axis | Inclusive companion | Glasses for assistive use | Agent accessibility | Equitable personalisation |
| Three-tier architecture | LLM + local + framework | Edge + paired + cloud | HMAF structure | Multi-tier LLM research |

**How to use this table:** if asked "how does this apply to Smart Hanhan?" — find the column, read down. If asked "what's the accessibility story?" — find the row, read across. Every cell is a 30-second talking point.

---

# END OF DOCUMENT

This is the complete brief. Every piece of context, every decision, every specification, every rehearsed answer, every contingency plan for the Huawei HMI Lab interview and the Implicit Interaction Intelligence build. Read it, then build.

**Claude Code: you have the brief. You have the latitude. Build the best version of this that you can.**

