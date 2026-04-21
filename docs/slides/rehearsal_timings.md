# Rehearsal Cue-Sheet — I³ Presentation

30-minute target. Demo opens at roughly minute 10, lands at minute 22.
Cumulative times below assume **no panel interruptions**. Budget 2–3
minutes total for interruption slippage; recovery tactics are listed
per slide.

## Master timing table

| # | Slide                                       | Target | Cumulative |
|---|---------------------------------------------|--------|------------|
| 1 | Title — "Noticing you, without asking"      | `0:45` | `0:45`     |
| 2 | The person who already knows you're tired   | `1:00` | `1:45`     |
| 3 | Your phone does not notice                  | `1:15` | `3:00`     |
| 4 | Why this lab, why now                       | `1:30` | `4:30`     |
| 5 | What I will show in 30 minutes              | `1:00` | `5:30`     |
| 6 | Seven layers, one sentence each             | `1:30` | `7:00`     |
| 7 | Listening to how you type                   | `1:30` | `8:30`     |
| 8 | The User Model                              | `1:30` | `10:00`    |
| 9 | Cross-attention conditioning                | `2:00` | `12:00`    |
| 10| Live Demo (intro + 4 phases)                | `10:00`| `22:00`    |
| 11| Routing — contextual Thompson sampling      | `1:30` | `23:30`    |
| 12| Privacy by architecture                     | `1:15` | `24:45`    |
| 13| What this prototype is *not*                | `2:00` | `26:45`    |
| 14| Where this goes next                        | `1:15` | `28:00`    |
| 15| Close                                       | `1:00` | `29:00`    |

**Budget remaining:** 1:00 buffer for interruption slippage and the
verbatim closing pause.

---

## Per-slide cues

### Slide 1 — Title (45 s)

- **Target:** 45 s. **Cumulative:** 0:45.
- **Must-say:**
  - *"Thanks for having me."*
  - *"30-minute technical presentation — I'll happily take questions throughout, but I'd suggest saving most for the end so I can show you the full arc."*
- **Likely interruption:** none — panel is still settling.
- **Recovery without losing time:** if the projector mirrors late, keep talking to the laptop screen; continue on Slide 2 as soon as it appears.

### Slide 2 — Hook (60 s)

- **Target:** 60 s. **Cumulative:** 1:45.
- **Must-say:**
  - *"Think about the person who knows you well enough to notice you're tired from the pace of your typing — without you saying so."*
  - *"They adapt to how you say it, not just what you say."*
- **Likely interruption:** "Is this emotion detection?" → answer: *"No — dynamics, not content; slide 7 is the specifics."* Move on.
- **Recovery:** if the quote lands flat, do not re-read — continue; the hook is cumulative with slides 3–4.

### Slide 3 — Tension (75 s)

- **Target:** 75 s. **Cumulative:** 3:00.
- **Must-say:**
  - *"Siri answers the words, not the pace."*
  - *"The signals exist in every keystroke — nobody is reading them."*
- **Likely interruption:** "Isn't ChatGPT memory doing this?" → *"Explicit, policy-driven, not behavioural — fair nuance. Let me continue and we'll return if it's still open."* 15-second cost.
- **Recovery:** if the panel looks unpersuaded, add one beat: *"The UI home for state today is Settings — that is the wrong home."* 8 seconds; then advance.

### Slide 4 — Context (90 s)

- **Target:** 90 s. **Cumulative:** 4:30.
- **Must-say:**
  - *"HarmonyOS 6 + HMAF — large, small, agent core. The three-tier mirror is deliberate."*
  - *"Smart Hanhan — November 2025, 64 MB class. AI Glasses — 21 April 2026."*
  - *"Edinburgh Joint Lab — 'sparse or implicit signals,' Nissim, 10 March."*
  - *"Eric Xu — 'experience, not computing power.'"*
  - *"The fit is not accidental."*
- **Likely interruption:** "What's your source for Hanhan specs?" → *"consumer.huawei.com, November 2025 press page — I'll share the URL."* 10-second cost.
- **Recovery:** if challenged on AI Glasses ("that's not 8 days ago, it's 10"), concede quickly: *"You're right — launch window rather than launch day; the point stands."*

### Slide 5 — Promise (60 s)

- **Target:** 60 s. **Cumulative:** 5:30.
- **Must-say:**
  - *"Four axes at once — cognitive load, style, tone, accessibility."*
  - *"Runs fully on this laptop, live, end-to-end."*
  - *"The user never explains themselves — not once."*
  - *"Honesty slide near the end."*
- **Likely interruption:** none — this slide is a promise; panels hold fire for the fulfilment.
- **Recovery:** if the laptop wakes late, keep talking; the demo slot is minutes away.

### Slide 6 — Architecture (90 s)

- **Target:** 90 s. **Cumulative:** 7:00.
- **Must-say:**
  - *"Seven layers, one sentence each."*
  - *"Perception → Encoder → User Model → Adaptation → Router → Generation → Diary."*
  - *"The next three slides spotlight the layers I'm most proud of."*
- **Likely interruption:** "Why seven?" → *"Empirical — this is where concerns separated cleanly. Maps to the codebase directly."* 15-second cost.
- **Recovery:** if Matthew zooms in on a specific layer, answer briefly, then: *"Slide seven covers that in depth — let's land there."*

### Slide 7 — Perception + Encoder (90 s)

- **Target:** 90 s. **Cumulative:** 8:30.
- **Must-say:**
  - *"32 features, four groups of eight — dynamics, content, session, deviation."*
  - *"Dilated causal convolutions, kernel three, dilations one, two, four, eight."*
  - *"Receptive field 31 messages — that's the Bai, Kolter, Koltun 2018 formula."*
  - *"NT-Xent contrastive training — Chen 2020, SimCLR."*
- **Likely interruption:** "Why TCN not Transformer?" → *"Short version: fixed memory per inference, interpretable receptive field, under 500K parameters. Q&A has the full answer."* 20-second cost.
- **Recovery:** if the formula renders poorly, cite it verbally and move on.

### Slide 8 — User Model (90 s)

- **Target:** 90 s. **Cumulative:** 10:00.
- **Must-say:**
  - *"Three timescales — instant, session, long-term — because three kinds of decision."*
  - *"Single EMA collapses these and loses structure."*
  - *"Welford online stats — O(1) per update."*
  - *"Baseline establishes after five messages — flag flips live in the demo."*
- **Likely interruption:** "Why not Kalman filter?" → *"Honest answer: no dynamics model evidence yet; EMA is the baseline. Candidate for phase two."* 15-second cost.
- **Recovery:** if running long here, drop the Welford mention — it is nice-to-have, not load-bearing.

### Slide 9 — Cross-attention (120 s)

- **Target:** 120 s. **Cumulative:** 12:00.
- **Must-say:**
  - *"Not prepending a prompt. Not fine-tuning per user."*
  - *"Four conditioning tokens, cross-attended at every layer."*
  - *"O(4·N·d) per layer — under 5% of self-attention cost."*
  - *"Auxiliary loss forces the model to use the conditioning."*
  - *"This is still a partial solution — slide 13 names the limit."*
- **Likely interruption:** "Isn't this CLIP-guided?" → *"Same family — dedicated cross-attention at transformer-block granularity."* 15-second cost.
- **Recovery:** if the annotated code block is unreadable, refer to the handout.

### Slide 10 — Live Demo (600 s)

- **Target:** 600 s (2 min intro + 8 min demo). **Cumulative:** 22:00.
- **Must-say, per phase:**
  - **Cold start:** *"Baseline not yet established — flag is red. Router favours cloud."* Then: *"That flipped. Baseline established."*
  - **Energetic:** *"Load rising. Style formality climbing. Response matches register."*
  - **Fatigue (key moment):** *"Embedding dot migrating. Load dropping. Response shorter, warmer. Router just flipped to local."*
  - **Accessibility (values moment):** *"Accessibility gauge lifting. Yes/no register. Simpler vocab. No settings menu. No toggle. It just adapts."*
- **Likely interruption:** "Is that really live, not scripted?" → *"Live. The backend is FastAPI on localhost — you can watch the network tab on the right monitor if I have one."* 20-second cost.
- **Likely interruption:** "What if the API call fails?" → *"Router falls back to local automatically. Watch — I'll deliberately drop Claude in the last minute."* Do not actually drop unless asked.
- **Recovery — API call fails:** *"Cloud just dropped — router flipped to local, fallback path. Adaptation layer unaffected. We're seeing exactly what a flaky network looks like."*
- **Recovery — a gauge doesn't move:** *"That one's lagging — Welford takes a few samples to confirm. Embedding dot is the leading indicator."*

### Slide 11 — Routing (90 s)

- **Target:** 90 s. **Cumulative:** 23:30.
- **Must-say:**
  - *"Two arms, 12-dim context."*
  - *"Bayesian logistic regression + Laplace + Newton–Raphson refit every 10 steps."*
  - *"Privacy override — sensitive topic masks the cloud arm to zero."*
  - *"Russo 2018, Chapelle and Li 2011 — Thompson beats UCB and epsilon-greedy on asymmetric-cost problems."*
- **Likely interruption:** "Why not UCB?" → *"UCB's confidence bounds are hard in 12-dim continuous context; Thompson handles it via sampling."* 15-second cost.
- **Recovery:** if running long, drop the Chapelle citation mention but keep the Russo one.

### Slide 12 — Privacy (75 s)

- **Target:** 75 s. **Cumulative:** 24:45.
- **Must-say:**
  - *"Raw text is never stored — enforced at the storage layer, no toggle."*
  - *"Fernet-encrypted embeddings, 10 PII regex patterns, sensitive-topic classifier."*
  - *"Fernet is a placeholder for TrustZone — honest caveat."*
- **Likely interruption:** "Can you reconstruct text from embeddings?" → *"No under this encoder — trained on dynamics, not content. Still lossy-not-zero; slide 13 owns that."* 20-second cost.
- **Recovery:** if the diary schema question comes up here, defer: *"Slide 14 has where this goes; the schema's in the handout."*

### Slide 13 — What this prototype is *not* (120 s)

- **Target:** 120 s. **Cumulative:** 26:45.
- **Must-say:**
  - *"Not a shipped product."*
  - *"Not trained on real user data."*
  - *"Not a strong SLM."*
  - *"Accessibility detection is one signal among many — must stay opt-out capable."*
  - *"Embeddings are lossy, not zero-information."*
  - *"Keystroke-only; the TCN itself is modality-agnostic."*
- **Likely interruption:** *"Why admit so much?"* → *"Because I'd rather name the limits than have you find them later. And your BS radar is better than my salesmanship."* 10-second cost, positive signal.
- **Recovery — Matthew challenges accessibility claim:** *"You're right — I'd want an accessibility lead to review this before any user-facing trial. That's in scope before shipping."*
- **The Matthew values moment: hold eye contact through the accessibility bullet.**

### Slide 14 — Where this goes next (75 s)

- **Target:** 75 s. **Cumulative:** 28:00.
- **Must-say:**
  - *"L1 to L5 — prototype lives at L2–L3."*
  - *"L4 is device-to-device handover via HarmonyOS distributed databus."*
  - *"Federated long-term profile — MindSpore Federated."*
  - *"Multi-modal is a drop-in — the TCN is modality-agnostic."*
  - *"Keystroke biometrics, fairness eval, interpretability panel, ablation toggle — in scope next."*
- **Likely interruption:** "How long to L4?" → *"Engineering, not research — the distributed data management part is HarmonyOS's job already."* 15-second cost.
- **Recovery:** if running long, drop "keystroke biometrics" to save 8 seconds.

### Slide 15 — Close (60 s)

- **Target:** 60 s. **Cumulative:** 29:00.
- **Must-say — verbatim, no paraphrasing:**
  - *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*
- **Must-say — follow-on:**
  - *"Thank you — happy to take questions."*
- **Likely interruption:** none — this is the panel's turn.
- **Recovery:** if the closing line comes out flat, do not repeat it. Move to the thank-you at normal cadence.

---

## Slippage strategy

If running over at Slide 10 completion (cumulative > 22:30):

1. Cut Slide 11's Chapelle citation (saves 5 s).
2. Cut Slide 14's ablation-toggle mention (saves 8 s).
3. Cut Slide 13's embeddings-lossy bullet only if honesty slide is still landing (saves 10 s).
4. **Never cut the honesty slide or the closing line.**

If running under at Slide 10 completion (cumulative < 21:30):

1. Add a second beat on the accessibility phase in the demo, re-narrating the "no toggle" line.
2. Add a sentence on Slide 13 linking accessibility detection back to TextSpaced's ARIA-compliant design without naming TextSpaced explicitly.
