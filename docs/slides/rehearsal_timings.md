# Rehearsal Cue-Sheet — I³ Presentation

30-minute target. Demo opens at roughly minute 11:15, lands at 21:15.
Cumulative times below assume **no panel interruptions**. Total lands at
**29:45** — 15-second buffer built in for the verbatim closing pause.
Budget 2–3 minutes of additional slippage for interruptions; recovery
tactics are listed per slide.

## Master timing table

| #  | Slide                                              | Target   | Cumulative |
|----|----------------------------------------------------|----------|------------|
| 1  | Hook — The person who already knows you're tired    | `1:15`   | `1:15`     |
| 2  | Tension — Your phone does not notice                | `1:15`   | `2:30`     |
| 3  | Context — Why this lab, why now                     | `1:30`   | `4:00`     |
| 4  | Promise — What I will show in 30 minutes            | `1:00`   | `5:00`     |
| 5  | Architecture — Seven layers, one sentence each      | `1:15`   | `6:15`     |
| 6  | Encoder — Listening to how you type                 | `1:30`   | `7:45`     |
| 7  | User Model — what "normal" looks like               | `1:30`   | `9:15`     |
| 8  | Cross-attention conditioning — the centrepiece      | `2:00`   | `11:15`    |
| 9  | Live Demo (intro + 4 phases)                        | `10:00`  | `21:15`    |
| 10 | Routing — Contextual Thompson sampling              | `1:30`   | `22:45`    |
| 11 | Privacy by architecture                             | `1:15`   | `24:00`    |
| 12 | Fits the devices — extrapolated, honestly           | `1:15`   | `25:15`    |
| 13 | What this prototype is *not*                        | `2:00`   | `27:15`    |
| 14 | Where this goes next                                | `1:30`   | `28:45`    |
| 15 | Close                                               | `1:00`   | `29:45`    |

**Buffer:** 15 s against the 30-minute hard stop, plus informal 2–3 min of
panel-interruption absorption that should be taken out of the Q&A slot
rather than the presentation itself.

---

## Per-slide cues

### Slide 1 — Hook (75 s)

- **Target:** 75 s. **Cumulative:** 1:15.
- **Must-say:**
  - *"Thanks for having me — 30-minute technical presentation; I'll happily take questions throughout, but I'd suggest saving most for the end so I can show you the full arc."*
  - *"Think about the person who knows you well enough to notice you're tired from the pace of your typing — without you saying so."*
  - *"They adapt to how you say it, not just what you say."*
- **Likely interruption:** "Is this emotion detection?" → *"No — dynamics, not content; slide 6 is the specifics."* Move on, 10-second cost.
- **Recovery without losing time:** if the projector flickers, keep talking to the laptop screen; continue on Slide 2 as soon as it appears.

### Slide 2 — Tension (75 s)

- **Target:** 75 s. **Cumulative:** 2:30.
- **Must-say:**
  - *"Siri answers the words, not the pace."*
  - *"ChatGPT is the same today as yesterday — you changed; it didn't."*
  - *"The signals exist in every keystroke — nobody is reading them."*
- **Likely interruption:** "Isn't ChatGPT memory doing this?" → *"Explicit, policy-driven, not behavioural — fair nuance. Let me continue and we'll return if it's still open."* 15-second cost.
- **Recovery:** if the panel looks unpersuaded, add one beat: *"The UI home for state today is Settings — that is the wrong home."* 8 seconds; then advance.

### Slide 3 — Context (90 s)

- **Target:** 90 s. **Cumulative:** 4:00.
- **Must-say:**
  - *"HarmonyOS 6 + HMAF — large, small, agent core. The three-tier mirror is deliberate."*
  - *"Smart Hanhan — November 2025, 64 MB class. AI Glasses — 21 April 2026."*
  - *"Edinburgh Joint Lab — 'sparse or implicit signals,' Nissim, 10 March."*
  - *"Eric Xu — 'experience, not computing power.'"*
  - *"The fit is not accidental."*
- **Likely interruption:** "What's your source for Hanhan specs?" → *"consumer.huawei.com, November 2025 press page — 399 RMB, 1800 mAh, I'll share the URL."* 10-second cost.
- **Recovery:** if challenged on AI Glasses launch date, concede quickly: *"You're right — launch window rather than launch day; the point stands."*

### Slide 4 — Promise (60 s)

- **Target:** 60 s. **Cumulative:** 5:00.
- **Must-say:**
  - *"Four axes at once — cognitive load, style, tone, accessibility."*
  - *"Runs fully on this laptop, live, end-to-end."*
  - *"The user never explains themselves — not once."*
  - *"Honesty slide near the end."*
- **Likely interruption:** rare — panels hold fire here for the fulfilment.
- **Recovery:** if the laptop wakes late, keep talking; the demo slot is still minutes away.

### Slide 5 — Architecture (75 s)

- **Target:** 75 s. **Cumulative:** 6:15.
- **Must-say:**
  - *"Seven layers, one sentence each."*
  - *"Perception → Encoder → User Model → Adaptation → Router → Generation → Diary."*
  - *"The next three slides spotlight the layers I'm most proud of."*
- **Likely interruption:** "Why seven?" → *"Empirical — this is where concerns separated cleanly. Maps to the codebase directly."* 15-second cost.
- **Recovery:** if Matthew zooms in on a layer, answer briefly and point forward: *"Slide 8 covers that in depth — let's land there."*

### Slide 6 — Perception + Encoder (90 s)

- **Target:** 90 s. **Cumulative:** 7:45.
- **Must-say:**
  - *"32 features, four groups of eight — dynamics, content, session, deviation."*
  - *"Dilated causal convolutions, kernel three, dilations one, two, four, eight."*
  - *"Receptive field 31 messages — that's the Bai, Kolter, Koltun 2018 formula."*
  - *"NT-Xent contrastive training — Chen 2020, SimCLR."*
- **Likely interruption:** "Why TCN not Transformer?" → *"Short version: fixed memory per inference, interpretable receptive field, under 500K parameters. Q&A has the full answer."* 20-second cost.
- **Recovery:** if the formula renders poorly, cite it verbally and move on.

### Slide 7 — User Model (90 s)

- **Target:** 90 s. **Cumulative:** 9:15.
- **Must-say:**
  - *"Three timescales — instant, session, long-term — because three kinds of decision."*
  - *"Single EMA collapses these and loses structure."*
  - *"Welford online stats — O(1) per update."*
  - *"Baseline establishes after five messages — flag flips live in the demo."*
- **Likely interruption:** "Why not Kalman filter?" → *"Honest answer: no dynamics model evidence yet; EMA is the baseline. Candidate for phase two."* 15-second cost.
- **Recovery:** if running long, drop the Welford mention — it is nice-to-have, not load-bearing.

### Slide 8 — Cross-attention conditioning (120 s)

- **Target:** 120 s. **Cumulative:** 11:15.
- **Must-say:**
  - *"Not prepending a prompt. Not fine-tuning per user."*
  - *"Four conditioning tokens, cross-attended at every layer."*
  - *"O(4·N·d) per layer — under 5% of self-attention cost."*
  - *"Auxiliary loss forces the model to use the conditioning."*
  - *"Still a partial solution — slide 13 names the limit."*
- **Likely interruption:** "Isn't this CLIP-guided?" → *"Same family — dedicated cross-attention at transformer-block granularity."* 15-second cost.
- **Recovery:** if the annotated code block is unreadable, refer to the handout.

### Slide 9 — Live Demo (600 s)

- **Target:** 600 s (2 min intro + 8 min demo). **Cumulative:** 21:15.
- **Must-say, per phase:**
  - **Cold start:** *"Baseline not yet established — flag is red. Router favours cloud."* Then: *"That flipped. Baseline established."*
  - **Energetic:** *"Load rising. Style formality climbing. Response matches register."*
  - **Fatigue (key moment):** *"Embedding dot migrating. Load dropping. Response shorter, warmer. Router just flipped to local."*
  - **Accessibility (values moment):** *"Accessibility gauge lifting. Yes/no register. Simpler vocab. No settings menu. No toggle. It just adapts."*
- **Likely interruption:** "Is that really live, not scripted?" → *"Live. The backend is FastAPI on localhost — you can watch the network tab if we have a second monitor."* 20-second cost.
- **Recovery — API call fails:** *"Cloud just dropped — router flipped to local, fallback path. Adaptation layer unaffected. We're seeing exactly what a flaky network looks like."*
- **Recovery — a gauge doesn't move:** *"That one's lagging — Welford takes a few samples to confirm. Embedding dot is the leading indicator."*

### Slide 10 — Routing (90 s)

- **Target:** 90 s. **Cumulative:** 22:45.
- **Must-say:**
  - *"Two arms, 12-dim context."*
  - *"Bayesian logistic regression + Laplace + Newton–Raphson refit every 10 steps."*
  - *"Privacy override — sensitive topic masks the cloud arm to zero."*
  - *"Russo 2018, Chapelle and Li 2011."*
- **Likely interruption:** "Why not UCB?" → *"UCB's confidence bounds are hard in 12-dim continuous context; Thompson handles it via sampling."* 15-second cost.
- **Recovery:** if running long, drop the Chapelle citation but keep Russo.

### Slide 11 — Privacy (75 s)

- **Target:** 75 s. **Cumulative:** 24:00.
- **Must-say:**
  - *"Raw text is never stored — enforced at the storage layer, no toggle."*
  - *"Fernet-encrypted embeddings, 10 PII regex patterns, sensitive-topic classifier."*
  - *"Fernet is a placeholder for TrustZone — honest caveat."*
- **Likely interruption:** "Can you reconstruct text from embeddings?" → *"Not under this encoder — trained on dynamics, not content. Still lossy-not-zero; slide 13 owns that."* 20-second cost.
- **Recovery:** if the diary schema question comes up, defer: *"It's in the handout — no `message` column, just `embedding BLOB` and `adaptation JSON`."*

### Slide 12 — Edge feasibility (75 s)

- **Target:** 75 s. **Cumulative:** 25:15.
- **Must-say:**
  - *"Full pipeline ~7 MB INT8; TCN under 1 MB, SLM around 6."*
  - *"Laptop P50 measured at 150–220 ms."*
  - *"Kirin numbers are extrapolated, not measured — named explicitly."*
  - *"Conversion target is MindSpore Lite via ONNX; re-calibrate INT8 inside MS Lite."*
- **Likely interruption:** "Where's measured Kirin data?" → *"Absent — extrapolation is the honest alternative; I'd run real profiling day-one in the lab."* 15-second cost.
- **Recovery:** if a table cell looks off, concede quickly — the extrapolation caveat carries the slide on its own.

### Slide 13 — What this prototype is *not* (120 s)

- **Target:** 120 s. **Cumulative:** 27:15.
- **Must-say (in order on the slide):**
  - *"Not a shipped product."*
  - *"Not trained on real user data."*
  - *"Not a strong SLM."*
  - *"Not universal accessibility detection — must stay opt-out capable."*
  - *"Embeddings are lossy, not zero-information."*
  - *"Keystroke-only; the TCN itself is modality-agnostic."*
- **Likely interruption:** *"Why admit so much?"* → *"I'd rather name the limits than have you find them later. Your BS radar is better than my salesmanship."* 10-second cost, positive signal.
- **Recovery — Matthew challenges accessibility claim:** *"You're right — an accessibility lead needs to review this before any user-facing trial. That's in scope before shipping."*
- **Matthew values moment:** hold eye contact through the accessibility bullet.

### Slide 14 — Where this goes next (90 s)

- **Target:** 90 s. **Cumulative:** 28:45.
- **Must-say:**
  - *"L1 to L5 — prototype lives at L2–L3."*
  - *"L4 is device-to-device handover via HarmonyOS distributed databus."*
  - *"Federated long-term profile — MindSpore Federated."*
  - *"Multi-modal is a drop-in — the TCN is modality-agnostic."*
  - *"Keystroke biometrics, fairness eval, interpretability panel, ablation toggle — in scope next."*
- **Likely interruption:** "How long to L4?" → *"Engineering, not research — the distributed data management part is HarmonyOS's job already."* 15-second cost.
- **Recovery:** if running long, drop "keystroke biometrics" to save 8 seconds.

### Slide 15 — Close (60 s)

- **Target:** 60 s. **Cumulative:** 29:45.
- **Must-say — verbatim, no paraphrasing:**
  - *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*
- **Must-say — follow-on:**
  - *"Thank you — happy to take questions, and I have three prepared for the candidate-Q&A slot."*
- **Likely interruption:** none — this is the panel's turn.
- **Recovery:** if the closing line comes out flat, do not repeat it. Move to the thank-you at normal cadence.

---

## Slippage strategy

If running over at Slide 9 completion (cumulative > 22:00):

1. Cut Slide 10's Chapelle citation (saves 5 s).
2. Cut Slide 14's ablation-toggle mention (saves 8 s).
3. Cut Slide 13's embeddings-lossy bullet only if honesty slide is still landing (saves 10 s).
4. **Never cut the honesty slide or the closing line.**

If running under at Slide 9 completion (cumulative < 20:45):

1. Add a second beat on the accessibility phase in the demo, re-narrating the "no toggle" line.
2. Add a sentence on Slide 13 linking accessibility detection back to an ARIA-compliant inclusive-design practice — without naming TextSpaced explicitly.
