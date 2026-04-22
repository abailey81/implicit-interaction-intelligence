# Speaker Notes — Implicit Interaction Intelligence (I³)

One H2 per slide. Target 60–90 seconds of speech, roughly 150–200 words.
Open each section with the target time, include 2–3 beat moments, and
include two recovery lines (API failure, gauge failure) so that anything
glitching on stage has a pre-drilled response.

---

## Slide 1 — Hook: The person who already knows you're tired (75 s)

**Target:** 75 s (includes brief opener). **Cumulative:** 01:15.

Walk in, plug in, let the slide settle for a full breath before speaking.
No rushing. **[Beat — eye contact across the panel.]**

Open:

> *"I'm Tamer. Thanks for having me. I've got a 30-minute technical
> presentation on a system I've built. I'll happily take questions throughout
> if anything's unclear — but if it's OK with you, I'd suggest saving most of
> them for the end so I can show you the full arc."*

**[Beat.]** Then advance into the hook itself. Read the pull quote slowly,
looking at the panel, not the screen. Let the word "tired" hang.

> *"Think about the person who knows you well enough to notice you're tired
> from the pace of your typing — without you saying so."*

Then, in your own words:

> *"That person adapts to how you speak, not just what you said. They notice
> when today is different from yesterday. They get quieter when you get
> quieter. They never ask you to declare their state."*

**[Beat — let that line land.]** This is the emotional hook — the whole
deck compressed. Do not say "ML" or "transformer" yet.

**Recovery lines.**
- *If the projector flickers:* "Give me five seconds — worst-case we present from the laptop screen; I've rehearsed both."
- *If the laptop fan spikes or a gauge flashes:* ignore it, keep pace — the hook works whether anything else renders or not.

---

## Slide 2 — Tension: Your phone does not notice (75 s)

**Target:** 75 s. **Cumulative:** 02:30.

The point here is specific enough to be unarguable.

> *"Siri answers your words — not the pace, the pauses, or the edits. That
> is by design; it has no access to those signals. ChatGPT is exactly the
> same today as it was yesterday. You are the one who changed; the system
> didn't."*

**[Beat.]** Then close the tension:

> *"The signals exist. They're in every keystroke. They're not being read."*

The second half of the slide names where state currently lives in products —
Settings. That is the wrong place. Settings are explicit and declarative;
state is implicit and continuous. The field has been trying to solve an
implicit problem with an explicit UI.

Do not bash competitors gratuitously. Two of the examples are Apple
products. Stay clinical — the claim is about the
interaction layer, not the companies.

**Recovery lines.**
- *If the footer citation link is broken:* "I've footnoted every product claim — if you want the exact source I'll send it."
- *If asked "ChatGPT is exactly the same":* "Fair — with memory on, less so. But the memory is explicit and policy-driven, not behavioural."

---

## Slide 3 — Context: Why this lab, why now (90 s)

**Target:** 90 s. **Cumulative:** 04:00.

Five bullets, each a named fact with a public source. Walk them slowly —
this is the "I have done my homework" slide.

> *"HarmonyOS 6 and HMAF name three tiers explicitly — large model, small
> model, agent core. The three-tier architecture in this prototype is a
> deliberate mirror of that, not a coincidence."*

**[Beat.]**

> *"Smart Hanhan launched in November 2025 — 64 MB memory class, 1800 mAh
> battery. AI Glasses launched eight days ago, on 21 April. Both need the
> same thing: a compact user-model layer that works without shipping every
> utterance to the cloud."*

> *"Edinburgh Joint Lab's March talk named 'sparse or implicit signals' —
> verbatim — as the research direction. And Eric Xu's framing — 'experience,
> not computing power' — tells you what success looks like."*

**[Beat.]** Close: *"The fit is not accidental."*

**Recovery lines.**
- *If a panelist says "launch date wrong":* "Thank you — I'll double-check; the relevant point stands either way."
- *If challenged on Hanhan specs:* "399 RMB, Nov 2025, XiaoYi inside — I cited consumer.huawei.com; I'll share the URL."

---

## Slide 4 — Promise: What I will show in 30 minutes (60 s)

**Target:** 60 s. **Cumulative:** 05:00.

Short and active. No hedging verbs.

> *"What I'll show is a working system — not a design, not a deck of
> screenshots — that builds a user model from how you type. It adapts
> across four axes at once: cognitive load, communication style, emotional
> tone, accessibility. It runs end-to-end on this laptop, live, in about
> eight minutes."*

**[Beat.]**

> *"Live demo at minute twelve. The user never explains themselves — not
> once. And near the end, there's an honesty slide. Because you will ask
> what this doesn't do, and I'd rather tell you first."*

This slide sets three contracts with the panel: (1) a working system will
be demonstrated, (2) four simultaneous axes, (3) explicit honesty.
Deliver those in order and they will spend the rest of the session
checking each.

**Recovery lines.**
- *If the demo laptop has been asleep too long:* "Let me wake the backend before we move on — three seconds."
- *If asked "just four axes?":* "Yes — more would correlate. Slide seven explains the pruning."

---

## Slide 5 — Architecture: Seven layers, one sentence each (75 s)

**Target:** 75 s. **Cumulative:** 06:15.

This slide is a map, not a deep-dive. Resist the urge to explain everything.

> *"Seven layers. Perception turns raw events into a feature vector. The
> encoder learns what state each pattern corresponds to. The user model
> tracks that state at three timescales. The adaptation layer decides what
> to do about it. The router decides where to compute. The generation layer
> actually speaks. And the diary records — embeddings only, never text."*

**[Beat — gesture to the table.]** *"One sentence per layer. The next
three slides spotlight the ones I'm most proud of."*

This is the slide where a panellist typically zooms in on one layer. If
someone interrupts, answer briefly and point forward to the spotlight slide.
Do not let him pull you off timing here; you have 23 minutes to protect.

**Recovery lines.**
- *If he asks "why seven?":* "Empirical — seven is what emerged as I separated concerns. The grouping maps cleanly to the codebase."
- *If the table renders misaligned:* "Don't worry about the alignment — I printed this; you have it in the handout."

---

## Slide 6 — Spotlight: Listening to how you type (90 s)

**Target:** 90 s. **Cumulative:** 07:45.

The first technical substance slide. Two things have to land: the features
are behavioural, and the encoder is built from scratch.

> *"The feature vector is 32-dimensional, four groups of eight — keystroke
> dynamics, content, session, deviation. Not sentiment on text. Temporal
> dynamics of how the interaction unfolds."*

**[Beat.]**

> *"The encoder is a TCN — dilated causal convolutions, kernel three,
> dilations one, two, four, eight. Receptive field is 31 messages. That's
> the Bai, Kolter, Koltun 2018 receptive-field formula on the slide."*

> *"Trained with NT-Xent contrastive loss — Chen 2020, SimCLR. Positive
> pairs are augmentations of the same archetype sequence; negatives are
> everything else in the batch. The output is 64-dim, L2-normalised."*

**[Beat.]** Close: *"Why TCN not Transformer? Fixed memory per inference,
interpretable receptive field, under 500K parameters. I'll defend that in Q&A."*

**Recovery lines.**
- *If the formula rendering fails:* "The formula is standard — Bai 2018, equation 1."
- *If asked why not BiTransformer:* "Short sequences, edge target — the Transformer's long-range strengths don't earn their quadratic cost."

---

## Slide 7 — Spotlight: The User Model (90 s)

**Target:** 90 s. **Cumulative:** 09:15.

This slide sells the structural decision. Lead with why, not what.

> *"Three timescales — because three kinds of decision. Instant state drives
> this response, right now. Session EMA detects drift inside one
> conversation — is the user getting more tired as we talk? Long-term EMA
> is the baseline — is today unusual for this person?"*

**[Beat.]**

> *"A single EMA collapses these and loses structure. Three keeps it.
> It also mirrors how humans track each other — what did you just say,
> what's the vibe of this conversation, and is this how you usually are."*

> *"Welford online stats — no re-reading history, O(1) per update. Deviation
> z-scores feed back as Group 4 of the feature vector. And baseline
> establishes after five messages — you'll see the flag flip in the demo."*

**[Beat.]** *"That flip is the moment the system stops being a blank slate."*

**Recovery lines.**
- *If he asks "why not a Kalman filter":* "Candidate — heavier, needs a dynamics model I don't have evidence for yet. EMA is the honest baseline."
- *If "how is Welford better than batch mean":* "Numerical stability and O(1) memory — slide nine of Welford 1962 in the handout."

---

## Slide 8 — The intellectual centrepiece: Cross-attention conditioning (120 s)

**Target:** 120 s. **Cumulative:** 11:15.

This is the slide the whole deck has been building toward. Two minutes.
Do not hurry it.

> *"This is the novel contribution. Not prepending a system prompt. Not
> fine-tuning per user. Cross-attention from the token sequence to four
> conditioning tokens — and these tokens carry the user state."*

**[Beat — point at the code block.]**

> *"The AdaptationVector is 8-dimensional, the user-state embedding is
> 64-dimensional — 72 scalars of conditioning. A linear projector expands
> them to four tokens at d_model=256. Every transformer block cross-attends
> to those four tokens after self-attention and before the feed-forward."*

> *"The cost is O(4·N·d) per layer — under five percent of self-attention's
> quadratic cost. Cheap. What it buys is that the conditioning continuously
> modulates token probabilities at every layer of abstraction, not just at
> the start."*

**[Beat.]**

> *"An auxiliary loss penalises conditioning-agnostic outputs — that
> forces the model to actually use the conditioning, not route around it.
> Slide thirteen is where I admit this is still a partial solution."*

**Recovery lines.**
- *If asked "isn't this just CLIP-guided":* "Same family, yes — dedicated cross-attention for conditioning is the CLIP intuition applied at transformer-block granularity."
- *If the code panel is too small:* "The annotated version is in the handout — same three lines."

---

## Slide 9 — Live Demo (600 s / 10 min — intro 120 s + 8 min demo)

**Target:** 600 s. **Cumulative:** 21:15.

Open the browser. Say:

> *"Two-minute intro, then eight minutes of demo across four phases. The
> dashboard is your anchor — four gauges top right, embedding trail centre,
> routing confidence bottom left."*

**Phase 1 — Cold Start (120 s).**
Type three normal messages. Narrate: *"Baseline not yet established —
flag is red. Router is favouring cloud — we don't know this user yet."*
Fourth message. Fifth message. **[Beat — point at the flag.]** *"That
flipped. Baseline established. From here the system adapts."*

**Phase 2 — Energetic (60 s).**
Type fast, long, rich sentences. Narrate: *"Cognitive-load gauge rising.
Style formality climbing. Response is matching length and register — look
at the reply."* **[Beat.]**

**Phase 3 — Fatigue (120 s) — the key visibility moment.**
Slow down, shorten, simplify. Narrate: *"Watch the embedding dot migrate
— it's moving toward the 'tired' cluster. Cognitive-load gauge dropping.
Response gets shorter. Warmer. And — the router just flipped to local.
Latency matters more now than elegance."* **[Beat — let the gauges settle.]**

**Phase 4 — Accessibility (120 s) — the hardest-to-fake moment.**
Many backspaces, fragments, long pauses. Narrate: *"Accessibility gauge
lifting. System's responses are yes/no-shaped. Simpler vocabulary. Shorter.
No settings menu. No toggle. It just adapts."* **[Long beat.]** *"This is
`user modeling` in the JD sense — shaping the interaction from how the
person is typing, not from a profile they had to fill in."*

**Recovery lines.**
- *If the API call fails:* "Cloud dropped — router switches to local automatically; that's the fallback path. The adaptation layer is unaffected."
- *If a gauge doesn't move:* "That one's lagging — Welford takes a few samples to confirm the shift. Watch the embedding dot; it's the leading indicator."

---

## Slide 10 — Routing: Contextual Thompson Sampling (90 s)

**Target:** 90 s. **Cumulative:** 22:45.

Post-demo, the panel's attention is highest. Use it.

> *"Two arms — local SLM and cloud Claude. Twelve-dimensional context — user
> state summary, query complexity, topic sensitivity, patience, session
> progress, baseline-established flag, previous route, previous engagement,
> time-of-day, message count, cloud-latency estimate, SLM confidence."*

**[Beat.]**

> *"Bayesian logistic regression per arm. Laplace approximation gives the
> Gaussian posterior. Newton–Raphson MAP refit every ten steps — that's
> the online update. At decision time, sample weights from each arm's
> posterior independently, pick the argmax."*

> *"Privacy override — sensitive topic → cloud arm masked to zero
> probability. Structural, not policy."*

**[Beat.]** *"Russo 2018 for the tutorial; Chapelle and Li 2011 for the
empirical evaluation that this scheme beats epsilon-greedy and UCB on
asymmetric-cost problems like this one."*

**Recovery lines.**
- *If asked "why not UCB":* "UCB's confidence bounds are hard in 12-dim continuous context; Thompson handles it via sampling."
- *If asked "is this measurable yet":* "Offline replay shows sublinear regret vs oracle — honest caveat, synthetic rollouts."

---

## Slide 11 — Privacy by architecture (75 s)

**Target:** 75 s. **Cumulative:** 24:00.

Lead with the structural claim, then the mechanisms.

> *"Raw text is never stored. That is enforced at the storage layer — the
> diary schema has no message column. You cannot toggle it on. An attacker
> who compromises the policy file does not unlock raw text, because there
> is no raw text."*

**[Beat.]**

> *"Embeddings are Fernet-encrypted at rest. Ten PII regex patterns strip
> email, phone, address, dates, cards before any cloud call. A sensitive-topic
> classifier — health, mental, financial, credentials — forces local
> processing via the router override."*

> *"Honest caveat: Fernet is software-only. Production moves the key into
> TrustZone on Kirin. I name this as a placeholder, not a claim."*

**[Beat.]** *"The Edinburgh Joint Lab's March 2026 session on
personalisation from sparse signals made the same move — treat the
signal itself as the privacy-sensitive asset, not just the payload.
That's the design axis this slide owns."*

**Recovery lines.**
- *If asked "can you reconstruct text from embeddings":* "No, not under this encoder — it's trained on dynamics features, not content. Still lossy-not-zero; slide 13 is honest about it."
- *If asked "why Fernet not AES-GCM":* "Fernet wraps AES-128-CBC + HMAC; same guarantee, better default. TrustZone supersedes both in production."

---

## Slide 12 — Fits the devices — extrapolated, honestly (75 s)

**Target:** 75 s. **Cumulative:** 25:15.

This slide is a table and a caveat. The table is the headline; the
caveat is the signal that you know the difference between measurement
and extrapolation.

> *"Full pipeline is around 7 megabytes INT8 — TCN under one megabyte,
> SLM around six. Fits every Kirin device I've listed. On this laptop,
> measured end-to-end P50 is 150 to 220 milliseconds. On Smart Hanhan's
> 64 megabyte class, the encoder alone fits — around 30 milliseconds —
> and the SLM would be offloaded to a paired phone, which is exactly
> the AI Glasses pattern."*

**[Beat.]**

> *"The Kirin numbers in the right column are extrapolated from quantised
> model sizes and published NPU throughput — they are not measured. I
> want to be explicit about that. Real profiling needs the hardware,
> which I did not have."*

> *"Conversion target is MindSpore Lite — PyTorch to ONNX to MindSpore Lite,
> re-run INT8 calibration inside MindSpore because its quantisation ops
> are not always bit-identical to PyTorch's. Validate with under one
> percent generation divergence."*

**[Beat.]** *"That's the edge story — honest numbers, clear path, no
pretence that I've run this on Kirin."*

**Recovery lines.**
- *If asked "where's the measured Kirin data?":* "Absent. Extrapolation is the honest alternative; I'd run real profiling day-one in the lab."
- *If a table cell looks wrong:* "Let me pull the source datasheet after — the important bullet is the extrapolation caveat, which stands either way."

---

## Slide 13 — What this prototype is *not* — the credibility slide (120 s)

**Target:** 120 s. **Cumulative:** 27:15.

Slow down. This slide is the signal of maturity — the single most
important moment outside the demo.

> *"Honesty slide. Six things this prototype is not."*

**[Beat.]**

> *"Not a shipped product — 17 days on a laptop. Not trained on real user
> data — synthetic archetypes from Epp, Vizer, Zimmermann. Not a strong
> SLM — eight million parameters is not going to compete with Claude on
> language; the router knows that."*

**[Beat — slow.]**

> *"Not universal accessibility detection. Keystroke-only misses
> screen-reader users, voice-control users, gaze users. Adaptation must
> stay opt-out capable. The worst failure mode is adapting someone out of
> the experience they actually want. The accessibility statement in
> `docs/responsible_ai/` is explicit: detection is one signal among
> many, it complements rather than replaces explicit settings, and it
> fades with recovery rather than pinning a label on the person."*

**[Long beat — eye contact.]**

> *"Not zero-information embeddings — lossy, abstract, still identity-signalling.
> Not multi-modal — keystroke only, though the TCN itself is modality-agnostic.
> That's the extension path."*

**Recovery lines.**
- *If he challenges the accessibility claim:* "You're right — I'd want to run it past a qualified accessibility lead. That's in scope before any user-facing trial."
- *If "why admit so much":* "Because I'd rather name the limits than have you find them later. And because your BS radar is better than my salesmanship."

---

## Slide 14 — Where this goes next (90 s)

**Target:** 90 s. **Cumulative:** 28:45.

Close the technical arc with the future.

> *"L1 through L5 intelligence — the Huawei–Tsinghua IAIR framework. This
> prototype lives at L2 to L3 — single-device, proactive. L4 is
> device-to-device handover — phone to glasses to Smart Hanhan over the
> HarmonyOS distributed databus. L5 is autonomous orchestration."*

**[Beat.]**

> *"The long-term profile federates via MindSpore Federated — embeddings
> averaged across devices, never leaving any of them. Multi-modal extension
> is a drop-in — the TCN is modality-agnostic, so keystroke becomes touch,
> gaze, voice pace, accelerometer, with the same architecture."*

> *"Next-quarter scope in my head: keystroke-biometric user identification
> for multi-user devices, per-subgroup fairness evaluation, an interpretability
> panel that shows which features drove this adaptation, and an ablation
> toggle that lets a designer watch the system with and without the encoder."*

**Recovery lines.**
- *If asked "how long to L4":* "Engineering, not research — the hard part is the distributed data management integration, which is HarmonyOS's job already."
- *If asked "why federated and not central":* "Central re-introduces the privacy problem I spent slide twelve removing."

---

## Slide 15 — Close (60 s)

**Target:** 60 s. **Cumulative:** 29:45.

Land slowly. Deliver the closing line verbatim, without preamble, and
then stop talking. Silence is the performance.

> *"So that's the system. Let me leave you with one line — because it's
> the honest version of everything on every slide."*

**[Long beat.]**

> *"I build intelligent systems that adapt to people.
> I'd like to do that in your lab."*

**[Beat — full three seconds.]**

> *"Thank you. Happy to take questions — and when we get to the candidate
> questions at the end, I have three prepared."*

Sit or stand still. Do not fill the silence. They will speak next.

**Recovery lines.**
- *If the laptop dies mid-close:* keep the line verbatim from memory; the printed deck is in the bag.
- *If the closing line comes out flat:* no recovery, do not repeat it — move to the thank-you.
