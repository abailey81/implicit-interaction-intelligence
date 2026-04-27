# Presentation Script — verbatim, slide-by-slide

> **Use this as a tutor's speech, not a teleprompter.** Read it through 3
> times tonight; deliver it tomorrow in your own voice. The phrases in
> *italic* are stage directions: pauses, clicks, eye contact. The bold
> phrases are the **lines that should land verbatim** — the rest can flex
> with the room.
>
> **Total speaking time: ~22 minutes.** Five-to-eight minutes of slack for
> mid-presentation questions, transitions, and the live demo. Stay
> *under* the 30-minute clock; never go over.
>
> **Print this script.** Have it on the table. Glance, don't read. The
> interviewers will respect a confident speaker over a perfect one.

---

## Pre-flight (before they start the clock)

When you arrive in MR1 and they introduce themselves, **the first words
out of your mouth**:

> *(brief smile, light handshake or nod, sit down)*
>
> "Thanks very much for having me, Matthew. Vicky."
>
> *(short pause; let them respond)*
>
> "Quick framing before I start, if that's helpful: the 30 minutes are
> organised in four short acts — the cascade architecture, the two
> from-scratch and fine-tune halves, the edge deployment story, and the
> HCI argument. I'll aim to land at minute 22 to leave room. Happy to
> take questions as we go if anything jumps out — otherwise we save them
> for the dedicated Q&A block."
>
> *(pause; look at Matthew)*
>
> "Sound good?"

*Wait for the nod. Then click to slide 1.*

---

## Slide 1 — Cover (0:00 → 0:30)

*The slide shows: "I³" / "Implicit Interaction Intelligence" / your
name + date.*

> *(do not read the slide)*
>
> "I'll set the scene with the problem first, then the architecture,
> then the artefacts."
>
> *(click to slide 2)*

**Time check:** end at 0:30.

---

## Slide 2 — The thesis (0:30 → 1:30)

*The slide shows: "What if the assistant adapted to* how *you typed?"
plus a short paragraph.*

> "The thesis is in the title.
>
> Most chat assistants treat the prompt as the only signal of intent.
> They are good at language. They are dead to *how* you typed it.
>
> But typing rhythm, edit patterns, prosody, gaze — they already encode
> cognitive load, affect, intent, even identity. **You generate them
> whether you want to or not.**
>
> A useful HMI assistant should be able to read those signals — on
> device, in real time — and adapt its phrasing, length, vocabulary, and
> even its routing decisions accordingly.
>
> I³ is a working prototype of that thesis end-to-end."
>
> *(brief pause; click to slide 3)*

**Time check:** end at 1:30.

---

## Slide 3 — Project at a glance (1:30 → 3:00)

*The slide shows three metric cards in a row: 204 M / 5.36 × 10⁻⁶ /
162 KB.*

> "Three numbers anchor the rest of the deck.
>
> *(point at the first card)* **204 million parameters** in the
> from-scratch transformer. Mixture-of-experts, adaptive computation
> time, byte-level BPE — all hand-implemented, no HuggingFace in the
> generation path.
>
> *(point at the second card)* **5.36 × 10⁻⁶ validation loss** on a
> Qwen 1.7B + LoRA fine-tune for HMI command parsing. 100% action
> accuracy on a held-out 253-example test set.
>
> *(point at the third card)* **162 kilobytes** of INT8-quantised TCN
> encoder, running in the user's browser tab — that's the edge story.
>
> One repository. Three language-model arms. Stitched together by a
> multi-signal smart router."
>
> *(click to slide 4)*

**Time check:** end at 3:00.

---

## Slide 4 — Divider (3:00 → 3:05)

*The slide shows: "01 / The cascade".*

> *(silent click-through; visual rest)*

**Time check:** 3:05.

---

## Slide 5 — Three arms, one cascade (3:05 → 5:05)

*The slide shows three columns: SLM mint / Qwen amber / Gemini lavender.*

> "Three arms, three roles, three cost-quality trade-offs.
>
> *(left column)* **The from-scratch SLM.** 204 M custom decoder
> transformer, on-device weights. It's the *default* for every chat
> turn. It's also the differentiator — it's the part of the project
> that closes the JD's 'build SLMs from scratch' bullet directly.
>
> *(centre column)* **Qwen 1.7B with a LoRA adapter.** This is the
> deterministic JSON intent parser — when the user says 'set a timer
> for 30 seconds', this arm emits exact structured output the
> actuator dispatcher can fire on. It fires only when a regex command
> gate matches.
>
> *(right column)* **Gemini 2.5 Flash, cloud.** OOD safety net. Fires
> only when the local arms can't ground the query, and only when the
> user opted in.
>
> The principle: **cheapest arm that can give a confident,
> schema-valid, on-topic answer wins.** Per-arm confidence is
> displayed on every reply."
>
> *(click to slide 6)*

**Time check:** end at 5:05.

---

## Slide 6 — `route_decision` ships on every reply (5:05 → 7:05)

*The slide shows a JSON dict with the routing decision.*

> "Every reply ships this dict — `route_decision`.
>
> *(point at the json)* **Arm.** Which one fired. **Model.** The exact
> model name. **Query class.** Which of five route classes the message
> hit. **Score.** The winning arm's confidence in [0, 1]. **Smart
> scores.** Per-class scores so the user can see *why* this arm won.
>
> The user sees this as a chip below the reply. Hover the chip and
> you see the routing math.
>
> Why bother surfacing it? Lee & See's 2004 paper on calibrated trust
> in automation. **Trust is calibrated when the user's confidence in
> the system matches the system's actual reliability.** Hiding the
> routing decision means the user can't calibrate. Surfacing it means
> they can.
>
> Validation: 22 out of 22 routing classifications match expectation
> on the precision smoke."
>
> *(click to slide 7)*

**Time check:** end at 7:05.

---

## Slide 7 — The 14-stage pipeline (7:05 → 8:35)

*The slide shows: a flow diagram of the pipeline stages.*

> "Zooming out — one turn end-to-end runs through fourteen stages.
>
> Group them in three: **perception** — intake, coref, encode through
> the TCN, build the adaptation vector. **Generation** — the smart
> router, command gate, Qwen LoRA, retrieval, the SLM forward pass,
> cloud fallback. **Post-processing** — tools, the adaptation
> rewrite, and the side-effect dispatcher that schedules the timer
> tasks.
>
> Every stage emits trace events. The Flow tab in the UI animates
> this on every reply with **real timings**, not synthetic
> animation."
>
> *(brief pause)*
>
> "**That's the cascade as a runtime. Now to the two halves of how
> each arm was actually built — the from-scratch piece and the
> fine-tune piece.**"
>
> *(click to slide 8)*

**Time check:** end at 8:35.

---

## Slide 8 — Divider (8:35 → 8:40)

*The slide shows: "02 / From-scratch and fine-tuned, side by side".*

> *(silent click-through)*

**Time check:** 8:40.

---

## Slide 9 — AdaptiveTransformerV2, hand-written (8:40 → 11:40)

*The slide shows an architecture table + a result table.*

> "The from-scratch SLM. **AdaptiveTransformerV2**.
>
> 12 layers, 12 heads, model dimension 768. Mixture-of-experts FFN
> with two experts and top-1 routing. ACT halting — adaptive compute,
> simple turns halt early at around layer 7. Per-layer cross-attention
> conditioning onto an 8-axis adaptation vector and a 64-dimensional
> TCN user-state embedding.
>
> **The conditioning is the architectural move.** Most personalisation
> stacks use prompt-prefix conditioning — 'you are a helpful
> assistant talking to a designer who prefers short answers'. That's
> brittle: the model can ignore it. Cross-attention puts the
> user-state vector in a separate tensor that every layer at every
> token attends to. **The model can no more ignore it than its own
> input.**
>
> Tokeniser: byte-level BPE, 32 000 vocab, hand-rolled — about 460
> lines of Python. **Zero HuggingFace dependencies in the generation
> path.**
>
> *(point at the result table)*
>
> Best `eval_loss` 4.987 at step 18 000. **Headline perplexity is
> approximately 147** — that's training-time, response-only,
> same-distribution holdout. The number you'd compare against
> published small-LM benchmarks.
>
> There's a second perplexity number — 1 725 — from a stress-test
> evaluation on 500 pairs from the full corpus, scoring all tokens.
> Both real. The 12× gap is distribution shift plus all-token loss,
> not overfitting. **The architecture is data-bound at this scale**,
> not epoch-bound. Open problem number two is the full-corpus
> retrain."
>
> *(click to slide 10)*

**Time check:** end at 11:40.

---

## Slide 10 — Qwen LoRA recipe + result (11:40 → 14:10)

*The slide shows a recipe table + a result table side by side.*

> "The fine-tune half closes the JD's 'adapt or fine-tune pre-trained
> models' bullet.
>
> Base model: **Qwen 1.7 billion** — the latest open-weight Qwen the
> transformers library version on the laptop recognised. Fine-tuned
> with **DoRA** — weight-decomposed LoRA — at rank 16, alpha 32. Plus
> **NEFTune** with α=5 for noise regularisation. Plus **8-bit AdamW**
> from bitsandbytes to fit the optimiser state on a 6 GB laptop GPU.
> Cosine warm-restart schedule.
>
> Effective batch size 8 — batch 2 with gradient accumulation 4.
> Three epochs, 1 704 optimiser steps total. 2.7 hours wall time.
>
> *(point at the result column)*
>
> **Best validation loss 5.36 × 10⁻⁶.** Held-out test set of 253
> examples: **100% action accuracy, 100% full-match, macro F1 of
> 1.000**.
>
> The val loss is unusually low because the task is highly structured
> — small action vocabulary, well-formed JSON. The number to *quote*
> is the test accuracy, not the val loss.
>
> *(brief pause)*
>
> "**Both halves close the JD's both/and bullet — building from
> scratch *and* adapting pre-trained models. The third pillar is
> edge.**"
>
> *(click to slide 11)*

**Time check:** end at 14:10.

---

## Slide 11 — Divider (14:10 → 14:15)

*The slide shows: "03 / The edge".*

> *(silent click-through)*

**Time check:** 14:15.

---

## Slide 12 — 162 KB encoder (14:15 → 17:15) **+ DEMO TIME**

*The slide shows: 162 KB hero metric + footer with provenance.*

> "**One hundred and sixty-two kilobytes.**
>
> *(pause; let it land)*
>
> That's the INT8-quantised TCN encoder, running in the user's
> browser tab via ONNX Runtime Web.
>
> *(point at the footer, left to right)*
>
> 441.4 kilobytes FP32 → 162.2 kilobytes INT8. Sixty-three percent
> size reduction. Parity with FP32 on random inputs is a mean absolute
> error of 0.00055 — for context, that's far below the noise floor of
> the original keystroke features.
>
> Latency on a CPU laptop: 460 microseconds p50. 2 176 encodes per
> second. The Kirin A2 watch's encoder budget is 2 megabytes; we are
> twelve and a half times under that."

### IF demoing live (recommended)

> *(stand up; switch projection input)*
>
> "Quick switch — I'd like to show this live, two minutes, then we'll
> come back to the slides."
>
> *(plug in laptop; switch projection input; pull up
> http://127.0.0.1:8000)*
>
> "I'm going to flip the **Edge inference** toggle in the State tab,
> then send a chat turn. Watch the Network panel in DevTools."
>
> *(toggle edge inference ON; switch back to Chat tab; type a message;
> point at DevTools)*
>
> "**No `/api/encode` request fires.** The 32-dimensional feature
> vector hit the encoder client-side. Keystrokes never left this
> page.
>
> *Privacy by architecture, not by policy.* There's no policy to
> violate if the network call doesn't happen.
>
> *(quick demo of the actuator if time allows: type 'set timer for
> 30 seconds'; show the routing chip → Qwen-active; describe what
> will happen)*
>
> "And while we wait for that timer to fire, back to the slides."
>
> *(switch projection input back to deck)*

### IF NOT demoing live (fallback)

> "I'll skip the live demo for time but the toggle is in the State
> tab, the file is `web/models/encoder_int8.onnx`, and the
> reproduction recipe is in `reports/edge_profile_2026-04-28.md`.
> Happy to walk through it on my laptop after the call."

### Either way, finish with the honest gap

> "**Honest framing.** The encoder ships to the edge today. The 204 M
> SLM does not yet — that's open problem number one, blocked on a
> Kirin dev kit. ONNX export plumbing exists; the field deployment is
> the next step. I'd rather scope the claim accurately than oversell.
>
> **Why does edge matter at all? That's the HCI argument.**"
>
> *(click to slide 13)*

**Time check:** end at 17:15.

---

## Slide 13 — Divider (17:15 → 17:20)

*The slide shows: "04 / The HCI argument".*

> *(silent click-through)*

**Time check:** 17:20.

---

## Slide 14 — Implicit > explicit (17:20 → 19:20)

*The slide shows: HCI argument + three references.*

> "Why on-device, why implicit, why now.
>
> Three HCI references ground the design.
>
> *(reference one)* **Strayer and Cooper, 2017**, in Human Factors,
> measured a 35% drop in reaction time during in-vehicle infotainment
> interaction. Drivers don't have spare cognitive bandwidth for
> self-reflective preference elicitation.
>
> *(reference two)* **Wobbrock et al., 2011**, ability-based design.
> Users with motor or cognitive impairments — or any user
> interacting on a 1.4-inch wrist screen — cannot reliably tap fine-
> grained sliders. Explicit preference declaration fails as an
> interaction pattern in those contexts.
>
> *(reference three)* **Lee and See, 2004**, calibrated trust in
> automation. Trust is calibrated when reasoning is visible to the
> user. The routing chip on every reply is the operationalisation of
> that principle.
>
> Reading state from typing rhythm costs the user **zero additional
> work**. The system infers context from the interaction itself.
>
> *(point at the callout)*
>
> "**Honest gap.** No real user study yet. Validating the eight
> adaptation axes against self-reports is open problem number five —
> 3 weeks, IRB-blocked. I'd rather flag the gap than claim
> validation I don't have."
>
> *(click to slide 15)*

**Time check:** end at 19:20.

---

## Slide 15 — Open problems + close (19:20 → 21:20)

*The slide shows the inverted off-white "punch list" — six open
problems with effort estimates.*

> "I built I³ solo. I could pretend everything is finished. Instead
> this is the punch list I'd hand a teammate on day one.
>
> *(point through the rows)*
>
> One. Kirin watch deployment of the encoder. One week, blocked on a
> dev kit.
>
> Two. Full-corpus SLM v2 retrain. Thirty hours of GPU time, target
> headline perplexity below 80.
>
> Three. The chip A/B harness. Two weeks, IRB-blocked.
>
> Four. Multilingual cascade. Three days.
>
> Five. The user-state validation study. Three weeks.
>
> Six. Replace the warm-restart notes with a real run. Four hours
> after item two.
>
> Each one has an acceptance criterion, a rough effort, and a
> blocker.
>
> *(brief pause; sit back slightly; deliver the closing line slowly)*
>
> "**Solo project. Honest list. This is how I'd work in HMI Lab —
> scope tight, constraints explicit, validation criteria
> pre-registered.**
>
> *(pause; look at Matthew, then Vicky)*
>
> "Thanks for the time. Happy to take questions."
>
> *(click to slide 16; sit back)*

**Time check:** end at 21:20. You have ~9 minutes of slack already
spent on transitions and the demo. Total elapsed should land around
22-25 minutes.

---

## Slide 16 — Closing (held during Q&A)

*The slide shows: "Questions?" + repo URL.*

*(do not speak. Wait for the first question. The Tech Q&A block runs
for 10 minutes from this point.)*

---

## Recovery scripts (use only if needed)

### If the projector won't accept your laptop input

> "No problem — let me skip the live demo and walk you through the
> screenshots in the deck. The reproduction recipe is in
> `reports/edge_profile_2026-04-28.md` if you'd like to verify
> afterwards."

*(continue from slide 12 footer; describe the demo verbally)*

### If the demo fails mid-flow (server crashes, browser hangs)

> "Let me park that — I'll switch back to the slides and come back to
> this if there's time."

*(switch projection back to slides; continue from where you left off;
do NOT troubleshoot the demo in front of them)*

### If you blank on a number

> "Let me check — I have it pinned in the verifier."

*(open laptop; run `python scripts/verify_numbers.py`; quote the
number from the output)*

This is acceptable, even good — it shows your numbers are
re-derivable from artefacts on disk. Don't fake a number you can't
remember.

### If you go over time

> "I'll wrap up — the open-problems list is on the slide; happy to
> take questions."

*(skip slide 14 if needed; jump straight to slide 15 if at minute 27;
land on the closing line and stop)*

### If they interrupt with a question mid-presentation

> "Good question — let me address that briefly, and I'll come back to
> the cascade thread."

*(answer in 60 seconds max; then say "now back to where I was on the
cascade"; click to the next slide)*

Treat interruptions as a gift — they signal engagement. Answer
crisply, then *return to your thread*. Don't let an interruption
derail the presentation arc.

---

## Final reminders before tomorrow

1. **Eat breakfast.** Eggs + toast. Hydrate. One cup of coffee max.
2. **Charge everything tonight** — laptop, phone, power bank.
3. **Test the live demo end-to-end** — Identity Lock reset, Edge
   inference toggle, timer firing. If anything fails, plan to fall
   back to Option C (verbal walkthrough with screenshots).
4. **Print this script + the rapid Q&A** (`INTERVIEW_QA_RAPID.md`).
   Have them on the table during the interview, even if you don't
   look at them.
5. **Run `python scripts/verify_numbers.py`** in the morning. Confirm
   22/22 PASS. If anything fails, *fix it before you leave the house*.
6. **Lights out by 23:00 tonight.**

You've prepared. Show up on time. Tell them what you built.

---

*Last updated: 2026-04-28 for the 29 April 2026 interview at
Huawei R&D UK, MR1, Gridiron Building.*
