# Interview Day Briefing — 29 April 2026

> **Purpose.** Everything specific to your scheduled interview at Huawei R&D UK,
> separate from the general project prep in `COMPLETE_GUIDE.md`. Read this
> alongside the guide.
>
> **The big change from earlier prep:** this is **in-person**, not video.
> The demo logistics, the dress code, and the room dynamics are different.

---

## §0 — TODAY (28 April 2026): the slides email

### THIS IS THE TOP PRIORITY OF TODAY

**Deadline.** End of day, 28 April 2026.

**Recipient.** matthew.riches@huawei.com (Hiring Manager).

**What to send.** A PDF (or .pptx) export of the deck at
`docs/slides/i3_interview_deck.md`.

### How to render the deck to PDF

The deck is a Marp markdown file. Render it with the Marp CLI:

```bash
# install once (Node.js required)
npm install -g @marp-team/marp-cli

# render to PDF
marp docs/slides/i3_interview_deck.md --pdf --allow-local-files \
    -o docs/slides/i3_interview_deck.pdf

# or to PowerPoint, if you'd rather they edit in PPT
marp docs/slides/i3_interview_deck.md --pptx --allow-local-files \
    -o docs/slides/i3_interview_deck.pptx
```

If `marp` isn't installed and you're short on time, alternatives in
order of preference:

1. **Marp for VS Code** — install the "Marp for VS Code" extension,
   open the markdown file, click the Marp icon → Export to PDF.
2. **Online Marp** — paste the markdown into <https://web.marp.app/>
   and export to PDF.
3. **Manual fallback** — if Marp doesn't work, copy each slide's
   content into Google Slides / PowerPoint with the same colour
   palette (background `#0E1116`, text `#E6EDF3`, accent `#F2C25B`)
   and Inter font. The fallback is acceptable but takes longer.

### Cover email (paste this, edit the bracketed parts)

> **Subject:** Interview slides — Tamer Atesyakar (29 April, 12:00)
>
> Dear Matthew,
>
> Please find attached my presentation slides for tomorrow's interview at 12:00.
>
> The deck is 15 slides covering the project end-to-end (architecture, the
> from-scratch SLM and Qwen LoRA fine-tune, edge deployment, HCI rationale,
> and an honest open-problems list). I have planned the speaking time to fit
> the 30-minute presentation block.
>
> A small ask: I would like to drive a brief live demo of the system from my
> own laptop during the technical-presentation block — flipping the
> in-browser ONNX inference toggle and showing the routing chip on a real
> reply. If projecting from my laptop is straightforward at your end I will
> bring the right adapters; if you would prefer I stick to the slides only,
> please let me know and I'll plan accordingly.
>
> I look forward to meeting you and Vicky tomorrow.
>
> Best regards,
> Tamer Atesyakar
> [your phone number, in case Matthew needs to reach you]

### What if you can't get the slides rendered today?

Send the markdown anyway with an email like:

> Dear Matthew,
> Attaching my slides as a Marp markdown file (`.md`); a rendered PDF is
> following shortly. If the markdown is awkward to project I am happy to
> drive from my laptop on the day. Apologies for the rough edges; I wanted
> to honour the 28 April deadline.

Sending *something* by EOD beats sending a perfect PDF on the morning of.

### Confirm Matthew received

Send the email with read-receipt requested if your client supports it. If
you don't get an automatic reply, an SMS to recruitment / HR confirming
delivery is acceptable but not required.

---

## §1 — Interview format (the structured 60 minutes)

| Block | Length | What happens | Your job |
|---|---|---|---|
| Technical presentation | **30 min** | You drive. Slides + (ideally) live demo. | Memorise §2 of this doc, deliver clean. |
| Technical Q&A | **10 min** | They ask, you answer. | Use COMPLETE_GUIDE Part 9 sub-banks. |
| Behavioural questions | **10 min** | Soft-skill / fit questions. | See §4 below — STAR-format prep. |
| Role overview | **5 min** | They tell *you* about the role. | Listen, take notes, surface follow-ups. |
| Candidate Q&A | **5 min** | You ask them. | Have 5–7 questions ready (§5 below). |

**Every block is its own clock.** Do not eat into the next block by
overrunning. If you finish presentation in 28 minutes you can stop on time
and they'll respect it. If you go to minute 31 they'll cut you off and
you'll feel rattled.

---

## §2 — The 30-minute presentation playbook

The deck has 15 slides. The time budget per slide:

| Slide | Layout | Topic | Time |
|---|---|---|---|
| 1 | cover | Title | 30 s |
| 2 | default | The thesis ("Read *how* you type, not *what* you say") | 1 m |
| 3 | metrics-row | Three headline numbers + one-sentence pitch | 1 m 30 s |
| 4 | divider | "01 The cascade" | 5 s |
| 5 | split-3 | Three arms (SLM / Qwen / Gemini) | 2 m |
| 6 | code-snippet | `route_decision` shipped on every reply | 2 m |
| 7 | diagram-anchor | The 14-stage pipeline | 1 m 30 s |
| 8 | divider | "02 From-scratch and fine-tuned, side by side" | 5 s |
| 9 | data-table | AdaptiveTransformerV2 architecture + result | 3 m |
| 10 | split-2 | Qwen LoRA recipe + result | 2 m 30 s |
| 11 | divider | "03 The edge" | 5 s |
| 12 | hero-metric | The 162 KB encoder + browser-inference story | 3 m + demo time |
| 13 | divider | "04 The HCI argument" | 5 s |
| 14 | split-2 | Why implicit > explicit + three references | 2 m |
| 15 | inverted | Open-problems punch list + close | 2 m |

**Total: 28 min.** Two minutes of slack for transitions / questions
mid-presentation.

### §2.1 The opening 60 seconds (memorise verbatim)

After they introduce themselves, say:

> "Thanks for the time today. Quick framing before I start: the 30 minutes
> are organised in four acts — the cascade architecture, the two from-scratch
> and fine-tune halves, the edge deployment story, and the HCI argument. I'll
> aim to leave a couple of minutes at the end and would love to take questions
> as we go if anything jumps out — otherwise we save them for the dedicated
> Q&A block. Sound good?"

Then click to slide 2. Do **not** read slide 1 aloud.

### §2.2 Critical transitions

Three transitions matter most:

1. **End of slide 7 → slide 8.** "That's the cascade as a runtime. Now to
   the two halves of how each arm was actually built — the from-scratch
   piece and the fine-tune piece."
2. **End of slide 10 → slide 11.** "Both halves close the JD's both/and
   bullet about building from scratch *and* adapting pre-trained models.
   The third pillar is edge."
3. **End of slide 12 → slide 13.** "The encoder ships to the edge today; the
   SLM does not yet — that's open problem #1. But the deeper question is
   why edge matters at all in HMI."

These transitions are where most candidates get lost. Memorise them.

### §2.3 The live-demo decision tree

You have three options for the live demo. Decide before tomorrow:

#### Option A — Bring laptop, ask to switch projection (recommended)

If Matthew confirms (or you confirm at reception) that you can plug your
laptop in:

1. Pre-flight at home: server running, browser tabs lined up, Identity Lock
   reset.
2. At the moment of slide 12, say: "Quick switch — I'd like to show this
   live, two minutes, then we'll come back to the slides."
3. Plug in (USB-C → HDMI; bring a USB-C → DisplayPort adapter as backup).
4. Demo the State-tab toggle + DevTools (zero `/api/encode`) + the
   `set timer for 30 seconds` actuator.
5. Switch back to slides.

**Required hardware to bring:**
- Laptop, charged + power adapter
- USB-C to HDMI adapter
- USB-C to DisplayPort adapter (backup)
- Phone hotspot in case the room WiFi blocks `localhost:8000` access from
  the projector (it shouldn't, but…)

#### Option B — Pre-recorded screen capture embedded in the deck

If you don't want to risk the projector switch:

1. Today, record a 90-second screen capture of the demo (OBS or QuickTime).
2. Export as MP4.
3. Add the video to slide 12 (Marp supports `<video>` HTML).
4. The deck plays the demo automatically when slide 12 lands.

This is reliable but loses the *live* feel. The interviewers know the
difference.

#### Option C — Talk through the demo using static screenshots

If the room logistics make A and B impossible:

1. Capture three screenshots: (a) State tab with the toggle, (b) DevTools
   showing zero requests, (c) the gold-pulse banner firing.
2. Add to slide 12 in a row.
3. Verbally walk through the demo as you would have done live.

Acceptable but the weakest of the three. Use only if A and B fail.

### §2.4 The closing 60 seconds (memorise the shape)

Land on slide 15, the inverted "punch list" slide:

> "Six open problems, scoped, with effort estimates. This is the punch list
> I'd hand a teammate on day one — exactly the shape of how I'd work in HMI
> Lab. Solo project, honest about gaps, validation criteria pre-registered.
> Thanks for the time — happy to take questions."

Pause. Sit back slightly. They'll start the technical Q&A block.

---

## §3 — Technical Q&A block (10 min)

This maps directly to COMPLETE_GUIDE Part 9 sub-banks. The most likely
questions are pulled from §9.2 (Architecture) and §9.8 (Honest-gaps). Re-read
both tonight.

### §3.1 The five questions to over-prepare

If they ask any of these, you should answer in 60 seconds, not 30 and not
180:

1. *"Why three arms — isn't that just complexity?"* → §9.2 in COMPLETE_GUIDE.
2. *"Why is the SLM perplexity so high?"* → §9.3.
3. *"Walk me through the LoRA recipe."* → §9.4.
4. *"You said it's edge-deployable, but the SLM hasn't shipped to a watch."*
   → §9.8.
5. *"Wouldn't a single GPT-4 call beat your whole cascade?"* → §9.8.

### §3.2 The structural-answer template (recall from §12.5)

Every technical answer follows this shape:

```
1. One-sentence direct answer (the thesis).
2. Two or three supporting facts (the evidence).
3. The artefact (where to verify).
4. Honest caveat / scope (what you're NOT claiming).
```

If you blank, fall back to: *"Let me think about that for a moment."*
Then five seconds of silence. Then either answer or *"I want to give you
a precise answer rather than a sloppy one — can I come back to this?"*

---

## §4 — Behavioural questions block (10 min)

This is **10 dedicated minutes** — a full sub-interview. It's worth
preparing as carefully as the technical block.

### §4.1 The STAR format

For every behavioural question, answer in **STAR** structure:

- **S**ituation. Set the scene in one sentence.
- **T**ask. What was your responsibility?
- **A**ction. What did *you* (specifically) do?
- **R**esult. What was the outcome? (Numbers, lessons, follow-ups.)

Most candidates over-weight Situation and skim Action. **Spend 60 % of
your answer on Action.** That's where the signal is.

### §4.2 The eight likely behavioural questions, with prepared answers

#### Q1. *"Tell me about a time you faced a difficult technical problem."*

Use the **iter 51 phase 6 warm-restart attempt** as the worked example.

> **S**: Building I³, the v2 SLM had landed at perplexity 147 on a 300 k
> training subset; I wanted to push it lower.
> **T**: Hypothesise + test whether more polish steps from a smaller LR peak
> would beat the baseline.
> **A**: Implemented `--resume` plumbing in the trainer to load
> `model_state_dict + optimizer_state + global_step`. Re-launched at lr=3e-5,
> warmup-ratio 0.001, +3 000 steps. Monitored eval-loss every 500 steps.
> **R**: At step 18 750 the eval landed at 5.10 / ppl 164.7 — *worse* than
> the 4.987 / 147 baseline. I halted the run, wrote the result up honestly
> in `reports/slm_v2_eval.md` with the conclusion that the architecture is
> data-bound, not epoch-bound. The path forward is the full-corpus retrain,
> which I scoped as open problem #2.
>
> Lesson: warm restarts don't help when the model has already converged on a
> small data slice. The right move was to admit the negative result and
> re-scope the work.

#### Q2. *"Tell me about a time you had to learn something new quickly."*

Use **the LoRA / DoRA / NEFTune fine-tuning recipe** for Qwen.

> **S**: I needed a deterministic JSON intent parser to close the JD's
> "fine-tune pretrained" bullet. I had no prior experience with `bitsandbytes`
> 8-bit AdamW or DoRA specifically.
> **T**: Read the original LoRA paper (Hu 2021), the DoRA paper (Liu 2024),
> and the NEFTune paper (Jain 2023). Pick a recipe that fits a 6 GB laptop
> GPU.
> **A**: Wrote `training/train_intent_lora.py` from scratch over 48 hours.
> Stacked DoRA + NEFTune α=5 + 8-bit AdamW + cosine warm restarts. Built
> a synthetic 5 050-example HMI command dataset.
> **R**: Best val_loss 5.36 × 10⁻⁶, 100 % action accuracy on a held-out
> 253-example test set. Total wall time 2.68 hours. The recipe is now
> reusable for adding new actions.
>
> Lesson: stacking known-good techniques is faster than re-deriving
> first-principles solutions for each one.

#### Q3. *"Tell me about a time you disagreed with someone."*

If you have a real example, use it. If not, frame around **a methodological
disagreement with a contributor or reviewer of your work** (or with an
earlier version of yourself, in iteration history).

Default fallback: *"During iter 49 I kept oscillating between the smart
router being deterministic vs learned. The honest disagreement was with
my earlier self — the deterministic version was easier to reason about
and faster to debug, but a learned router would generalise better. I
landed on the deterministic spec for shippable v1, with the learned
version flagged as a future work item. Resolving the disagreement meant
admitting both positions had merit and scoping the trade-off
explicitly."*

#### Q4. *"Tell me about a time something didn't go as planned."*

Use the **iPhone deployment attempt** that failed for laptop firewall /
network reasons.

> **S**: Trying to demo I³ on a phone (the user-test-sized device), I
> ran into Windows firewall + network issues that blocked
> `localhost:8000` from being reached over LAN.
> **T**: Either fix the firewall path or pivot.
> **A**: Spent ~30 minutes on firewall rules and network configs, hit a
> dead end. Pivoted: the encoder ONNX path was already running in-browser
> via WebGPU/WASM, so I leaned into that as the edge-deployment story
> rather than pursuing a phone runtime.
> **R**: The deck's edge slide is now anchored on the in-browser INT8
> encoder demo, which is *more* reliable than a phone demo would have
> been (no LAN dependency at all). The Kirin watch deployment is open
> problem #1 with a clear hardware blocker.
>
> Lesson: when a path closes, the right move is often a smaller / cleaner
> demo that's actually shippable, not a heroic recovery of the original
> path.

#### Q5. *"Tell me about a time you took initiative."*

Use **`scripts/verify_numbers.py`** — the audit script you wrote after
catching a perplexity misquote.

> **S**: A reviewer caught me misquoting the SLM perplexity as 1 725 when
> the headline number is 147 — different metrics, easy to confuse.
> **T**: Prevent that class of error.
> **A**: Wrote a re-runnable verifier that loads each artefact and asserts
> 22 specific claims against the on-disk values: model config, training
> step, eval loss, parameter count, encoder size, parity MAE, etc. Tied
> every recruiter-facing claim to one of those assertions. Added a
> mandatory re-run before any release.
> **R**: 22 / 22 PASS as of this morning. Any future drift fails the
> check immediately.
>
> Lesson: build the audit *before* you need it. A 2-hour script paid for
> itself the first time it caught drift.

#### Q6. *"Why this role at Huawei?"*

> Three reasons. First, the JD's both/and — building from scratch *and*
> fine-tuning pre-trained — is rare. Most labs do one or the other; I3 is
> built around exactly that combination. Second, HMI is where implicit-
> interaction signals matter most, because cognitive bandwidth is scarce
> in vehicles, on wearables, and behind AR glasses. Third, the Edinburgh
> Joint Lab has published on small-LLM personalisation, which is the
> direct extension of I3's TCN-encoder + adaptation-vector path. The
> forward roadmap maps each I3 component to a HarmonyOS / Kirin extension.

#### Q7. *"What's your biggest weakness?"*

Avoid the cliché "I'm a perfectionist". Use a **real, specific, working-
on-it weakness**:

> Honestly: scope. I'm a generalist by instinct — I3 covers SLM, BPE, TCN
> encoder, bandit, safety, UI, edge — and it'd be easy to dilute focus by
> accepting too many directions. I'm working on this by leaning more
> heavily on tech leads to scope ruthlessly, and by writing open-problems
> docs early so I can park future work without losing it.

#### Q8. *"Where do you see yourself in five years?"*

> Doing the kind of work that requires both the research depth I'm
> building now and the deployment scale Huawei has access to. I'm not
> planning a PhD next; I want to ship things to real users and validate
> the research questions I find interesting at scale. The HMI Lab is
> exactly that combination — the research questions on implicit
> interaction matter, and the deployment surface (HarmonyOS, Kirin,
> the wearable line) lets you validate them.

### §4.3 Behavioural anti-patterns (avoid)

- **Don't** use a school-project example for every question. They want
  professional or recent-substantial examples.
- **Don't** name people you disagreed with by name (or even by role) in
  a way that's identifiable. Disagreements are with *positions*, not
  individuals.
- **Don't** turn a behavioural answer into a technical lecture. The
  STAR is shorter than your technical answers, not longer.
- **Don't** lie about real failures. If you don't have a great Q4
  example, say "the closest I have is…" — interviewers respect that.

---

## §5 — Candidate Q&A block (5 min)

You have **5 dedicated minutes**. Plan **5–7 questions**; ask **2–4**.

### §5.1 Pre-prepared questions (in priority order)

Pick the ones that match the conversational flow.

#### About the work

1. *"What does a typical week look like for an HMI Lab intern?"*
2. *"What's the most interesting open research question the lab is
   working on right now?"*
3. *"How does the lab decide what gets shipped to a HarmonyOS product
   vs what stays research?"*
4. *"What does success look like for an intern by the end of summer?"*

#### About the team / collaboration

5. *"Who would I be working with most closely, and what's their
   focus?"*
6. *"How does the UK lab collaborate with the broader Huawei R&D
   footprint — Edinburgh, Shenzhen?"*

#### The closing question (save for last)

> **"Based on what we discussed today, is there anything about my
> background or experience you'd want me to clarify or expand on?"**

This is the strongest closing question because it (a) signals you can
take feedback in real time, (b) surfaces concerns *before* they become
reasons not to extend an offer, (c) is confident without being arrogant.

If they say "no, I think we covered everything well", that's a positive
signal. If they raise a concern, address it directly using the structural-
answer template.

### §5.2 Questions NOT to ask in this block

- Salary (HR conversation later).
- Vacation / working hours (same).
- Anything you could have learned from the website.
- "What do you like most about working here?" (too soft).

---

## §6 — Travel + arrival logistics

### §6.1 The route

**Address.** 5th Floor, Gridiron Building, 1 Pancras Square, London,
N1C 4AG.

**Nearest station.** **King's Cross St Pancras** (Piccadilly,
Northern, Victoria, Hammersmith & City, Circle, Metropolitan, plus
National Rail). Pancras Square is a 2-minute walk north of the
station — exit toward King's Boulevard / Granary Square.

**Plan to arrive: 11:50.** That's 10 minutes before 12:00. They
recommended 5 minutes early; budget 10 to absorb security /
registration.

**Buffer.** Add 30 minutes to your transit estimate. London delays
are routine. If you're in town the night before, do a dry-run walk
from station to building so you know the route.

### §6.2 Arriving at the building

1. **Ground floor.** Inform reception you are visiting **Huawei** on
   the 5th floor. They'll grant lift access.
2. **5th floor.** Check in at Huawei reception. Ask for **Matthew
   Riches** (Hiring Manager) **or Vicky** (Talent Acquisition) for
   your interview in **MR1**.
3. **Visitor registration.** Have your **photo ID** ready. They'll
   issue a visitor badge.

### §6.3 What to bring

Print this checklist and tick off the night before.

```
☐ Photo ID (passport or driving licence — passport preferred for
   security desks)
☐ Laptop, charged
☐ Laptop charger
☐ USB-C to HDMI adapter
☐ USB-C to DisplayPort adapter (backup)
☐ Phone (for hotspot if needed)
☐ Phone charger / power bank
☐ A printed copy of your CV (2 copies in case there are 2 interviewers)
☐ A small notebook + pen (taking notes during the role-overview block
   is a positive signal)
☐ Water bottle (small, ideally tinted so it isn't distracting)
☐ Tissues / cough drops if you've had any cold symptoms
☐ Light snack (cereal bar) for AFTER the interview, not before
☐ Breath mints (use 5 minutes before, not during)
☐ Your printed copy of this briefing's §2 time table (memory aid)
```

### §6.4 Dress code

**Smart business casual.** Specifically:

- **Shirt.** Ironed button-down, plain colour (white / light blue /
  charcoal). Long sleeves rolled at the cuff is fine.
- **Trousers.** Chinos or smart trousers. Not jeans. Not joggers.
- **Shoes.** Leather or leather-look. Polished. Not trainers.
- **Optional.** A blazer or smart jacket if the weather is mild
  enough; you can take it off if the room is warm.
- **No tie required** (this is a tech role, not banking) but
  acceptable if you'd be more comfortable.

**Do not wear:**

- T-shirts (even branded ones).
- Hoodies.
- Trainers (running shoes, sneakers).
- Loud patterns or saturated primary colours on the shirt.
- Anything new you haven't worn before — clothes need to be
  comfortable, not just look right.

**Personal grooming.** Clean shave or trimmed beard. Hair tidy.
Cologne / perfume *minimal* — close-quarter rooms penalise heavy
scent.

### §6.5 The morning of

```
07:30  Wake up. Light breakfast (eggs + toast; no heavy carbs).
       Avoid coffee if you're already wired; one cup if you usually
       have one. Hydrate.

08:30  Shower, dress, final grooming.

09:30  Final review: print + read this doc's §2 (the slide-by-slide
       playbook), §3.1 (the five Q&A questions to over-prepare),
       §4.2 (the eight behavioural answers), §5.1 (your candidate
       questions).

10:30  Pack the §6.3 checklist.

11:00  Leave home (if you live in London) / arrive at station early
       (if commuting).

11:35  Arrive at King's Cross station.

11:45  Arrive at Gridiron Building, 1 Pancras Square.

11:50  Ground-floor reception: ask for Huawei 5th floor.

11:55  5th-floor Huawei reception: ask for Matthew Riches or Vicky.

12:00  Interview begins.
```

### §6.6 If something goes wrong

- **Running late.** Call Vicky (TA) — get the number from the email
  thread before tomorrow morning. Better to call at 11:45 saying
  "I'm 15 minutes delayed" than to walk in at 12:08 having said
  nothing.
- **Wrong building.** Pancras Square has multiple buildings. Gridiron
  is the one with the angular criss-cross pattern on the facade
  (it's named for that).
- **Locked out of laptop.** This is why you've sent the slides ahead
  by email. They can project from their machine. Demo becomes
  Option C (verbal walkthrough with stills).
- **Demo fails.** Don't troubleshoot in front of them. Say "let me
  switch to the slides while I have a look" — switch back to the
  deck, finish the presentation, return to the demo only if you
  genuinely fix it inside 30 seconds.

---

## §7 — Meeting Matthew + Vicky

### Matthew Riches — Hiring Manager

He'll lead the technical questions. He's the person with the most
weight in the hire/no-hire decision.

**Posture toward him:** technical equal. Don't talk *up* to him as if
he's a professor; talk *across* as if he's a colleague who's deeper
in the domain.

**The single move that lands well:** when he asks a question, repeat
the *frame* of the question briefly before answering, to confirm
you're answering the right thing. *"You're asking about the perplexity
trade-off specifically — let me address that…"* This is a senior-
engineering communication style.

### Vicky — Talent Acquisition

She'll likely run the role-overview block and may attend the
behavioural block.

**Posture toward her:** warm and direct. TA people care about culture
fit, communication, and genuine interest in the company. The
behavioural questions in §4 above are her territory.

**The single move that lands well:** when she explains the role,
take notes (literally, with the notebook). It signals you take her
seriously and that you're processing the info. Reference her points
in your candidate Q&A if natural.

---

## §8 — The night before (28 April evening)

After you've sent the slides:

```
☐ Lay out tomorrow's clothes
☐ Pack the §6.3 checklist into a bag
☐ Print: this doc's §2 (slide playbook), §4.2 (behavioural answers),
   §5.1 (candidate questions)
☐ Test the laptop demo end-to-end one more time
☐ Run `python scripts/verify_numbers.py` — confirm 22/22 PASS
☐ Charge laptop, phone, power bank
☐ Set two alarms for 07:30
☐ Eat a normal dinner. No alcohol. No new foods (bad food = bad
   morning).
☐ Re-read COMPLETE_GUIDE Parts 1, 6, 7, 9.1, 9.8 (P0 sections).
☐ Lights out by 23:00
```

Don't try to learn anything new tonight. The night before is for
*consolidation*, not new material. If something feels like a gap,
write a one-line note about it for the morning — don't try to fix
it now.

---

## §9 — One last thing

You've built I³ over 100+ hours. The deck represents what you built.
The interview is a conversation about that work, not an exam on
unrelated material.

The interviewers want to find a colleague they'd enjoy working with
— that's why the conversation is happening. They're already inclined
toward you.

**Be the candidate who says the honest thing first.**

If the SLM perplexity is high, you say so before they ask. If the
encoder ships but the SLM doesn't, you scope the claim. If the user
study isn't done, you flag it as open problem #5.

Honesty builds trust faster than polish.

You've prepared. Sleep well. Show up on time. Tell them what you
built and what you'd build next.

Good luck.

---

*Last updated: 2026-04-28. Verified against
`scripts/verify_numbers.py` (22 / 22 PASS).*
