# I³ Live Demo Script

> A detailed 4-phase walkthrough for the Huawei HMI Lab interview demo.
> Target runtime: **~8 minutes** of live interaction, framed by ~3 minutes
> of introduction and ~2 minutes of closing Q&A scaffolding. **Total: ~13
> minutes** of demo time within a 30–45 minute presentation.

---

## Table of Contents

1. [Before the Demo: Pre-Flight Checklist](#before-the-demo-pre-flight-checklist)
2. [The Narrative Arc](#the-narrative-arc)
3. [Phase 1 — Cold Start (2 min)](#phase-1-cold-start-2-min)
4. [Phase 2 — Energetic Interaction (2 min)](#phase-2-energetic-interaction-2-min)
5. [Phase 3 — Fatigue Simulation (2 min)](#phase-3-fatigue-simulation-2-min)
6. [Phase 4 — Accessibility Adaptation (2 min)](#phase-4-accessibility-adaptation-2-min)
7. [Post-Demo: What to Point At](#post-demo-what-to-point-at)
8. [Recovery Playbook](#recovery-playbook)

---

## Before the Demo: Pre-Flight Checklist

Run through this list **15 minutes before** the demo. Total pre-flight
time: ~5 minutes.

- [ ] **Power and network**
  - [ ] Laptop plugged in (not running on battery)
  - [ ] Do Not Disturb enabled (macOS: `⌥` + click notification icon)
  - [ ] Wi-Fi connected, network speed test ≥ 50 Mbps down
  - [ ] A mobile hotspot is available as a fallback
- [ ] **Services**
  - [ ] FastAPI server running: `make demo` → "Application startup complete"
  - [ ] Seed data loaded (the home screen shows 3 pre-built profiles)
  - [ ] `ANTHROPIC_API_KEY` exported in the shell (verify with `echo ${ANTHROPIC_API_KEY:0:8}`)
  - [ ] Fernet encryption key exported (`echo ${I3_ENCRYPTION_KEY:0:8}`)
- [ ] **Browser**
  - [ ] Chrome or Safari open at `http://localhost:8000`
  - [ ] Zoom set to 125% (so panellists can read from across the room)
  - [ ] Browser dev-tools **closed** (distracting; opens automatically if you press F12)
  - [ ] Only one tab in the window (no clutter)
- [ ] **Dashboard state**
  - [ ] Left panel: chat box empty
  - [ ] Right panel: dashboard gauges all neutral (cognitive load ~0.3, warmth ~0.5)
  - [ ] Bottom panel: empty state embedding visualisation (neutral centre)
- [ ] **Fallback**
  - [ ] A screen recording of the full demo is queued in QuickTime
  - [ ] A printed fallback script is on the desk
  - [ ] You have the 4 phase scripts **memorised** — do not read from a screen

---

## The Narrative Arc

The demo tells a **single coherent story**: over the course of four
phases, the same user transitions from fresh and energetic, through
fatigue, to an accessibility-sensitive state — and the system adapts,
without ever being told, at every step.

The panellists should come away with three impressions:

1. **"It noticed."** — The system sees subtle signals the user never
   explicitly mentions.
2. **"It reacted."** — The response changes in real time, visibly,
   across multiple dimensions.
3. **"It's the same model throughout."** — Nothing is scripted; the user
   state drives everything.

The dashboard on the right panel should be **the panellists' visual
anchor**. Periodically draw their eye back to it: "notice the cognitive
load gauge climbing as I type".

---

## Phase 1 — Cold Start (2 min)

**Objective:** Establish a neutral baseline. Show that the system starts
from zero assumptions and begins learning immediately.

**Dashboard state at start:** Every gauge neutral. State visualisation
a small dot at the centre. Diary panel empty.

### Opening Line (to panellists)

> "Before I begin, let me point out the dashboard on the right. The four
> gauges show the system's internal model of my cognitive load, my style,
> my emotional tone, and whether it has detected any accessibility needs.
> Right now they're all neutral because the system has never seen me
> before. Watch what happens as we interact."

### The Interaction

Type each message at a **neutral, relaxed pace** (~60–70 WPM). Keep your
posture upright, hands relaxed.

**Message 1** — opening, neutral-polite:

> Hi, I'm interested in learning about temporal convolutional networks.
> Can you give me an overview?

Let the response render fully. Expected behaviour:

- **Route:** most likely `local` (Thompson sample with flat prior picks
  local by default early on)
- **Latency badge:** ~150ms
- **Dashboard:** cognitive load still ~0.3, formality begins to tick
  upward (the system notices you used a full sentence, asked politely)
- **State viz:** the dot moves slightly upward and to the right

**What to say** (while the response renders):

> "The route badge there says `local` — that's the Thompson sampling
> bandit deciding the query is simple enough for the on-device SLM.
> It's faster, it's private, and the system learns it can be trusted
> for queries like this."

**Message 2** — a follow-up that probes content:

> What's the difference between dilated convolutions and standard
> convolutions, and why would you use dilated ones in a TCN?

Expected behaviour:

- **Route:** possibly `cloud` (the complexity estimator rates this
  higher; the bandit might still explore `local`)
- **Latency badge:** ~600–800ms if cloud, ~200ms if local
- **Dashboard:** formality continues to rise; the state viz dot drifts
  further from the origin

**What to say:**

> "This query is more complex — the router picked cloud this time
> because the complexity estimator gave it a higher score. Notice the
> latency badge — cloud responses are slower, which is why we want
> the bandit to use local whenever it can."

### Phase 1 Wrap

> "Two messages in, and the system has already begun building a model
> of how I type. You can see the dashboard isn't neutral anymore.
> Let's see what happens when I ramp up the energy."

**Expected final dashboard state:**
- Cognitive load: ~0.25 (low)
- Formality: ~0.60 (mid-high)
- Warmth: ~0.50 (neutral)
- Accessibility: ~0.05 (none detected)

---

## Phase 2 — Energetic Interaction (2 min)

**Objective:** Demonstrate **style mirroring**. When the user gets
energetic, the system matches the energy.

**Persona shift:** You are now a fast-typing, enthusiastic user. Type
briskly (~100+ WPM), use exclamation marks, be casual.

### Opening Line

> "Now let me switch personas. Watch the style gauges."

### The Interaction

**Message 3** — energetic, casual, short:

> Ok this is super cool!! Can you just give me the gist of how attention
> works?? Like the 30-second version!

Type this **fast**. The feature extractor should pick up:

- Elevated `typing_speed_cpm`
- High `burst_ratio`
- Elevated `exclamation_ratio` and `punctuation_density`
- Lower `formality_score`
- Positive `sentiment_valence`

Expected behaviour:

- **Route:** `local` (short, casual, already learned the local arm works well)
- **Latency badge:** ~150ms
- **Dashboard:** formality **drops** visibly, warmth rises, verbosity
  target drops (short response expected)
- **Response:** casual, short, matches the energy

**What to say** (while response renders):

> "Notice the formality gauge dropping in real time. The style-mirror
> adapter picked up that I'm typing fast, using exclamations, being
> informal. The response adapts — it's shorter, more casual, more
> energetic to match me."

**Message 4** — continue the energy:

> Sweet! And is attention the same as self-attention? Or different??

Expected behaviour: fast local response, continued informal tone.

### Phase 2 Wrap

> "The response style has shifted. This isn't a system prompt — the
> adaptation is happening architecturally, through cross-attention
> conditioning in the SLM, at every token position. I'll come back to
> that after the demo."

**Expected final dashboard state:**
- Cognitive load: ~0.25 (still low)
- Formality: ~0.35 (**dropped**)
- Verbosity preference: ~0.30 (**dropped** — user wants short)
- Warmth: ~0.65 (**risen**)
- Accessibility: ~0.05

---

## Phase 3 — Fatigue Simulation (2 min)

**Objective:** This is **the key demo moment**. Show that the system
detects fatigue from **typing patterns alone** and responds with gentleness,
structure, and simpler language — without the user saying they are tired.

**Persona shift:** You are now tired. Slow down dramatically. Pause
between words. Make typos and fix them with visible backspaces. Sigh
audibly if the panellists are quiet enough to hear.

### Opening Line

> "Now I'm going to do something interesting. I'm not going to say I'm
> tired. I'm just going to type like I'm tired. Watch the dashboard."

### The Interaction

**Message 5** — deliberately slow, with corrections:

Type slowly (~30 WPM). Include at least 3 visible backspace/correction
events. Pause mid-sentence.

> I want to... understand how backprop works through time but I'm
> [backspace ×4 to fix typo] having trouble keeping it straight in my
> head

Expected feature extraction:

- `mean_iki_ms` elevated (slow typing)
- `std_iki_ms` elevated (uneven rhythm)
- `correction_rate` elevated (backspaces)
- `pause_ratio` elevated (mid-sentence pauses)
- `iki_zscore_vs_baseline` strongly positive
- `correction_zscore` strongly positive

Expected behaviour:

- **Dashboard:** cognitive load climbs to ~0.75–0.85, visibly
- **Route:** likely `cloud` (complexity is moderate, cognitive load is
  high, and the router has learned that high cognitive load benefits from
  a more capable response)
- **Response:** warm, structured with numbered bullet points, shorter
  sentences, simpler vocabulary, uses the phrase "let me break this down"
  or similar

**What to say** (while response renders — this is the *punch line* moment):

> "Look at the cognitive load gauge. It just climbed from 0.25 to
> roughly 0.8 **and I never told it I was struggling.** The only thing
> that changed is how I typed. The typing speed dropped, the rhythm
> became uneven, the correction rate shot up. The TCN encoder saw all
> of that, pushed it to the user model, and the cognitive-load adapter
> said: 'this person is under load; simplify the response'."

**Pause for effect.** Let the panellists absorb the dashboard state.

**Message 6** — continue the tired persona:

Type this one just as slowly.

> can you [pause] maybe give me a simple analogy

Expected behaviour: response is gentle, uses an analogy, short sentences.

**What to say:**

> "The response used an analogy because the style adapter noted I asked
> for one, but more importantly, the cognitive-load adapter is still
> active — look at the response: short sentences, no jargon, explicit
> structure. Compare that to the response style in phase 2 when I was
> energetic."

### Phase 3 Wrap

> "This is the single most important capability in the system. Everyone
> can personalise to explicit signals. I³ personalises to implicit ones —
> how you interact, not what you said."

**Expected final dashboard state:**
- Cognitive load: **~0.80** (high)
- Formality: ~0.45
- Verbosity: ~0.40 (shorter)
- Warmth: ~0.75 (**elevated to support the user**)
- Accessibility: ~0.15 (slightly elevated)

---

## Phase 4 — Accessibility Adaptation (2 min)

**Objective:** Show that the accessibility adapter — which is **distinct**
from cognitive load — kicks in when the typing pattern looks like motor
difficulty, not just mental fatigue.

**Persona shift:** You are now simulating motor difficulty. Type very
slowly. Every 2–3 characters, introduce a deliberate error and correct
it. The content should be **simple** (this is the key distinction from
phase 3). The mental content is easy; the *typing* is the problem.

### Opening Line

> "There's one more adapter I want to show. This one is distinct from
> cognitive load — it looks at whether the typing pattern itself
> suggests motor difficulty, regardless of whether the content is
> complex. I'm going to type a very simple question, very laboriously."

### The Interaction

**Message 7** — simple content, very effortful typing:

Type this at ~15 WPM with deliberate corrections every few characters.

> whta [backspace ×4] what time is it in tokyo

Expected feature extraction:

- `mean_iki_ms` very elevated
- `correction_rate` very elevated (many corrections)
- `pause_ratio` very elevated
- `complexity_zscore` **low** (simple content — this is the diagnostic)
- The accessibility adapter's specific pattern-match fires

Expected behaviour:

- **Dashboard:** **accessibility gauge rises to ~0.60–0.80**, while
  cognitive load stays moderate. This is the key visual.
- **Route:** `local` (simple question)
- **Response:** very short, direct answer, no jargon, no fluff, explicit
  formatting

**What to say** (while response renders):

> "The cognitive load gauge didn't spike this time — because the content
> is simple. But look at the accessibility gauge — it just jumped to
> roughly 0.7. The accessibility adapter looks for a specific pattern:
> elevated correction rate, elevated inter-key intervals, elevated
> pause ratio, **but without** a rise in linguistic complexity. That
> pattern is diagnostic of motor or cognitive difficulty — and the
> response becomes maximally short, maximally direct, with no idioms
> or jargon."

**Message 8** — push the point:

> thanks [pause] can you also tell [backspace ×10] actually where is
> the closest coffee shop

Expected behaviour: short direct response, no creative phrasing.

### Phase 4 Wrap

> "The system now believes it is talking to someone who needs help with
> typing. It is not making assumptions about who they are, or why —
> that would be presumptuous. It is simply adapting its response shape
> so that interaction is easier. And if the user's typing recovers
> in a future session, the adaptation will fade — it is a running EMA,
> not a permanent label."

**Expected final dashboard state:**
- Cognitive load: ~0.50
- Formality: ~0.50
- Verbosity: **~0.20** (very short)
- Warmth: ~0.70
- **Accessibility: ~0.70** (high)

---

## Post-Demo: What to Point At

After the live interaction, take **60 seconds** to show three things
that the demo itself cannot show without slowing down:

1. **The diary panel.** Open it (there should be a button on the web UI).
   Point out: "every row is a logged exchange. Notice there's no
   `message` column. The only columns are `embedding`, `topics`,
   `adaptation`, and the scalar metrics. Raw text is not in this table.
   It is not anywhere on disk. By construction."

2. **The dashboard state visualisation.** Point at the 2D canvas viz on
   the bottom of the dashboard panel. "This is a UMAP projection of the
   last 32 user-state embeddings. You can see them clustered in phases:
   the tight cluster top-left is phase 1 (neutral), the spread cluster
   on the right is phase 2 (energetic), and the lower cluster is phase
   3 and 4 (load and accessibility). The clusters are **well separated**,
   which is what tells us the TCN encoder learned a meaningful state
   space."

3. **The latency numbers.** Pull up the terminal showing the FastAPI
   access log. "Notice every local-path request returned in ~170ms
   total. That's on my laptop CPU, on a model I trained myself, with
   a model I built myself from zero PyTorch. On a Kirin 9000 NPU it
   would be ~50–80ms. This is **feasible edge AI** — not theoretically,
   actually feasible, today."

---

## Recovery Playbook

### If the web UI doesn't load

1. Check the server log in the terminal — look for a stack trace
2. If it's a config error: `make demo` again
3. If it's a deeper failure: switch to the pre-recorded screen capture
   and narrate over it
4. **Do not panic.** Say "the demo server had a hiccup — let me show
   you the recording I made this morning as a backup. The interaction
   I'll describe is real, not staged."

### If cloud routing fails (Anthropic API down)

1. The router will fall back to local for **every** query
2. Narrate: "the cloud arm is unavailable, so the router is keeping
   everything local — which is exactly the graceful degradation we
   designed for"
3. The demo still works; phase 3 just looks slightly less impressive

### If the dashboard gauges don't update

1. This usually means the WebSocket disconnected
2. Refresh the page once (the profile is persisted in SQLite, so state
   is not lost)
3. If it still fails: narrate "the WebSocket hit a transient disconnect,
   let me show the same interaction from my recording"

### If a panellist interrupts with a question mid-phase

1. **Finish the current message rendering** — do not abandon a phase
   mid-demo; the dashboard will look wrong if you do
2. Answer the question briefly
3. Say "let me come back to this after the phase — it's directly
   relevant to what happens next" if the question is big
4. Continue from where you were

### If the panellist says "this looks staged"

1. **Good — lean into it**
2. Say "it's scripted in the sense that I know what I'm going to type,
   but there's no pre-recorded response path. Let me demonstrate — I'll
   type something I haven't typed before"
3. Type any off-script message at the current phase's persona speed
4. The dashboard will respond — because the system is real

---

## Timing Summary

| Phase            | Live time | Gauge cues                              |
|:-----------------|----------:|:----------------------------------------|
| Pre-flight       | 5 min     | Performed beforehand, not during demo   |
| Introduction     | 3 min     | Slide deck, no interaction              |
| Phase 1 — Cold   | 2 min     | All gauges start neutral                |
| Phase 2 — Energy | 2 min     | Style shifts informal, warmth rises     |
| Phase 3 — Fatigue| 2 min     | **Cognitive load spike — the moment**   |
| Phase 4 — A11y   | 2 min     | **Accessibility gauge spike**           |
| Post-demo        | 1 min     | Diary panel, state viz, latency         |
| Q&A              | 5+ min    | Panel-driven                            |
| **Total**        | **~22 min**| **(leaves buffer for deep questions)** |

---

## Final Note

The demo is only impressive if the panellists **watch the dashboard**. 
The dashboard is more important than the chat responses. If you ever feel 
that the panellists are reading the chat instead of the gauges, stop 
typing and say:

> "Before I continue, let me draw your attention back to the dashboard
> on the right. This is where the interesting story is."

The chat is the stage; the dashboard is the play.

---

*Good luck. The demo has been practised. The system works. Trust the
pipeline.*
