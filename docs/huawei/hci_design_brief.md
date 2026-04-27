# I³ — HCI design brief

> Why an *implicit-first* HMI assistant is the right shape for an
> in-vehicle / wearable assistant, what the design choices were,
> and what an embedded UX team would do next.  One page so it
> survives a 30-min interview without skimming.

## The HMI problem

Conventional voice / chat assistants ask the user to **explicitly
declare their state**: "be more concise", "use less jargon", "I'm
in a hurry".  In an in-vehicle / wearable context this fails along
three axes:

1. **Cognitive load.**  When a driver is navigating a junction or a
   wearer is mid-workout, the spare cognitive bandwidth for
   self-reflective preference elicitation is approximately zero.
   Strayer & Cooper (2017, *Hum. Factors*) measured a 35 % drop in
   reaction time during in-car infotainment interaction; preference
   prompts directly compound this.
2. **Motor accessibility.**  Users with tremor / coordination
   impairments, or any wearer interacting on a 1.4-inch screen,
   cannot reliably tap fine-grained sliders or toggle 8 separate
   "make-it-formal" controls.  Wobbrock et al. (2011, *CHI*)
   characterise this as the *ability-based design* gap.
3. **Recurring overhead.**  Even when the user *can* declare
   preferences, doing it every turn is friction; doing it once and
   forgetting it produces stale state.  HCI-as-recall-task fails.

## The design choice

**Read state from how the user types, not what the user says.**

The 32-d feature vector our TCN encoder consumes — keystroke
inter-arrival times, edit count, composition cadence, linguistic
complexity — is **already produced by the act of typing**.  Cost to
the user: zero additional work.  This is what we call *implicit
interaction*: the system infers context from the interaction itself.

Eight adaptation axes downstream, each with a defensible HCI rationale:

| Axis | Signal it conditions on | HCI rationale |
|---|---|---|
| Cognitive load | inter-key interval entropy + edit count | High load → shorter, simpler responses (Sweller 1988) |
| Formality | content-word ratio, hedging frequency | Match register → reduces social friction (Pickering 2004) |
| Verbosity | message length history | Mirror the user's pacing |
| Emotionality | sentiment of recent turns | Soften / warm tone in response to distress (Picard 1997) |
| Directness | imperative vs interrogative ratio | Skip preamble when user is direct |
| Emotional tone | valence of token statistics | Affective tuning (Russell 2003 circumplex) |
| Accessibility | sustained motor-difficulty signals | Auto-engage simplified vocabulary + larger components |
| Biometric (typing rhythm) | Monrose & Rubin (1997) keystroke dynamics | Continuous authentication without re-prompting |

## Why the cascade — UX argument, not just engineering

The cascade isn't only a cost-routing trick.  It's an HCI commitment:

- **Local first** keeps keystrokes on-device.  Privacy-by-architecture
  beats privacy-by-policy: there's no policy to violate if the
  network call doesn't happen.
- **Cloud as last resort** means the latency / data trade-off is
  framed correctly: cloud only fires when the local arm genuinely
  can't be confident.  Users get on-device speed for the common
  case, cloud breadth for the rare case.
- **Per-arm routing chip on every reply** is a transparency move.
  *Calibrated trust* (Lee & See 2004) requires the user knows when
  the system was confident vs hedging.  Hover the chip, you see the
  routing math.  No black-box "the AI decided".

## Three references that shaped this design

1. **Strayer & Cooper (2017).** *"Cognitive distraction while
   multitasking in the automobile"*, Human Factors.  Quantifies the
   cognitive cost of in-car interaction and motivates the
   load-tightens-phrasing rule.
2. **Wobbrock et al. (2011).** *"Ability-based design"*, CHI.  The
   ability-modelling argument behind the accessibility axis.
3. **Lee & See (2004).** *"Trust in automation: designing for
   appropriate reliance"*, Human Factors.  The transparency argument
   behind the per-arm routing chip + "used: X" badge.

(Plus the keystroke-biometrics literature: Monrose & Rubin 1997,
Killourhy & Maxion 2009 — both cited inline in the Identity Lock
component's tooltip.)

## What I'd do with a UX team

This project is a solo portfolio piece.  Embedded in an HMI Lab UX
team, the next four moves are:

1. **Validate the 8 axes against actual user reports.**  Run a
   within-subjects study (n ≈ 20) where users self-report
   stress / fatigue / interest after each turn; correlate against
   the live adaptation vector.  Drop axes the model can't recover.
2. **A/B the cascade routing decisions.**  Half the cohort sees the
   per-arm chip with reasoning; the other half sees the text only.
   Measure trust-calibration (Madsen-Gregor scale) and task
   completion rate.
3. **Co-design the adaptation rules with accessibility users.**
   The current accessibility axis triggers on "sustained motor
   difficulty" — needs validation with users who have tremor /
   limited dexterity.  Wobbrock-style ability-modelling sessions.
4. **In-vehicle field study.**  Simulator first (CARLA + a dummy
   infotainment harness), then a small in-car pilot with the
   Cambridge / Edinburgh research team.  Measure cognitive
   distraction with NASA-TLX before and after the implicit-
   adaptation rules engage.

## What this brief is *not*

- Not a substitute for a real user study.  No users were recruited;
  no IRB; no inter-rater reliability on the adaptation rules.
- Not a deployed product.  The cascade runs, the encoder is INT8 +
  in-browser, but no Kirin watch shipped any of this yet.
- Not a peer-reviewed claim.  The references above ground the
  *design choices*, not the system's performance.

The honest framing: **I built an HMI prototype that takes the HCI
literature seriously enough to make defensible design choices, and
I know which validation work an HMI team would do next.**
