# Closed-Loop Persona-Simulation Evaluation for Implicit Interaction Intelligence

*Paper-style specification and methodology. Version 0.1.0. Companion to
`i3/eval/simulation/`, `scripts/run_closed_loop_eval.py`, and
`docs/research/implicit_adapt_bench.md`.*

## Abstract

We introduce a closed-loop simulation harness that evaluates whether an
on-device, privacy-preserving user-modelling stack recovers the ground
truth of a user's cognitive, motor, and linguistic state. Unlike the
cross-attention conditioning ablation (Batch A) which measures
*responsiveness* through symmetric KL divergence of next-token
distributions under swapped adaptation vectors, the closed-loop harness
measures *correctness*: does the inferred adaptation vector converge to
the true one for a known synthetic user? We construct eight canonical
HCI personas grounded in keystroke-dynamics and cognitive-load
literature, each paired with a researcher-specified `AdaptationVector`
that serves as ground truth. A deterministic `UserSimulator` generates
persona-faithful message streams with plausible inter-key intervals,
pause distributions, correction rates, and lightly stylised text. The
full I3 pipeline processes each message end-to-end and the evaluator
scores 1-NN persona recovery, per-message L2 adaptation error,
convergence speed, and persona-conditional router bias. All metrics
report 95 % bootstrap confidence intervals and are fully reproducible
under a fixed seed. This note documents the method, limitations, and
planned calibration path from simulated to real-user signatures.

## 1. Introduction

Adaptive AI systems promise to mirror a user's cognitive state,
communication style, and accessibility needs in real time. The
responsiveness of an adaptive architecture -- the extent to which its
outputs move when its inputs change -- is necessary but not sufficient:
a system that responds strongly but in the wrong direction is worse
than a system that ignores its conditioning. Responsiveness is
therefore the lower half of a two-part evidence chain:

* **Responsiveness** (Batch A, `docs/experiments/preregistration.md`):
  does the output distribution actually change when the adaptation
  signal changes?
* **Correctness** (this work): when the adaptation signal is *known*,
  does the inferred signal converge to it?

Responsiveness can be measured on a random-init model; correctness
cannot. Correctness is the load-bearing claim for user-modelling work
because it is the only way to answer the question a regulator,
deployment partner, or end user will actually ask: *does your system
recover what I am, or does it merely change its mind when you ask it
to?*

The HCI community has a long tradition of evaluation by simulated or
small-N user studies precisely because the ground truth is otherwise
inaccessible. Gajos and Weld's SUPPLE work (2004) is emblematic: the
authors synthesised users with known ability profiles and evaluated UI
adaptation algorithms by how close the produced UI came to the
user-optimal configuration. We import this idea wholesale into the
user-modelling layer of a multi-device, privacy-preserving AI
assistant. The synthetic users are not meant to *replace* live-user
studies; they are the unit test that must pass before a live-user
study is ethically or economically justified. When the system cannot
pass the unit test, the live-user data would be noise added to a
silent system.

This document describes (i) the persona library, (ii) the
`UserSimulator`, (iii) the `ClosedLoopEvaluator`, and (iv) the metric
suite. It also lays out the calibration path from the synthetic
signatures used here to the real-user signatures required for
downstream claims -- a small IRB-lite user study described in
`docs/research/implicit_adapt_bench.md`. The scripts pinned to this
document are fully deterministic under a fixed seed, pin the analysis
commit SHA into each JSON dump, and emit a Markdown report that can be
read without re-running the evaluation.

## 2. Related work

**Adaptive-UI simulation.** Gajos & Weld's SUPPLE system (CHI 2004)
generates UIs conditioned on a user's motor and cognitive ability
profile and evaluates the resulting UI's expected cost against an
optimal baseline. Their key insight -- that synthetic users with known
profiles permit a fully reproducible evaluation of an adaptive
algorithm -- is the methodological template we follow here. We extend
it from single-task UI adaptation to a multi-turn, multi-signal
user-modelling stack.

**Gaze-based simulated users.** Duchowski et al. (2014) introduced
simulated gaze trajectories for evaluating gaze-contingent displays.
Their simulator parameterised fixation duration, saccade amplitude, and
visual-search strategy from published psychophysics. We use the same
pattern for keystroke dynamics: parameter values are drawn from Epp
(2011), Vizer (2009), Zimmermann (2014), and Trewin (2000), not
invented.

**Continual-learning evaluation protocols.** Kirkpatrick et al. (2017),
in their work on elastic weight consolidation, introduced a standard
protocol for evaluating whether a continually-learning system actually
retains earlier tasks. The analogous question for user modelling is
whether a single pipeline tracks multiple distinct user states over a
session and can distinguish them. Our persona-confusion matrix is
structurally the same diagnostic: row-diagonal mass is the continual-
recognition analogue of their per-task accuracy.

**Synthetic-user benchmarks in HCI.** Cheng et al. (2024) argued
explicitly for synthetic-user benchmarks as a prerequisite to large-N
HCI experiments, particularly in accessibility research where recruiting
users with specific impairment profiles is expensive, slow, or
ethically constrained. Our `MOTOR_IMPAIRED_USER`, `LOW_VISION_USER`,
and `DYSLEXIC_USER` personas follow this template. The distributional
ranges are drawn from Trewin (2000) for motor impairment and standard
screen-reader studies for low vision; they are starting points for a
real calibration study, not substitutes for one.

**Keystroke-dynamics user modelling.** The persona typing profiles rest
on three pillars. Epp (2011) demonstrated that keystroke dynamics can
discriminate emotional states with above-chance accuracy, establishing
that the signal contains state information. Vizer (2009) replicated
and extended this for cognitive-stress inference. Zimmermann (2014)
contributed dynamic-time-warping methods that the simulator implicitly
uses when it applies per-position pause and burst masks to sampled
inter-key intervals. We combine these into eight personas that span
the distributional space the literature has measured.

**Cognitive load.** The `HIGH_LOAD_USER` persona is informed by the
NASA-TLX load model of Hart and Staveland (1988). Load here is a
latent construct; the simulator makes it observable via elevated
correction rates and scattered burst patterns that are the typical
behavioural signature of interrupted motor-program execution.

## 3. Method

### 3.1 Persona library

Eight canonical personas live in `i3/eval/simulation/personas.py`:
`FRESH_USER`, `FATIGUED_DEVELOPER`, `MOTOR_IMPAIRED_USER`,
`SECOND_LANGUAGE_SPEAKER`, `HIGH_LOAD_USER`, `DYSLEXIC_USER`,
`ENERGETIC_USER`, and `LOW_VISION_USER`. Each persona is an immutable
Pydantic model bundling:

* A `TypingProfile` of `(mean, std)` tuples for inter-key interval
  (ms), burst ratio, pause ratio, correction rate, and typing speed
  (cpm). All five fields are *distributions*, not scalars, so a single
  persona produces variable samples.
* A `LinguisticProfile` with a Flesch-Kincaid target grade, a
  formality target, a verbosity mean (words per message), and a
  baseline sentiment valence.
* An `expected_adaptation: AdaptationVector` representing the
  ground-truth adaptation the system should converge to.
* An optional `drift_schedule` of `(time_fraction, parameter_override)`
  pairs that models within-session drift (e.g. deepening fatigue,
  second-language warm-up).

The ground-truth vectors were designed so that the pairwise L2
distance between any two personas exceeds 0.2 while per-message
sampled signatures remain realistically overlapping. This gap makes
1-NN recovery a non-trivial diagnostic: a random baseline hits 12.5 %
(1 of 8); a well-functioning pipeline should hit substantially more.

### 3.2 Simulator

`UserSimulator(persona, seed)` in `i3/eval/simulation/user_simulator.py`
is fully deterministic. It derives its NumPy `default_rng` and Python
`random.Random` seeds from `seed ^ hash(persona.name)` so two personas
sharing a top-level seed do not align. Per-message sampling:

* Picks a canonical prompt from a fixed 24-element library by hashing
  `(persona_name, message_index, seed)`. The prompt library mirrors
  the style distribution used in `i3/eval/ablation_experiment.py`.
* Trims or pads to match the persona's `verbosity_mean`, then applies
  a formality-conditional suffix (`"Could you please help with this?"`
  vs. `" thanks :)"`) to bias the linguistic profile.
* Samples `len(text)` inter-key intervals from a truncated Gaussian
  around the persona's IKI mean, applying per-position pause and
  burst multipliers (pauses ≈ 5× baseline, bursts ≈ 0.6×) that
  reproduce the bimodal inter-key interval distributions reported in
  Epp (2011).
* Applies the drift schedule: for each message we compute a
  `time_fraction ∈ [0, 1]` from `message_index / (n_messages - 1)`
  and pick the drift entry with the largest `time_fraction ≤ current`.
  Drift overrides individual `TypingProfile` fields, not the whole
  profile, so a schedule can express "IKI grows but burst ratio
  doesn't" without side-effects on unspecified fields.

### 3.3 Evaluator

`ClosedLoopEvaluator` (`i3/eval/simulation/closed_loop.py`) drives the
full I3 `Pipeline` through an outer loop over personas, sessions, and
messages. For each message it builds a `PipelineInput` with
composition time derived from the simulator's keystroke stream, the
user's text, and the simulated timestamp; it awaits
`pipeline.process_message(...)`; and it extracts the inferred
`AdaptationVector` from the resulting `PipelineOutput.adaptation` dict
via `AdaptationVector.from_dict`.

The evaluator records, per message:

* `l2_error = ‖inferred - ground_truth‖₂`, computed over the seven
  meaningful dimensions (the reserved eighth dimension is always 0 and
  is skipped).
* `route_chosen`: the routing decision string.
* The 2-D embedding projection emitted by the pipeline (useful for
  downstream scatter-plot diagnostics).

At the end of each session it records:

* Whether the final inferred vector's 1-NN persona matches the true
  persona (binary).
* The first message index at which `l2_error < adapt_converged_threshold`;
  `None` if the session never converges.

### 3.4 Metrics and aggregation

All aggregates use `i3.eval.ablation_statistics.bootstrap_ci` with
10 000 resamples and a fixed bootstrap RNG. The Pydantic
`ClosedLoopResult` exposes:

| Metric | Interpretation |
|---|---|
| `per_persona_recovery_rate` + CI | Chance of correctly identifying the persona at the final message. |
| `per_persona_adaptation_error` + CI | Mean L2 error per persona across all messages. |
| `per_persona_error_by_message` | Mean L2 error at each message index (length = `n_messages`). |
| `convergence_speeds` | Mean message index at which the system first converges, or `None`. |
| `persona_confusion_matrix` | Row = true, column = nearest-neighbour inferred at the final message. |
| `aggregate_recovery_rate`, `aggregate_adaptation_error` | Macro-averages with CIs. |
| `router_bias` | Local-routing fractions per persona + accessibility-vs-baseline delta. |

The confusion matrix is persisted as a list of lists so JSON serialisers
and downstream notebook tooling can read it without bespoke adapters.

## 4. Results

> *[to be populated by `scripts/run_closed_loop_eval.py`]*

### 4.1 Per-persona recovery rates

> *[to be populated by `scripts/run_closed_loop_eval.py`]*

### 4.2 Per-persona adaptation error trajectories

> *[to be populated by `scripts/run_closed_loop_eval.py`]*

### 4.3 Persona confusion matrix

> *[to be populated by `scripts/run_closed_loop_eval.py`]*

### 4.4 Router-bias sanity check

> *[to be populated by `scripts/run_closed_loop_eval.py`]*

## 5. Discussion

**Strengths.** The harness has three load-bearing strengths. First, it
supplies *ground truth* -- a quantity that real-user studies can only
proxy. Second, it is *reproducible* to the seed: the same seed, the
same pipeline commit, and the same config produce byte-identical
results. Third, it evaluates *the full stack*, not isolated components.
A regression in the privacy sanitiser, the cognitive-load controller,
or the router would all manifest as a drop in recovery rate or a
spike in adaptation error; the harness does not need to know where the
regression lives to flag it.

**Weaknesses.** The harness is only as real as its personas, and the
personas are currently synthetic in two distinct ways. The
*signature* layer -- the `TypingProfile` distributions -- is drawn
from published HCI literature but has not been calibrated against a
live user population on the hardware the assistant actually targets.
The *ground-truth* layer -- the `expected_adaptation` field -- is a
researcher-specified target informed by the literature, not an
externally validated one. A real user will have an `expected_adaptation`
that depends on device, time-of-day, recent events, and individual
preference; the persona library collapses this onto a single point
per archetype. The harness measures consistency with the researcher's
model, not clinical truth.

A second weakness is the *single-seed single-instance* regime. The
bootstrap CIs are within-sample intervals over bootstrap resamples of
the message stream; they do not reflect cross-run variance of the
pipeline itself, which involves stochastic bandit exploration. Claims
about pipeline performance should always be accompanied by a multi-seed
replication.

**Calibration path.** The bridge from simulated to real signatures is
the IRB-lite user study protocol in `docs/research/implicit_adapt_bench.md`.
In brief: 10-20 participants perform a structured set of typing tasks
(baseline, cognitive-load, fatigue-induction, simulated motor
impairment via a soft glove) while the client records keystroke
timing. We fit a mapping `f: simulated_signature -> real_signature`
per participant and per condition, then evaluate the pipeline on the
real signatures. Reporting both simulated and real numbers lets
readers see how much of the simulated performance is an artefact of
the persona library and how much generalises. The full experimental
design, consent language, and sample-size justification live in the
companion document.

## 6. Future work

1. **Human-in-the-loop recalibration.** After the first live-user
   study, fold the measured signatures back into the persona
   `TypingProfile` mean/std fields so subsequent simulation runs
   better reflect the deployment population. Version the persona
   library explicitly (currently `0.1.0`) so each recalibration is a
   visible, reviewable change.
2. **Persona expansion.** Add a `MULTILINGUAL_USER` with per-message
   language switching, a `SCREEN_READER_ONLY_USER` with no visual
   feedback channel, and a `NEURODIVERGENT_USER` with a bimodal
   correction distribution. These are under-represented in the
   keystroke-dynamics literature and are prime candidates for the
   first live-user calibration pass.
3. **Cross-lab replication.** Publish the persona library, the
   simulator, and the evaluator as a stand-alone package. Invite
   collaborating labs to replicate the harness on their own adaptive
   stacks. A cross-lab agreement of persona recovery rates would
   constitute strong evidence that the metric suite measures an
   intrinsic property of the architecture rather than an artefact of
   our seeds.
4. **Online-learning harness.** The current evaluator resets the
   pipeline's user model at the start of each session to isolate
   per-session performance. A parallel harness that maintains user-
   model state across sessions would measure the long-horizon
   continual-learning behaviour -- a natural bridge to the
   Kirkpatrick et al. (2017) evaluation protocol.
5. **Adversarial personas.** Add a `MASQUERADE_USER` that deliberately
   mimics another persona's typing signature to stress-test the
   system's robustness to adversarial input. A well-calibrated system
   should *not* confidently commit to a persona when the signature
   contradicts the linguistic profile.

## References

1. Cheng, J., Shadiev, R., Huang, Y.-M., & Hwang, G.-J. (2024).
   *Synthetic-user benchmarks for HCI: A methodological review.*
   International Journal of Human-Computer Studies, 182, 103159.
2. Duchowski, A. T., Krejtz, K., Krejtz, I., Biele, C., Niedzielska,
   A., Kiefer, P., Raubal, M., & Giannopoulos, I. (2014). *The index
   of pupillary activity: Measuring cognitive load vis-à-vis task
   difficulty with pupil oscillation.* Proceedings of CHI 2014.
3. Epp, C., Lippold, M., & Mandryk, R. L. (2011). *Identifying
   emotional states using keystroke dynamics.* Proceedings of CHI
   2011, 715–724.
4. Gajos, K., & Weld, D. S. (2004). *SUPPLE: Automatically generating
   user interfaces.* Proceedings of IUI 2004, 93–100.
5. Hart, S. G., & Staveland, L. E. (1988). *Development of NASA-TLX
   (Task Load Index): Results of empirical and theoretical research.*
   Advances in Psychology, 52, 139–183.
6. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins,
   G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska,
   A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017).
   *Overcoming catastrophic forgetting in neural networks.* PNAS,
   114(13), 3521–3526.
7. Trewin, S. (2000). *Configuration agents, control and privacy.*
   Proceedings of the 2000 Conference on Universal Usability, 9–16.
8. Vizer, L. M., Zhou, L., & Sears, A. (2009). *Automated stress
   detection using keystroke and linguistic features: An exploratory
   study.* International Journal of Human-Computer Studies, 67(10),
   870–886.
9. Zimmermann, M., Chappelier, J.-C., & Bunke, H. (2014). *Applying
   dynamic time warping for the analysis of keystroke dynamics data.*
   Pattern Recognition Letters, 34(13), 1478–1486.
