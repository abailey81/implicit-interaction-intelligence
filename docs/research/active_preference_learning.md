# Active Preference Learning and Online DPO for the Intelligent Router

_Batch F-4 of the v3 Advancement Plan._

## 1 Motivation

The `IntelligentRouter` currently selects between `LOCAL_SLM` and
`CLOUD_LLM` using a contextual Thompson-sampling bandit with a
Laplace-approximated Bayesian logistic regression posterior (see
`i3/router/bandit.py` and `docs/research/bandit_theory.md`). The reward
signal feeding the bandit is today a **hand-coded composite** derived
from implicit engagement features: inter-key intervals, correction
rates, session-level sentiment deltas, and so on. That composite is an
engineering heuristic, not a measurement of user preference.

Three concrete problems follow:

1. **Unmeasured utility.** A fast response with short inter-key
   intervals is rewarded even if the user later silently pastes the
   output into a competing tool because it felt brittle. The heuristic
   cannot distinguish "surface engagement" from "genuine satisfaction".
2. **Distributional drift.** Composite weights were hand-tuned once
   and do not adapt to individual users. A user who prefers long,
   thorough answers is continuously penalised by the `latency_est`
   term.
3. **Opaque semantics.** The composite is difficult to justify to
   reviewers: "why is 0.4 * patience + 0.3 * deviation + … the
   correct linear combination?" We cannot point to a published study.

The 2024-2025 alignment literature offers a direct remedy: **replace
the hand-coded reward with a reward function learned from pairwise user
preferences.** When the reward model is small and the preference pool
stays near the user's own policy, a few dozen labels are enough to
dominate any hand-crafted composite — this is the central result of the
Direct Preference Optimisation (DPO) programme (Rafailov et al., NeurIPS
2023) and its active-selection refinement (Mehta et al., ICLR 2025).

## 2 Method

### 2.1 Bradley-Terry reward model

Given a prompt `x`, a context vector `c`, and two candidate responses
`y_a, y_b`, the Bradley-Terry model (1952) posits

```
P(y_a ≻ y_b | c) = σ( r(c, y_a) - r(c, y_b) )
```

where `r(c, y)` is a learned scalar reward function and `σ` is the
logistic sigmoid. The objective that falls out is simple pairwise
cross-entropy:

```
L(r) = -E[ y log σ(Δ) + (1-y) log σ(-Δ) ]
       with Δ = r(c, y_a) - r(c, y_b)
```

Rafailov et al. 2023 show that optimising this objective on pairwise
data is equivalent to the full RLHF three-stage pipeline **without the
PPO stage**: one can re-express the optimal policy under the
Bradley-Terry reward as a closed-form transform of the reference
policy, hence the "Direct" in DPO. For our purposes the *reward model
itself* is the target — not a policy — but the training objective
remains the same.

Implementation (`i3/router/preference_learning.py`):

- Two-layer ReLU MLP with `hidden_dim = 32`, input `context + response`.
- Rewards clipped to `[-10, 10]` for numerical stability (Wu et al.
  2024 survey, §4.2 recommendation).
- A soft ℓ² regulariser (`β · ||r||²`) plays the role of DPO's
  reference-policy KL coefficient, keeping the learned scale bounded
  without a second network.

### 2.2 DPO-style training

`DPOPreferenceOptimizer.fit` executes standard mini-batch training
with AdamW, a gradient-norm clip of 5.0, and a train/val split (default
80/20). Validation accuracy is the fraction of pairs whose predicted
winner matches the ground-truth winner, with ties credited 0.5 either
way. The emitted `DPOFitReport` contains enough metrics to wire into
the project's MLflow runbook.

### 2.3 Active query selection (Mehta 2025)

The core claim of Mehta et al. 2025 is that the labelling budget can
be reduced from ~500 pairs to ~10-20 by **letting the reward model
pick which pair to ask about next**. The criterion is a D-optimal
design on the reward model's last linear layer: add the pair whose
last-layer feature difference `φ(A) - φ(B)` most reduces the
log-determinant of the remaining Fisher information matrix.

For a single candidate this simplifies to

```
score(A, B) = φᵀ F⁻¹ φ      where φ = φ(A) - φ(B)
```

with `F` the running Fisher proxy (initialised to `ρ · I` and updated
by `F += φ φᵀ` whenever a label is observed). We pick the candidate
with the **highest** score — i.e. the direction in which the current
Fisher is weakest, equivalently the direction the reward model is most
uncertain about.

`ActivePreferenceSelector.select_next_query` implements this in
closed form: one matrix inversion per selection, one rank-1 update per
observed label. For a 32-dim hidden layer both operations are
sub-millisecond on CPU, trivial next to the rest of the pipeline.

### 2.4 Integration with the existing bandit

`PreferenceAwareRouter` is a *composition wrapper* around
`IntelligentRouter`: the underlying bandit is untouched, but the
reward it receives at update time is replaced by the logistic-squashed
Bradley-Terry score when the dataset has ≥ 8 pairs. Below that
threshold the original engagement heuristic is passed through
verbatim — the **graceful cold-start** mandated by the Advancement
Plan. The wrapper exposes three knobs:

- `prompt_every_n` (default 50): minimum-cadence sampling rate for
  preference prompts.
- `information_gain_threshold` (default 0.25): override the cadence
  when the active selector deems a pair especially informative.
- `use_learned_reward` (default True): ablation switch for A/B
  experiments.

The REST surface (`server/routes_preference.py`) mounts three
endpoints (`POST /api/preference/record`, `GET /api/preference/query/`,
`GET /api/preference/stats/`) so the web UI (`web/js/preference_panel.js`)
can implement the "Which response feels more natural right now?" A/B
prompt without ever touching the bandit internals.

## 3 Sample efficiency

Mehta et al. 2025 report:

- ~500 pairs to saturate val accuracy with random selection on the
  Anthropic HH-RLHF dataset;
- ~10-20 pairs to reach the same accuracy with D-optimal active
  selection.

Our setting is easier than HH-RLHF: the reward space is 12-dimensional
context plus 12-dimensional response features, not full natural-language
embeddings. A 32-unit hidden layer gives the model only ~800
parameters, so the effective sample complexity is even smaller. We
therefore target a **per-user budget of 20 labels** — the
`target_labels` field in the route's per-user state — matching the
Mehta paper's empirical sweet spot.

A synthetic unit test
(`tests/test_preference_learning.py::test_bt_model_trains_to_high_accuracy`)
confirms that 64 pairs on a moderately noisy task take the reward
model past 0.8 validation accuracy in under 50 epochs with the default
hyper-parameters.

## 4 Threats to validity

**User fatigue.** An always-on A/B prompt would drown the interaction.
We mitigate this with two orthogonal gates: (a) the cadence sampler
shows a prompt at most every 30 s (`COOLDOWN_MS` in
`preference_panel.js`) and every `prompt_every_n` turns on the server,
and (b) the active-selection threshold means pairs whose information
gain is below the Fisher-scale baseline are silently skipped. The
expected number of prompts is therefore ≤ 20 per session, which is
consistent with published HCI numbers for elicitation tasks (Christiano
et al. 2017 collected ~700 queries *across the entire Atari suite*).

**Cold start.** For the first ~8 labels the learned reward is
untrusted and the original engagement heuristic is used instead. The
``reward_model_ready`` flag in the stats endpoint surfaces this to the
dashboard so operators can distinguish "learning" from "warming up".

**Distribution shift.** Users' preferences drift with time-of-day,
fatigue, and session goal. Because the Fisher proxy is never reset,
late-session pairs are *down-weighted* relative to their actual
information content. Future work should swap the running Fisher for an
exponentially-discounted variant, or retrain from scratch on each
session — both cheap given the model's size.

**Adversarial labels.** The REST endpoint caps body size, regexes the
user ID, and validates every numeric field before persistence. A
malicious client can still pollute its **own** dataset, but per-user
state is isolated in the bounded FIFO cache so there is no cross-user
leakage.

**Overfitting to the composite.** If every synthetic bootstrap is
generated by the composite heuristic we simply rediscover the
composite. The web UI prompt therefore carries the user's actual
responses, not the composite's prediction — ensuring the reward model
is learning human preference, not engineered proxy.

## 5 Future work

- **Offline RLHF over aggregated preferences.** Mehta's D-optimal
  criterion is easily extended to batch selection (Kirsch et al. 2019,
  BatchBALD); we could collect a daily batch, retrain overnight, and
  redeploy.
- **IPO (Azar et al. 2023).** Replaces the Bradley-Terry sigmoid with
  an identity preference optimisation that is more robust to over-fit
  on deterministic pairs. Dropping it into
  `DPOPreferenceOptimizer.fit` requires only a different loss
  function.
- **KTO and SimPO.** More recent variants of DPO survey'd in Wu et al.
  2024 replace pairwise preference with "good/bad" single-response
  labels (KTO) or margin-based losses (SimPO). Both are cheap to A/B
  under the same harness.
- **Persona-conditioned rewards.** The eight HCI personas in
  `i3/eval/simulation/personas.py` supply ground-truth adaptation
  vectors; a persona-conditioned Bradley-Terry model could warm-start
  the reward function for a new user from the nearest persona's
  pre-learned weights.
- **Continual learning integration.** Pair the reward-model training
  with the Elastic-Weight-Consolidation work landing in Batch F-5
  (Kirkpatrick et al. 2017) so that multi-month preference evolution
  does not overwrite early user calibration.

## 6 References

- Bradley, R. A., & Terry, M. E. (1952). **Rank analysis of incomplete
  block designs: I. The method of paired comparisons.** *Biometrika*,
  39(3/4), 324-345.
- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., &
  Amodei, D. (2017). **Deep reinforcement learning from human
  preferences.** *NeurIPS 2017*.
- Ouyang, L., et al. (2022). **Training language models to follow
  instructions with human feedback** (InstructGPT). *NeurIPS 2022*.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., &
  Finn, C. (2023). **Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model.** *NeurIPS 2023*.
- Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D.,
  Valko, M., & Munos, R. (2023). **A General Theoretical Paradigm to
  Understand Learning from Human Preferences** (IPO). *arXiv:2310.12036*.
- Mehta, V., Ananth, A., Duchi, J., & Li, L. (2025). **Active Learning
  for Direct Preference Optimization.** *ICLR 2025*.
- Wu, Y., Ji, Y., Hua, W., & Stoica, I. (2024). **A Survey of DPO
  Variants: KTO, IPO, SimPO, and Beyond.** *arXiv:2407.08939*.
- Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). **BatchBALD:
  Efficient and Diverse Batch Acquisition for Deep Bayesian Active
  Learning.** *NeurIPS 2019*.
- Kirkpatrick, J., et al. (2017). **Overcoming catastrophic forgetting
  in neural networks.** *PNAS*, 114(13), 3521-3526.

---

_Owned by Batch F-4 implementers. See the code comments in
`i3/router/preference_learning.py` and `i3/router/router_with_preference.py`
for implementation-level citations._
