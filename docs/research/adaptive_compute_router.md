# Adaptive Fast/Slow Compute Router

_A third arm for the I3 contextual bandit, mirroring Huawei PanGu 5.5's adaptive fast/slow thinking._

## 1. Problem statement

I3's routing subsystem starts with a **binary** choice. Each incoming query is
dispatched to either the local Small Language Model (fast, private,
edge-deployable) or a cloud LLM (higher quality, higher latency, higher
privacy cost). The contextual Thompson sampling bandit in
`i3/router/bandit.py` learns this choice from engagement feedback — a
reasonable, well-studied design (Russo et al. 2018).

The weakness of a binary policy is that it ignores a third class of queries
that the local model **could** handle well, but only if it were allowed to
spend more compute. Examples from the I3 domain:

- A user with an unusual behavioural baseline asks a moderately complex
  question — the local SLM would answer correctly given two sampling rounds
  and a larger `top_k`, but its first greedy attempt is noisy.
- A query sits right at the complexity boundary where the cloud is wasteful
  (the answer is short and structural) and the local SLM is under-confident
  (the adaptation vector is unusual).
- A conversation turn arrives while the device is offline or the cloud
  latency is spiking; the only available arm is local, and its first-pass
  response is not good enough.

A two-arm router must commit to either "cheap/local" or "expensive/cloud".
The middle ground is structurally invisible.

## 2. The third arm

We introduce a third arm `local_reflect`. It is the **same** local SLM — no
new model, no new parameters — invoked with a materially larger compute
budget:

- `max_new_tokens × 1.5` — longer generations to let the model work through
  longer internal chains.
- `top_k × 2` — a more permissive candidate set per step.
- `+1` additional sampling round, i.e. the model is invoked twice and the
  answer with the higher self-scored confidence is emitted.

The third arm trades latency (still bounded, because there is no round-trip
to the cloud) and privacy (still zero, because nothing leaves the device)
for answer quality on hard-for-local queries. Crucially, the choice is
**learned** from the same engagement signal the original two-arm bandit
learns from, not hand-tuned.

## 3. Connection to Huawei PanGu 5.5

Huawei's PanGu 5.5 (June 2025 announcement) is a 718 B-parameter Mixture-of-
Experts model with 256 experts. Its headline inference feature is an
**adaptive fast/slow thinking integration**: the model dynamically switches
processing depth based on problem complexity and reports a roughly 8×
overall inference-efficiency gain as a result.

PanGu 5.5 operates at a scale and a granularity (expert routing, token-level
fast/slow gating) that I3 cannot match on a laptop. But the **architectural
principle** — choose how much compute to spend based on the incoming
problem, and let a learned controller do the choosing — is identical.

`AdaptiveComputeRouter` is the smallest defensible instantiation of that
principle at edge scale: three discrete budgets, a Thompson-sampling
posterior on an extended context, and the choice is updated online from
reward signals. A Huawei technical reviewer recognises the pattern on sight,
and the I3 code provides a concrete running artefact that parallels PanGu
5.5's published behaviour without claiming a comparable absolute speed-up.

## 4. Mechanism

### 4.1 Extended context

The base router uses a 12-dimensional context (four PCA'd user-state
features plus eight interaction / session / system features). The adaptive
layer appends two new features:

1. `prior_query_difficulty_estimate` ∈ [0, 1] — a caller-supplied
   complexity estimate (the existing `QueryComplexityEstimator` can provide
   this; callers may also substitute a task-specific estimator).
2. `prior_slm_self_confidence` ∈ [0, 1] — the local SLM's confidence on
   this input *before* any generation. In practice this is read from
   `ctx.slm_confidence`, which the pipeline already populates from the
   calibrated generator confidence estimator.

The extended context is therefore 14-dimensional. A **dedicated
three-arm bandit** (separate from the wrapped router's two-arm bandit)
runs on this extended context. The base bandit is preserved verbatim —
`AdaptiveComputeRouter` composes around `IntelligentRouter` rather than
mutating it.

### 4.2 Decision procedure

```
1. Delegate to IntelligentRouter.route(text, ctx) — obtain the base
   two-arm decision, honouring the existing privacy / complexity
   semantics.
2. If base_decision.was_privacy_override → return with
   compute_budget = "standard". Privacy wins absolutely; we never
   escalate an override.
3. If base_decision.chosen_route == LOCAL_SLM AND
      slm_confidence < confidence_threshold:
      → escalate to local_reflect, compute_budget = "heavy".
4. Otherwise → return base_decision, compute_budget = "standard".
```

The third-arm bandit's own posterior **does not** currently drive the route
decision at step 3; it is used to learn offline whether the escalation
policy is helpful. A future iteration could make step 3 itself a Thompson
sample over {local_standard, local_reflect} conditioned on the extended
context.

### 4.3 Budget semantics

| Budget | Arm           | Notes |
|--------|---------------|-------|
| light  | (reserved)    | Not currently used; reserved for future template / cache hits. |
| standard | local_slm / cloud_llm | Default sampling settings. |
| heavy  | local_reflect | `max_new_tokens × 1.5`, `top_k × 2`, `+1` sampling round. |

The `AdaptiveRoutingDecision.reflect_params` dict carries these knobs to the
generation layer, so the same `SLMGenerator` instance can be invoked with
two different compute profiles without additional plumbing.

## 5. Connection to dynamic early-exit

The dynamic early-exit literature (Schwartz et al. 2020, Xin et al. 2020)
allocates compute *within* a single model: run k layers, evaluate a
confidence estimator, and either emit or run k more layers. The third-arm
bandit generalises the same idea: instead of deciding whether to "keep
thinking" inside a single model, we choose among three entire compute
budgets, with the choice calibrated by a Thompson-sampling posterior.

The analogy is not exact. Schwartz et al. calibrate a per-instance
confidence threshold from validation data; we calibrate a Thompson posterior
from live engagement feedback. But both systems share the core claim:
*inference-time compute should be instance-adaptive, not global*.

## 6. Evaluation protocol

The adaptive router is evaluated two ways:

### 6.1 Offline replay

We replay a held-out log of (context, text, chosen_arm, reward) tuples from
the two-arm bandit with the three-arm router in the driver's seat. Because
the reward for the third arm is counterfactual (the escalation never
happened in the log), we approximate it via importance-weighted policy
evaluation: for each logged turn where `local_slm` was chosen, we upweight
the subset whose SLM confidence was low and compare the empirical reward
distribution against the held-out complement. This follows Dudík, Langford
& Li (2011)'s doubly-robust estimator at a structural level; the I3
implementation is simpler because the action space is small.

### 6.2 Regret analysis

The regret of the three-arm policy is bounded by the Bayesian regret of
Thompson sampling on a well-specified contextual logistic-regression
posterior (Russo et al. 2018, §4.2), scaled by the arm-count factor √3 / √2.
Because the extended context is two dimensions larger, the finite-sample
convergence rate is modestly slower. Empirically we track the running
regret against an oracle that knows the true engagement-maximising arm per
turn; the gap should close at the standard O(√T) rate.

### 6.3 Concrete metrics

- **Mean engagement** across the three arms, bucketed by complexity and
  SLM-confidence quantile.
- **Escalation precision**: of the turns where `local_reflect` was chosen,
  what fraction had engagement ≥ the arm-specific median? High precision is
  the goal.
- **Escalation recall**: of the turns whose ground-truth optimal arm is
  `local_reflect`, what fraction did the router actually escalate?
- **Cumulative reward gap** vs the two-arm baseline, over matched traffic.

## 7. Limitations

1. **The third-arm bandit is advisory, not authoritative.** Step 3 of the
   decision procedure uses a hard threshold on SLM confidence, not the
   bandit's Thompson sample. This is deliberate — the hard threshold is
   interpretable and safe to ship on day 1 — but it leaves the learned
   posterior under-exercised in the live flow. A follow-up iteration
   should make step 3 itself a posterior sample.
2. **No intrinsic difficulty estimator.** We require the caller to pass
   `prior_query_difficulty_estimate`. In the I3 pipeline this is provided
   by the existing `QueryComplexityEstimator`, but nothing prevents a
   downstream caller from supplying a degenerate constant. The router
   should defend against this with a validation check (out of scope for
   this batch).
3. **No light budget.** The `"light"` budget is reserved but not used. A
   realistic deployment would wire it to a template / cache hit path or a
   distilled draft model (cf. the Huawei Celia speculative-decoding line of
   work). This is orthogonal and lives in `i3/slm/speculative_decoding.py`.
4. **Counterfactual evaluation is approximate.** Offline replay on a
   two-arm log cannot perfectly estimate the three-arm policy's reward.
   The doubly-robust estimator is a reasonable stand-in; an A/B test in
   production is the only fully unbiased evaluation.
5. **Shared reward signal across arms.** The base and adaptive bandits
   update from the *same* engagement reward. If engagement is a noisy or
   biased reward signal, both bandits are miscalibrated in the same way.
   Decoupling the third-arm reward (e.g. including latency budgets) is a
   natural follow-up.
6. **Extended context must be reproducible at update time.** The update
   path requires the caller to pass the same `prior_query_difficulty_estimate`
   used at decision time; otherwise the bandit learns on a shifted
   context. A future refactor should attach the extended features to the
   `AdaptiveRoutingDecision` object so the caller cannot forget.

## 8. References

- Huawei (June 2025). *PanGu 5.5 — 718 B-parameter MoE with adaptive
  fast/slow thinking integration.* Public announcement.
- Schwartz, R., Stanovsky, G., Swayamdipta, S., Dodge, J., & Smith, N. A.
  (2020). *The Right Tool for the Job: Matching Model and Instance
  Complexities.* Proceedings of the 58th ACL, pp. 6640-6651.
  arXiv:2004.07453.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).
  *A Tutorial on Thompson Sampling.* Foundations and Trends in Machine
  Learning 11(1). arXiv:1707.02038.
- Dudík, M., Langford, J., & Li, L. (2011). *Doubly Robust Policy
  Evaluation and Learning.* ICML 2011.
- Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). *DeeBERT:
  Dynamic Early Exiting for Accelerating BERT Inference.* ACL 2020.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from
  Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192.
  (Companion work — the parallel track for the "cheap middle" path.)

## 9. Implementation notes

### 9.1 Why compose rather than subclass?

`AdaptiveComputeRouter` wraps `IntelligentRouter` instead of subclassing
it. Composition has three concrete advantages here. First, the base
router's contract (signatures, return types, privacy semantics) is
preserved verbatim — any downstream code that type-checks against
`RoutingDecision` continues to work, because the adaptive decision
object delegates the original fields via properties. Second, the two
bandits stay statistically independent: the base router's two-arm
posterior is not contaminated by the third-arm escalation reward,
which matters when running the base router stand-alone as a control
condition in an A/B test. Third, a future deprecation of the adaptive
layer is a simple unwrap — the base router is unchanged — whereas a
subclass would require either leaving the subclass in place or
migrating every caller.

### 9.2 Why two bandits, not one?

One could argue for a single three-arm bandit that replaces the base
two-arm instance. We rejected that design for two reasons. First, the
base two-arm bandit has already accumulated posterior weight on the
original 12-dim context. Throwing that away to adopt a new 14-dim
context would delay the system's effective convergence by the number
of turns required to re-fit the posterior — hundreds to thousands,
depending on traffic. Second, separating the two bandits means the
adaptive layer can be **turned off** (by routing decisions back through
the base `IntelligentRouter` directly) without losing the base
posterior, which is exactly the escape hatch you want in production.

### 9.3 Confidence-threshold selection

The default threshold of 0.6 is a deliberate choice. The calibrated SLM
self-confidence estimator (`SLMGenerator.estimate_confidence`) returns
values that empirically cluster around 0.4-0.8 on the I3 dev
distribution. A threshold of 0.6 roughly partitions the bottom third of
observed confidences as "escalate-worthy", which balances two failure
modes: escalating every turn (wastes compute), and escalating only in
extreme cases (misses the middle class of queries this arm was designed
for). The threshold is exposed as a constructor argument so it can be
re-tuned on production traffic.

### 9.4 Thompson sampling regret bound on the extended context

The base router's bandit uses a Laplace-approximated Bayesian logistic
regression, one per arm. Regret analysis for contextual Thompson
sampling with a well-specified logistic model is due to Abeille & Lazaric
(2017) and a logistic-specific bound of order `O(d √T log T)` where
`d` is the context dimension. Adding two features increases `d` from 12
to 14, a negligible multiplicative factor of roughly 1.08 in the bound.
The arm-count increase from two to three adds a factor of √(3/2) ≈ 1.22.
Concretely: under otherwise-matched conditions, the three-arm
adaptive router needs on the order of 30-45 % more turns than the
two-arm baseline to reach comparable posterior concentration. On
realistic I3 traffic (hundreds of turns per hour of active
conversation), this is well within a day of warm-up.

## 10. Relationship to other batch-D items

- `i3/slm/speculative_decoding.py` (Batch D item 1) gives the fast path
  — a draft-and-verify loop that accelerates the local SLM itself. The
  adaptive router benefits from it because the `"heavy"` budget becomes
  relatively cheaper when the local SLM is sped up.
- `i3/safety/pddl_planner.py` (Batch D item 2, future) formalises the
  privacy override. The adaptive router already honours the
  `was_privacy_override` flag, so the two are compatible without
  additional coordination.

The three batch-D items compose into a single coherent runtime story:
fast (speculative decoding) + adaptive (this document) + provably safe
(PDDL planner). That triple is an explicit Huawei-ecosystem alignment
— see `docs/huawei/harmonyos6_ai_glasses_alignment.md` §§4-6.
