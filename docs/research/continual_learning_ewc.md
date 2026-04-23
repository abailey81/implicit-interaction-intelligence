# Continual Learning for I³: Elastic Weight Consolidation


> document motivates why the existing three-timescale EMA is not, by
> itself, enough to avoid catastrophic forgetting in the shared TCN
> encoder, and specifies the EWC-based architecture that closes the gap.

## 1. Motivation

The current I³ user model tracks three temporal resolutions: an instant
state (most recent encoder output), a session-level EMA, and a
long-term per-user profile persisted to SQLite. This is an effective
plasticity-stability mechanism *at the level of the user profile*: the
long-term EMA changes slowly, so an unusual day does not overwrite
months of behaviour. However, the user profile is only one of the two
state variables exposed to the training loop. The other, and by far
the more expressive, is the **shared TCN encoder** that maps
interaction feature sequences to 64-dim unit-sphere embeddings.

When the encoder is continually fine-tuned — either in response to
online contrastive losses, reconstruction losses, or periodic batch
retraining on newly-collected diary data — its parameters evolve on a
timescale that the user-profile EMA is blind to. The encoder is shared
across users and sessions; a week of training dominated by the
"fatigued developer late-night" persona updates exactly the same
weights that produced "fresh morning user" embeddings a month earlier.
Nothing in the EMA prevents those updates from being *destructive*
with respect to the earlier regime. This is the classical neural
network problem of **catastrophic forgetting** (McCloskey & Cohen
1989; French 1999).

The consequence for I³ is concrete: the `current_vs_baseline` cosine
distance, which the router depends on to gate cloud calls and the TTS
layer uses to pick emotional tone, starts to degrade silently. Old
baseline embeddings remain on disk; the encoder that produced them no
longer lives. Every subsequent comparison is a stale measurement.

## 2. Method

### 2.1 Elastic Weight Consolidation (Kirkpatrick et al. 2017)

EWC treats the parameters found by training on task A as a Bayesian
prior for task B. Under a Laplace approximation of the posterior
p(θ | D_A), the log-posterior after learning task A is

    log p(θ | D_A, D_B) ≈ log p(D_B | θ) − (λ/2) · Σ_i F_i · (θ_i − θ*_A,i)²,

where F_i is the diagonal of the Fisher Information Matrix at θ*_A and
λ trades the two objectives. The Fisher diagonal is cheap:

    F_i ≈ E_{x ~ D_A} [ (∂ log p(y|x; θ) / ∂ θ_i)² ].

Implementation in `i3/continual/ewc.py` follows this directly. The
`estimate_fisher` method runs a fixed-size minibatch loop, zero-grads
each step, backpropagates the task loss, and squares the resulting
gradient before accumulating into a name-keyed diagonal dictionary. An
`fisher_epsilon` term is added after normalisation so parameters the
task barely activated still receive a non-zero penalty (Schwarz 2018
§3.1). `consolidate()` snapshots the current θ* alongside the Fisher,
and `penalty_loss()` returns a scalar tensor suitable for addition to
the next task's training loss. A `state_dict` / `load_state_dict` pair
enables the EWC state to be persisted alongside the model checkpoint.

### 2.2 Online EWC (Schwarz et al. 2018)

In a streaming user-modelling deployment there is no discrete
"end of task A" boundary. Online EWC replaces the one-shot Fisher with
an exponentially-decaying running average:

    F̃_t = γ · F̃_{t−1} + F_t,     θ̃*_t = θ_t,

with γ ∈ (0, 1]. This implements *Progress & Compress* — the model is
progressively compressed into an ever-growing diagonal prior as new
data arrives, without ever having to name the task boundaries
explicitly. `i3.continual.ewc.OnlineEWC` subclasses the vanilla EWC
implementation; only the accumulation rule differs.

### 2.3 Reservoir experience replay

Diagonal-Fisher EWC ignores off-diagonal covariance, so it is not a
complete regulariser. The standard complement is **experience replay**:
mix a small, unbiased sample of past interactions into the current
minibatch. We use Vitter 1985's reservoir sampling to maintain a fixed-
capacity buffer with the invariant that each of the n items observed so
far is stored with probability k / n. Chaudhry et al. 2019 ("On Tiny
Episodic Memories") showed that even 200-500 stored samples are
sufficient to close most of the remaining gap.

`i3.continual.replay_buffer.ReservoirReplayBuffer` implements the
classical Algorithm R; `ExperienceReplay` wraps it in a
`integrate_into_training(task_loss_fn)` helper that the caller can
apply inside their training step. The two mechanisms — parameter-
space regularisation via EWC and data-space regularisation via replay
— address distinct failure modes (Rolnick et al. 2019) and stack
linearly.

### 2.4 Drift-triggered consolidation

Discrete calls to `consolidate()` require a trigger. In I³ the trigger
is detected, not configured: when the behavioural distribution shifts
enough to matter, consolidate.

`i3.continual.drift_detector.ConceptDriftDetector` implements an
ADWIN-style adaptive window (Bifet & Gavaldà 2007) on the encoder's
scalar deviation magnitude. ADWIN keeps a variable-length sliding
window and searches for any split point at which the two sub-windows'
means differ by more than the Hoeffding bound

    ε_cut = sqrt( (1 / (2m)) · ln(4 / δ') ),   m = harmonic mean of sub-sizes,

where δ' = δ / (n − 1) applies a Bonferroni correction over the possible
splits. If such a split exists, the older half is discarded and a
`DriftDetectionResult` is raised through `on_drift_detected`. A
Population Stability Index helper (`population_stability_index`) adds a
complementary magnitude score (Siddiqi 2006): PSI > 0.25 is commonly
treated as a significant shift in financial modelling, and is a useful
second-opinion diagnostic.

### 2.5 Integration: `EWCUserModel`

`i3.continual.ewc_user_model.EWCUserModel` composes the above around
the existing `UserModel` *without modifying it*:

* `update_state(embedding, features)` forwards to the inner model and
  feeds the resulting deviation magnitude into the drift detector.
* When the detector fires, an `on_drift_detected` callback (wired in
  the constructor) invokes `ElasticWeightConsolidation.consolidate`
  with a dataloader either registered explicitly by the caller or
  reconstructed from the reservoir buffer.
* An optional external drift hook is available for observability
  dashboards.

Crucially, the composition is additive: the vanilla `UserModel` — the
one covered by the existing 140+-test suite under
`tests/test_user_model.py` — remains untouched. A deployment that
wants continual-learning guarantees wraps its user model at construction
time; a deployment that does not, does not.

## 3. Results (placeholder)

`scripts/demos/ewc.py` generates a three-persona curriculum
(FRESH_USER → FATIGUED_DEVELOPER → MOTOR_IMPAIRED_USER), trains the
TCN encoder twice — once with λ = 0 and once with the configured λ —
and reports the retention of FRESH_USER performance after each
subsequent task. The headline metric is the Δ column: positive values
mean EWC preserved more of task 0 than the baseline did. Kirkpatrick
et al. 2017 §3.2 report forgetting reductions of roughly 45 % on
permuted-MNIST for comparable λ and network sizes; we expect a similar
order of magnitude here, noting that (a) our personas are more
pairwise distinct than permuted-MNIST classes and (b) the TCN's 4-
block residual stack is much smaller than the MLPs in the original
paper. Results will be populated in a follow-up once the demo has been
run end-to-end on a real machine.

## 4. Threats to validity

**Empirical Fisher is a diagonal approximation.** The off-diagonal
covariance between parameters is discarded; for wide layers with
strongly correlated weights this underestimates the true importance of
joint updates. Aljundi et al. 2018's Memory Aware Synapses sidesteps
the likelihood assumption entirely by using output-sensitivity
gradients; blend MAS into the OnlineEWC update when we observe
instability.

**λ is sensitive.** Kirkpatrick 2017 swept λ across two orders of
magnitude for every task switch. Too small and the old tasks are
forgotten; too large and the new task cannot learn. Our default is
1000 based on Schwarz 2018; PA-EWC (arXiv 2511.20732, 2025) adapts λ
per-parameter based on a parameter-wise uncertainty estimate and would
be the natural next upgrade.

**ADWIN confidence is Hoeffding-based.** The bound assumes iid
observations; real user sessions violate this with autocorrelation
structure. The `min_sub_window` constant mitigates trivial alarms, but
a Page-Hinkley test or KSWIN statistic (Raab et al. 2020) would be
more principled for correlated streams.

**Empirical Fisher ≠ True Fisher.** Kunstner, Balles & Hennig 2019
pointed out the empirical Fisher can differ materially from the true
Fisher when the model is far from the MLE. We could use K-FAC (Martens
& Grosse 2015) for a block-diagonal approximation closer to the true
curvature; Schwarz et al. compare the two and report marginal gains,
so we leave it for future work.

## 5. Future work

1. **PA-EWC** (arXiv 2511.20732): per-parameter adaptive λ driven by
   gradient uncertainty; straightforward bolt-on.
2. **Memory Aware Synapses** (Aljundi 2018): replace Fisher with
   output-sensitivity gradients so the importance estimator no longer
   presumes a likelihood interpretation.
3. **Gradient Episodic Memory** (Lopez-Paz & Ranzato 2017): project
   gradients into the intersection of feasible directions for each
   past task, a harder constraint than EWC's soft penalty.
4. **Learning Without Forgetting** (Li & Hoiem 2017): distillation-
   style soft targets from the pre-update network; fits naturally into
   the existing DSPy / judge infrastructure.
5. **Generative replay** (Shin et al. 2017) once the local SLM can
   produce plausible synthetic interactions; removes the need for a
   reservoir buffer.
6. **Task-free detection with KSWIN** (Raab et al. 2020) for noisy,
   autocorrelated streams.

## 6. References

* Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., &
  Tuytelaars, T. (2018). "Memory Aware Synapses: Learning what (not) to
  forget". *ECCV*.
* Bifet, A., & Gavaldà, R. (2007). "Learning from Time-Changing Data
  with Adaptive Windowing". *SIAM International Conference on Data
  Mining*, 443-448.
* Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania,
  P. K., Torr, P. H. S., & Ranzato, M. (2019). "On Tiny Episodic
  Memories in Continual Learning". *arXiv 1902.10486*.
* French, R. M. (1999). "Catastrophic forgetting in connectionist
  networks". *Trends in Cognitive Sciences* 3(4):128-135.
* Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia,
  A. (2014). "A Survey on Concept Drift Adaptation". *ACM Computing
  Surveys* 46(4):44.
* Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins,
  G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-
  Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R.
  (2017). "Overcoming catastrophic forgetting in neural networks".
  *Proceedings of the National Academy of Sciences* 114(13):3521-3526.
* Kunstner, F., Balles, L., & Hennig, P. (2019). "Limitations of the
  empirical Fisher approximation for natural gradient descent".
  *NeurIPS*.
* Li, Z., & Hoiem, D. (2017). "Learning Without Forgetting". *IEEE
  TPAMI* 40(12):2935-2947.
* Lopez-Paz, D., & Ranzato, M. (2017). "Gradient Episodic Memory for
  Continual Learning". *NIPS*.
* McCloskey, M., & Cohen, N. J. (1989). "Catastrophic interference in
  connectionist networks: The sequential learning problem".
  *Psychology of Learning and Motivation* 24:109-165.
* arXiv 2511.20732 (2025). "Parameter-Adaptive Elastic Weight
  Consolidation" (PA-EWC).
* Raab, C., Heusinger, M., & Schleif, F.-M. (2020). "Reactive
  Soft Prototype Computing for Concept Drift Streams". *Neurocomputing*
  416:340-351.
* Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G.
  (2019). "Experience Replay for Continual Learning". *NeurIPS*.
* Schwarz, J., Luketina, J., Czarnecki, W. M., Grabska-Barwinska, A.,
  Teh, Y. W., Pascanu, R., & Hadsell, R. (2018). "Progress & Compress:
  A scalable framework for continual learning". *ICML*.
* Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). "Continual Learning
  with Deep Generative Replay". *NIPS*.
* Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and
  Implementing Intelligent Credit Scoring*. Wiley.
* Vitter, J. S. (1985). "Random sampling with a reservoir". *ACM
  Transactions on Mathematical Software* 11(1):37-57.
