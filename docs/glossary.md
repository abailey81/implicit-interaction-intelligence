# Glossary

Every I³-specific term, acronym, and concept used in this documentation.
Definitions are deliberately short and link to the canonical treatment
where one exists.

!!! tip "Looking for something?"
    Use the search box (top right) — every term below is indexed.

## A { #a }

`AdaptationVector`
:   The 8-dim output of Layer 4. Contains `cognitive_load`, four-dim
    `style_mirror` (formality, verbosity, emotionality, directness),
    `emotional_tone`, and `accessibility`. See
    [Layers § L4](architecture/layers.md#l4).

`AdaptiveSLM`
:   The custom 6.3M-parameter transformer in `i3/slm/`. Self-attention,
    per-block cross-attention to user conditioning, FFN. See
    [ADR 0001](adr/0001-custom-slm-over-huggingface.md).

`AdaptiveTransformerBlock`
:   One layer of the SLM. Pre-LN self-attention, cross-attention to the
    4 × 256 conditioning tokens, feed-forward. See
    [Cross-attention conditioning](architecture/cross-attention-conditioning.md).

Arm (bandit)
:   One of the two routing options, `local` or `cloud`. See
    [Router](architecture/router.md).

## B { #b }

Baseline (user)
:   A user's long-term statistical profile — mean and variance of every
    feature, updated online with Welford's algorithm. Deviations are
    computed as z-scores against the baseline.

Bayesian logistic regression
:   The reward model underlying the Thompson sampler. Gaussian prior on
    the weights, logistic likelihood, Laplace-approximated Gaussian
    posterior. See [Bandit theory](research/bandit_theory.md).

Beta-Bernoulli
:   The cold-start fallback for the bandit. For fewer than five
    observations per arm, we sample from a Beta posterior on reward.

## C { #c }

Causal convolution
:   A 1-D convolution that only uses past inputs — i.e. at position `t`
    the receptive field is `[t - k + 1, t]`. Implemented in
    `i3/encoder/blocks.py::CausalConv1d`.

Cognitive load
:   The first dimension of the `AdaptationVector`. A scalar in \([0, 1]\)
    controlling the target linguistic complexity of the response.

Cold start
:   The phase before the bandit or the user model has enough data for
    calibrated posteriors. The router falls back to Beta-Bernoulli; the
    encoder's conditioning dropout produces a graceful null response.

Conditioning token
:   One of four 256-dim vectors produced by the `ConditioningProjector`
    from the concatenation of `AdaptationVector` (8-dim) and
    `UserStateEmbedding` (64-dim). Passed to every SLM block as K/V for
    cross-attention.

Contextual bandit
:   A bandit whose arm-reward distribution depends on an observed
    context vector. I³'s router uses a 12-dim context.

Cross-attention conditioning
:   The I³ architectural centrepiece: attention where the **query** comes
    from the token states and the **keys/values** from user-state
    conditioning tokens. See
    [Cross-attention conditioning](architecture/cross-attention-conditioning.md).

## D { #d }

Diary
:   The privacy-safe interaction log in `i3/diary/`. Stores embeddings,
    TF-IDF topics, and metrics — **never raw text**.

Dilation
:   A spacing between kernel taps in a convolution. The TCN's dilations
    `[1, 2, 4, 8]` produce an exponentially growing receptive field.

## E { #e }

Edge feasibility
:   The question of whether a model fits a target device's memory and
    latency budget. I³ measures this via `i3/profiling/`.

EMA (exponential moving average)
:   The update rule `x ← α * new + (1 - α) * x`. I³ uses two:
    session-level (`α = 0.3`) and long-term (`α = 0.1`).

## F { #f }

Fernet
:   A symmetric AEAD primitive (AES-128-CBC + HMAC-SHA256) from the
    Python `cryptography` library. Used for at-rest encryption of user
    profiles. See [ADR 0008](adr/0008-fernet-over-custom-crypto.md).

Flesch-Kincaid grade
:   A classical readability score. I³'s linguistic module implements it
    from scratch with a rule-based syllable counter.

## I { #i }

Implicit signal
:   Information about the user's cognitive, motor, or emotional state
    that comes from *how* they interact, not from what they explicitly
    say. I³'s core thesis is that this signal is rich enough to drive
    architectural personalisation.

Inter-key interval (IKI)
:   The time in milliseconds between two consecutive keystrokes. The
    most load-bearing feature in the 32-dim vector.

## K { #k }

KV cache
:   A per-layer tensor cache of keys and values computed during
    autoregressive decoding. Turns \(\mathcal{O}(T^2)\) repeated work
    into \(\mathcal{O}(T)\). Implemented in `i3/slm/attention.py`.

## L { #l }

Laplace approximation
:   A Gaussian approximation to an intractable posterior, centred at the
    MAP and with precision equal to the negative Hessian of the
    log-posterior at the MAP. Used on the bandit's per-arm logistic
    posterior.

Long-term profile
:   The third timescale of the user model. EMA with `α = 0.1`, persisted
    across sessions, Fernet-encrypted at rest.

## N { #n }

NT-Xent
:   Normalised temperature-scaled cross-entropy loss — the SimCLR
    objective used to train the TCN encoder. See
    [Contrastive loss](research/contrastive_loss.md).

Newton–Raphson
:   The iterative MAP solver for the bandit's logistic posterior. See
    [Bandit theory § MAP](research/bandit_theory.md#map).

## P { #p }

PII (personally identifiable information)
:   Category name for the ten regex patterns enforced by the sanitiser
    (email, phone, SSN, credit card, IBAN, address, IP, URL, DOB,
    passport). See [Privacy architecture § PII](architecture/privacy.md#pii).

Pipeline
:   The top-level async orchestrator in `i3/pipeline/engine.py` — a
    nine-step sequence from keystroke to response.

Pre-LN
:   Pre-layer-norm transformer variant — LayerNorm applied *before* each
    sub-layer rather than after. More stable at small width / shallow
    depth, which matters at 6.3M params.

Privacy override
:   The router's hard rule that any message containing a sensitive topic
    (health, mental health, financial credentials, security credentials)
    force-routes to `local`. Mandatory, pre-sampling.

Profiling
:   The cross-cutting `i3/profiling/` package. Measures parameter count,
    FP32/INT8 size, P50/P95/P99 latency, and emits a device feasibility
    Markdown report.

## Q { #q }

Quantization (INT8 dynamic)
:   Post-training conversion of the SLM's weight matrices to INT8, with
    scales computed dynamically at inference. Reduces on-disk size from
    ~25 MB to ~7 MB with minimal perplexity impact.

## R { #r }

Receptive field
:   For a stack of dilated convolutions with kernel size \(k\) and
    dilations \(\{d_\ell\}\), the receptive field is
    \(1 + (k-1)\sum_\ell d_\ell\). For I³'s TCN, ~61 timesteps.

Relationship strength
:   A derived scalar in \([0, 1]\) reflecting the number of sessions
    observed and the time span. Consumed by the adaptation controller.

Router
:   Layer 5 of I³ — the contextual Thompson sampling bandit. See
    [Router](architecture/router.md).

## S { #s }

`SessionState`
:   The in-memory within-session EMA. Discarded at `session_end`; its
    summary is EMA-merged into the long-term profile.

SLM (small language model)
:   A language model at the ~1–10M parameter scale, designed for edge
    deployment. I³'s SLM is 6.3M.

Sensitivity detector
:   `i3/router/sensitivity.py`. Regex-based classifier for the router's
    privacy override.

Style vector
:   A 4-dim sub-vector of `AdaptationVector`: formality, verbosity,
    emotionality, directness. Mirrored from the user's long-term profile.

## T { #t }

TCN (temporal convolutional network)
:   A stack of dilated causal convolutions with residual connections. Our
    encoder architecture. See
    [ADR 0002](adr/0002-tcn-over-lstm-transformer.md).

TF-IDF
:   Term frequency × inverse document frequency — the topic extraction
    used by the diary. Implemented from scratch with a ~175-word
    stopword list.

Thompson sampling
:   The bandit policy: draw a weight sample per arm from the posterior,
    score the context, pick the argmax. See
    [ADR 0003](adr/0003-thompson-sampling-over-ucb.md).

## U { #u }

`UserProfile`
:   The persistent record of a user — long-term means and variances,
    style baseline, relationship strength, last-seen timestamp.

`UserStateEmbedding`
:   The 64-dim output of the TCN encoder. Consumed by the adaptation
    controller, by the router's context-builder, and by the SLM's
    conditioning projector.

## W { #w }

Welford's online algorithm
:   A numerically stable, single-pass algorithm for incremental mean and
    variance. Avoids re-reading history. Used in `i3/user_model/deviation.py`.
