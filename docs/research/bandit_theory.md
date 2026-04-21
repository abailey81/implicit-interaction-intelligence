# Contextual Thompson Sampling for Cloud-vs-Local Routing

A paper-style note on the routing bandit — a two-arm contextual Thompson
sampler with Bayesian logistic regression posteriors, a Laplace
approximation, Newton–Raphson MAP, and a mandatory privacy override.

!!! note "TL;DR"
    We model per-arm reward as Bernoulli with a logistic link on the
    12-dimensional routing context. The weights have a Gaussian prior and a
    Gaussian-approximate posterior via Laplace. Thompson sampling draws a
    weight per arm, scores, and picks the argmax. Privacy-sensitive topics
    bypass the sampler entirely.

## 1. Problem setup { #setup }

At each message, observe context \(x \in \mathbb{R}^{12}\). Choose
arm \(a \in \{local, cloud\}\). Receive scalar reward
\(r \in [0, 1]\) — a composite of latency, quality proxy, and cost:

\[
r = \alpha_L \phi(\ell) + \alpha_Q q - \alpha_C c,
\quad \alpha_L + \alpha_Q + \alpha_C = 1.
\]

We want a policy \(\pi\) minimising cumulative regret under
non-stationary rewards (user preferences drift, device pressure changes,
network cost varies).

## 2. Model { #model }

Per arm \(a\):

\[
p(r = 1 \mid x, w_a) = \sigma(w_a^\top x),
\quad
w_a \sim \mathcal{N}(0, \lambda^{-1} I),
\]

with \(\sigma(u) = 1/(1 + e^{-u})\). Binarise continuous \(r\) at 0.5 for
the Bernoulli link.

### Prior

\(\lambda^{-1} = 1.0\) (weak, dimensionally reasonable). Stronger priors
cause cold-start conservatism.

### Likelihood

Given data
\(\mathcal{D}_a = \{(x_i, r_i)\}\) collected while arm \(a\) was chosen:

\[
\log p(\mathcal{D}_a \mid w_a) =
  \sum_i r_i \log \sigma(w_a^\top x_i) +
         (1-r_i) \log(1-\sigma(w_a^\top x_i)).
\]

### Posterior

Intractable in closed form. We use the Laplace approximation:

\[
p(w_a \mid \mathcal{D}_a) \approx \mathcal{N}(\mu_a, \Sigma_a),
\]

with

\[
\mu_a = \arg\max_w \left[ \log p(\mathcal{D}_a \mid w) + \log p(w) \right],
\quad
\Sigma_a^{-1} = \lambda I + \sum_i \sigma_i (1 - \sigma_i) x_i x_i^\top,
\]

where \(\sigma_i = \sigma(\mu_a^\top x_i)\). The MAP \(\mu_a\) is computed
with Newton–Raphson (see §4).

## 3. Thompson sampling { #thompson }

At each step:

1. For each arm \(a\), draw \(\tilde w_a \sim \mathcal{N}(\mu_a, \Sigma_a)\).
2. Score \(\tilde p_a = \sigma(\tilde w_a^\top x)\).
3. Choose \(\arg\max_a \tilde p_a\).

Thompson sampling has sub-linear Bayesian regret in contextual linear
bandits [^1][^2]. It is particularly well suited here because:

- **Non-stationarity.** As the user's reward distribution drifts, the
  posterior widens naturally where data is stale; exploration revives.
- **Calibration.** A single sample per arm is enough — no tunable
  exploration parameter to keep in sync with data.
- **Integration.** Sampling from a Gaussian posterior is cheap; no need
  for an exploration bonus that requires careful scaling.

See [ADR 0003](../adr/0003-thompson-sampling-over-ucb.md) for the full
comparison with UCB.

## 4. MAP via Newton–Raphson { #map }

We iteratively apply

\[
\mu_a \gets \mu_a - H^{-1} g,
\]

with gradient

\[
g = \lambda \mu_a + \sum_i (\sigma_i - r_i) x_i
\]

and Hessian

\[
H = \lambda I + \sum_i \sigma_i (1 - \sigma_i) x_i x_i^\top,
\]

which is positive-definite by construction. Typical convergence is under
15 iterations; the tolerance is \(\|g\|_\infty < 10^{-4}\).

## 5. Cold-start fallback { #cold-start }

For the first \(n_0 = 5\) observations per arm we use a **Beta-Bernoulli**
model, marginally:

\[
p(r=1 \mid a) \approx \mathrm{Beta}(\alpha_a, \beta_a),
\quad
\alpha_a = 1 + \#\text{successes},
\quad
\beta_a  = 1 + \#\text{failures}.
\]

Sample from each Beta, choose the argmax. This avoids the Newton iteration
exploding on tiny data.

## 6. Privacy override { #override }

Before any sampling, we check

\[
\mathbb{1}\{\mathrm{sensitivity}(x_\text{raw}) = 1\} \Rightarrow a = local.
\]

The override is mandatory and pre-sampling — the router cannot even draw
\(\tilde w_a\) for the cloud arm when the check fires. See
[Privacy architecture](../architecture/privacy.md).

## 7. Refit schedule { #schedule }

We refit \(\mu_a, \Sigma_a\) every `laplace_refit_every` observations
(default 10). This caps per-decision compute at an \(\mathcal{O}(1)\)
matrix-vector sample + scoring. The refit itself is \(\mathcal{O}(n d^2)\).

## 8. Regret analysis { #regret }

Under standard assumptions (sub-Gaussian rewards, bounded context norm,
Gaussian prior), Russo & Van Roy [^2] give

\[
\mathrm{Reg}_\pi(T) = \tilde{\mathcal{O}}(d\sqrt{T}).
\]

With \(d=12\) and \(T\) = messages per user per day, the constant is the
practical bottleneck. Empirically, on the 10 k-message synthetic trace
the online router reaches within 3 % of the Bayes-optimal policy by
~200 messages.

## 9. Implementation notes { #impl }

- **Numerical stability.** We clamp \(\sigma_i \in [10^{-6}, 1-10^{-6}]\)
  before forming the Hessian to avoid zero diagonals when confident.
- **Covariance storage.** We store \(\Sigma_a^{-1}\) (precision matrix)
  not \(\Sigma_a\). Sampling uses a Cholesky solve.
- **Scaling.** All 12 features are squashed to \([-1, 1]\) in
  `i3/router/complexity.py` / `sensitivity.py`, so the prior variance is
  calibrated across features.

## 10. References { #refs }

[^1]: Agrawal, S. and Goyal, N. "Thompson sampling for contextual bandits with linear payoffs." **ICML** (2013).
[^2]: Russo, D. and Van Roy, B. "Learning to optimize via posterior sampling." **Math. Oper. Res.** (2014).
