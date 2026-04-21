# ADR-0003 — Thompson sampling over UCB

- **Status**: Accepted
- **Date**: 2026-01-08
- **Deciders**: Tamer Atesyakar
- **Technical area**: router (Layer 5)

## Context and problem statement { #context }

The router (Layer 5) must choose between the local SLM and the cloud LLM
based on a 12-dim context, per message, per user. The reward signal is
scalar in \([0, 1]\) (a latency / quality / cost composite) and varies
by user, time, and session.

We need an exploration strategy that:

- Handles **non-stationarity** — preferences drift, device pressure
  shifts, network cost moves.
- Integrates with a **Bayesian posterior** (we already want a
  logistic-regression model for calibration and priors).
- Adds no tunable exploration parameter the operator must baby-sit.

## Decision drivers { #drivers }

- Sub-linear Bayesian regret with realistic priors.
- No manual exploration-rate schedule.
- Compatible with Laplace-approximated Gaussian posteriors.
- Simple to audit: a single posterior sample per arm.

## Considered options { #options }

1. **Thompson sampling** with Laplace-approximated posteriors.
2. **Upper Confidence Bound (UCB)** — LinUCB or logistic UCB.
3. **Epsilon-greedy**.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — Thompson sampling. It handles
> non-stationary rewards naturally, integrates cleanly with our Laplace
> posterior, and requires no exploration parameter to tune.

### Consequences — positive { #pos }

- Posterior automatically widens on stale data, reviving exploration.
- Stateless exploration: a single sample per arm per decision.
- Straightforward privacy override — the override runs before sampling.
- Proven sub-linear Bayesian regret (Russo & Van Roy, 2014).

### Consequences — negative { #neg }

- Variance in chosen actions is higher than UCB at small sample sizes.
  *Mitigation*: Beta-Bernoulli cold-start for the first `cold_start_n`
  observations.
- Posterior sampling is slightly more expensive than a UCB scalar
  comparison. *Mitigation*: the matrix is 12 × 12; Cholesky + triangular
  solve is ~10 µs.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — UCB { #opt-2 }

- Yes Deterministic; easier to reproduce in tests.
- Yes Well-understood confidence-bound tuning.
- No Requires an exploration parameter (`c`) that is non-trivial to set
  correctly under a logistic reward model.
- No Does not adapt as gracefully as Thompson to non-stationary rewards.
- No Sample complexity constants are worse than Thompson in the
  contextual linear case at realistic \(d\) and \(T\).

### Option 3 — Epsilon-greedy { #opt-3 }

- Yes Simplest possible.
- No Exploration is uncorrelated with data uncertainty.
- No The \(\epsilon\) schedule is either too aggressive early or too lazy
  late; every user drifts on a different schedule.
- No Rejects Bayesian machinery we already want for calibration.

## References { #refs }

- [Research: Bandit theory](../research/bandit_theory.md)
- [Architecture: Router](../architecture/router.md)
- Agrawal, S. and Goyal, N. "Thompson sampling for contextual bandits
  with linear payoffs." ICML 2013.
- Russo, D. and Van Roy, B. "Learning to optimize via posterior sampling."
  Math. Oper. Res. 2014.
