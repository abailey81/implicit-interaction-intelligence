# Differential Privacy for the I³ Router — Research Note

> *"Federated learning as privacy architecture, not bolt-on."*
> — THE_COMPLETE_BRIEF §12

## 1. Why DP matters for the router, specifically

The I³ router is a contextual Thompson-sampling bandit. Its posterior is
per-user: each user's reward history (continuation, sentiment, latency) is
consumed by a Laplace-approximated Bayesian logistic regression to update
per-arm weights. Two risks follow:

1. **Membership inference.** Small reward histories (tens to low hundreds of
   observations) make the posterior an easy target for membership-inference
   attacks: an adversary with access to the bandit weights can infer with
   non-trivial probability whether a particular interaction was used to
   fit them.
2. **Feature-level leakage.** The Laplace posterior encodes, implicitly, the
   distribution of the 12-dim routing context — including sensitivity-
   classifier outputs. Even though sensitive topics are *routed* locally,
   the *fact that they were routed locally* is written into the posterior,
   and that fact can be read back.

Differential privacy (Dwork & Roth 2014) provides a principled defence.
Below, we restrict ourselves to (ε, δ)-DP: the usual relaxation that admits
Gaussian-mechanism composition and is compatible with the RDP accountant.

## 2. DP-SGD applied to the MAP step

The Laplace-approximated bandit refits every `refit_interval` updates via
Newton-Raphson on the MAP of the posterior. A single Newton step is a
gradient-descent move on the negative log-posterior

$$
\mathcal{L}(w) \;=\; \sum_{i} \mathrm{BCE}(y_i, \sigma(w^\top x_i))
             \;+\; \tfrac{1}{2}\lambda\|w\|^2.
$$

**DP-SGD** (Abadi et al. 2016) instruments the gradient with two steps:

1. *Per-sample clipping.* Clip each $\nabla_w \ell_i$ to an L2 norm of
   $C$ (``max_grad_norm``). This bounds each individual's influence on
   the update.
2. *Gaussian noise injection.* Add $\mathcal{N}(0, \sigma^2 C^2 I)$ to the
   summed clipped gradient, where $\sigma$ is ``noise_multiplier``.

The update is then $w \leftarrow w - \eta \cdot \frac{1}{n}\left(\sum_i \mathrm{clip}_C(\nabla \ell_i) + \mathcal{N}(0, \sigma^2 C^2 I)\right)$.

`DPRouterTrainer.fit_one_arm` in `i3/privacy/differential_privacy.py` is
exactly this, with Opacus' `RDPAccountant` tracking spent ε and δ across
steps.

## 3. Why the MAP step, not the Thompson sampling step

Thompson sampling is already randomised — one might naïvely hope the
sampling provides some privacy. It does not: a single posterior sample
deterministically reveals the posterior *shape*, and over many rounds an
observer can reconstruct the weight means.

The MAP step is the right place to intervene because:

- It is where user-specific observations first touch the weights.
- Its gradient is a well-understood additive sum over per-example terms —
  DP-SGD's native setting.
- The Laplace covariance is refit as a side effect. Perturbing the MAP
  propagates the noise into the covariance cleanly; there is no second
  place where un-noised gradients enter the model.

## 4. Privacy-accounting considerations

`DPRouterTrainer` pairs three parameters:

| Parameter | Default | Source |
|:---|:---:|:---|
| `epsilon` | 3.0 | Abadi et al. (2016) mid-range; Apple's private federated learning targets similar budgets. |
| `delta` | 1e-5 | Below $1/n$ for all realistic user counts. |
| `max_grad_norm` | 1.0 | Standard DP-SGD default. Clipping too tight destroys signal; too loose destroys privacy. |
| `noise_multiplier` | 1.1 | ≈ 6-ε at 100 epochs under RDP composition. Sufficient headroom for the small number of MAP refits per user. |

### Per-user budget

The bandit is per-user. `epsilon=3.0` is therefore a **per-user lifetime
budget**. When the user's cumulative ε reaches the target, `fit_one_arm`
raises `RuntimeError`: further refits would break the privacy guarantee.
The router continues to serve — it just stops updating the posterior.

For the expected number of refits per user (`~50 / session × 50 sessions
= 2500` MAP calls), the RDPAccountant spend under `noise_multiplier=1.1`
stays below `ε=3.0` with a safe margin. Callers who need tighter budgets
should raise `noise_multiplier` to 1.5+ and accept slower convergence.

### Amplification by sampling

The module sets `sample_rate=1.0` because the router's MAP step uses the
entire arm history (full-batch). Privacy amplification by subsampling
(Balle et al. 2018) is **not** available to us. This is a real cost — a
mini-batched variant of the Newton step would sharpen the bound — but it
adds machinery the bandit does not currently have.

### Clipping-norm selection

The natural MAP gradient norm scales with the number of observations and
the prior precision. For the default `prior_precision=1.0` and a bandit
with 50-100 observations per arm, a per-sample gradient L2 norm of 0.3-0.8
is typical — so `max_grad_norm=1.0` leaves headroom without truncating
most updates. Callers with heavier-tailed reward signals should measure
gradient norms empirically before tightening the clip.

## 5. What we don't claim

- **Not audited.** The module is a sketch. A deployment would need
  independent verification that the gradient path actually respects DP —
  including confirmation that no non-DP copy of the gradients leaks into
  telemetry, logs, or the Laplace covariance.
- **Not composable with federated DP.** If the router is later federated
  (`i3/federated/`), the per-client DP-SGD budget composes with any
  aggregate DP applied at the server. That composition is another Abadi
  et al. (2016) problem; don't reuse this budget for both.
- **Not usable without a privacy-preserving evaluation story.** Reporting
  the router's offline regret numbers is itself a privacy release. Any
  evaluation pipeline must Laplace-noise the regret statistics or apply
  an equivalent output-perturbation step before publication.

## 6. References

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I.,
  Talwar, K., Zhang, L. (2016). *Deep learning with differential
  privacy.* CCS.
- Dwork, C., Roth, A. (2014). *The algorithmic foundations of
  differential privacy.* Foundations and Trends in Theoretical Computer
  Science 9(3-4).
- Mironov, I. (2017). *Rényi differential privacy.* CSF.
- Balle, B., Barthe, G., Gaboardi, M. (2018). *Privacy amplification by
  subsampling: tight analyses via couplings and divergences.* NeurIPS.
- Canonne, C., Kamath, G., Steinke, T. (2020). *The discrete Gaussian
  for differential privacy.* NeurIPS.
- Yousefpour, A. et al. (2021). *Opacus: User-friendly differential
  privacy library in PyTorch.* arXiv:2109.12298.
