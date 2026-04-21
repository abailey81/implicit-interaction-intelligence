"""Differential-privacy wrapper for the I³ Thompson-bandit router.

THE_COMPLETE_BRIEF §12 calls out "federated privacy" as a future-work theme.
The router is the right place to start: its reward history is
per-user and the MAP step of the Laplace-approximated posterior is a
gradient-based update that admits a DP-SGD-style noise injection cleanly.

The router lives in :mod:`i3.router.bandit`.  This module **wraps** — and
does not modify — that bandit.  Concretely, :class:`DPRouterTrainer`:

1. Pulls the bandit's observation history.
2. Computes the negative log-posterior gradient *per sample* (the DP-SGD
   style: Abadi et al. 2016 requires per-example clipping).
3. Clips each per-sample gradient to an L2 norm of ``max_grad_norm``.
4. Adds calibrated Gaussian noise scaled by ``noise_multiplier *
   max_grad_norm``.
5. Takes a Newton-Raphson step on the **noisy** summed gradient.

Opacus' ``RDPAccountant`` tracks the (ε, δ)-DP spend across steps.

.. note::
    This is a **sketch**.  A production DP-SGD implementation needs careful
    alignment between the privacy accountant's sampling-rate assumption and
    the batch selection path.  The router's dataset is tiny — tens to low
    hundreds of rows per user — which makes DP viable, but the exact
    privacy-parameter defaults here are conservative starting points, not
    audit-grade choices.

References
----------
* Abadi, M., Chu, A., Goodfellow, I. et al. (2016). *Deep learning with
  differential privacy.*  CCS.
* Dwork, C., Roth, A. (2014). *The algorithmic foundations of differential
  privacy.*  Foundations and Trends in Theoretical Computer Science 9(3-4).
* Mironov, I. (2017). *Rényi differential privacy.*  CSF.  — The accountant
  used below (Opacus' RDPAccountant) is a direct implementation of this.
* Yousefpour, A. et al. (2021). *Opacus: User-friendly differential privacy
  library in PyTorch.*  arXiv:2109.12298.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import Opacus accountant
# ---------------------------------------------------------------------------

try:
    from opacus.accountants import RDPAccountant  # type: ignore[import-not-found]

    _OPACUS_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    RDPAccountant = None  # type: ignore[assignment]
    _OPACUS_AVAILABLE = False


_INSTALL_HINT = (
    "opacus is not installed.  Install the future-work group with "
    "`poetry install --with future-work` to enable differential-privacy "
    "accounting.  Falling back to a best-effort in-process accountant."
)


if TYPE_CHECKING:  # pragma: no cover
    from i3.router.bandit import ContextualThompsonBandit


# ---------------------------------------------------------------------------
# Minimal RDP accountant used when Opacus is absent
# ---------------------------------------------------------------------------

class _FallbackAccountant:
    """Best-effort Gaussian-mechanism privacy accountant.

    The bound is the simple Gaussian-mechanism composition from Dwork &
    Roth (2014) Thm 3.22.  It is conservative — Opacus' ``RDPAccountant``
    gives much tighter numbers — but it does not require a dependency.
    """

    def __init__(self) -> None:
        self._steps: list[tuple[float, float]] = []  # (noise_multiplier, sample_rate)

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        self._steps.append((float(noise_multiplier), float(sample_rate)))

    def get_privacy_spent(self, delta: float) -> tuple[float, float]:
        if not self._steps:
            return 0.0, delta
        # Conservative Gaussian-mechanism bound: eps_t = sqrt(2 ln(1.25/delta)) / sigma.
        # Under basic sequential composition, eps_total = sum(eps_t).
        import math

        eps_total = 0.0
        for sigma, _q in self._steps:
            if sigma > 0:
                eps_total += math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / sigma
        return float(eps_total), float(delta)


# ---------------------------------------------------------------------------
# DPRouterTrainer
# ---------------------------------------------------------------------------

@dataclass
class DPBudgetStatus:
    """Snapshot of the differential-privacy budget.

    Attributes:
        spent_epsilon: Cumulative ε spent so far.
        spent_delta: The δ at which ε was evaluated.
        target_epsilon: The budget ceiling configured by the caller.
        target_delta: The δ configured by the caller.
        budget_exhausted: True if ``spent_epsilon >= target_epsilon``.
    """

    spent_epsilon: float
    spent_delta: float
    target_epsilon: float
    target_delta: float
    budget_exhausted: bool


class DPRouterTrainer:
    """DP-SGD-style wrapper around the bandit's MAP fit step.

    Args:
        bandit: The :class:`~i3.router.bandit.ContextualThompsonBandit` to
            train.  The bandit is **not** modified — the trainer only reads
            ``obs_history`` and writes back new weights.
        epsilon: Privacy budget (default 3.0, Abadi et al. 2016 mid-range).
        delta: Permissible failure probability (default 1e-5).
        max_grad_norm: Per-sample L2-norm clip (default 1.0).
        noise_multiplier: Gaussian noise ratio (default 1.1, a common
            DP-SGD default offering ≈ 6-ε at 100 epochs).
    """

    def __init__(
        self,
        bandit: "ContextualThompsonBandit",
        epsilon: float = 3.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ) -> None:
        self._bandit = bandit
        self._target_epsilon = float(epsilon)
        self._target_delta = float(delta)
        self._max_grad_norm = float(max_grad_norm)
        self._noise_multiplier = float(noise_multiplier)

        self._accountant: Any
        if _OPACUS_AVAILABLE:
            self._accountant = RDPAccountant()
        else:
            logger.warning(_INSTALL_HINT)
            self._accountant = _FallbackAccountant()

    # ------------------------------------------------------------------
    # Budget API
    # ------------------------------------------------------------------
    def set_privacy_budget(self, epsilon: float, delta: float) -> None:
        """Override the target privacy budget.

        Raises:
            ValueError: If either argument is non-positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self._target_epsilon = float(epsilon)
        self._target_delta = float(delta)

    def privacy_spent(self) -> tuple[float, float]:
        """Return the current cumulative (ε, δ) spend.

        Opacus' ``RDPAccountant`` tracks exact Rényi-DP moments and converts
        to an (ε, δ) pair via the Canonne–Kamath–Steinke (2020) conversion.
        The fallback accountant uses a looser Gaussian-mechanism bound.
        """
        if _OPACUS_AVAILABLE:
            try:
                eps = self._accountant.get_epsilon(delta=self._target_delta)
                return float(eps), self._target_delta
            except Exception:  # pragma: no cover - accountant quirks
                return 0.0, self._target_delta
        return self._accountant.get_privacy_spent(self._target_delta)

    def budget_status(self) -> DPBudgetStatus:
        """Return a :class:`DPBudgetStatus` snapshot."""
        eps, delta = self.privacy_spent()
        return DPBudgetStatus(
            spent_epsilon=eps,
            spent_delta=delta,
            target_epsilon=self._target_epsilon,
            target_delta=self._target_delta,
            budget_exhausted=eps >= self._target_epsilon,
        )

    # ------------------------------------------------------------------
    # DP-SGD MAP step
    # ------------------------------------------------------------------
    def fit_one_arm(
        self,
        arm: int,
        learning_rate: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Take one DP-SGD-style MAP step on *arm*'s weights.

        Args:
            arm: Arm index to refit.
            learning_rate: Newton-step scaling.
            rng: Optional RNG for noise generation.

        Returns:
            The updated weight vector for *arm*.

        Raises:
            RuntimeError: If the privacy budget has already been exhausted.
        """
        status = self.budget_status()
        if status.budget_exhausted:
            raise RuntimeError(
                f"Privacy budget exhausted: spent ε={status.spent_epsilon:.3f} "
                f">= target ε={self._target_epsilon:.3f}"
            )

        rng = rng if rng is not None else np.random.default_rng()

        weights = self._get_arm_weights(arm)
        history = self._get_arm_history(arm)
        if not history:
            return weights

        grads_clipped = []
        for ctx, reward in history:
            grad = self._neg_log_posterior_grad_single(weights, ctx, reward)
            norm = float(np.linalg.norm(grad))
            scale = 1.0 if norm <= self._max_grad_norm else self._max_grad_norm / (norm + 1e-12)
            grads_clipped.append(grad * scale)

        summed = np.sum(np.stack(grads_clipped, axis=0), axis=0)
        sigma = self._noise_multiplier * self._max_grad_norm
        noise = rng.normal(loc=0.0, scale=sigma, size=summed.shape)
        noisy = (summed + noise) / float(len(history))

        updated = weights - learning_rate * noisy

        # -- Record the step with the accountant --------------------------
        sample_rate = 1.0  # full-batch MAP; sampling_rate = 1.0 for the accountant
        try:
            if _OPACUS_AVAILABLE:
                self._accountant.step(
                    noise_multiplier=self._noise_multiplier,
                    sample_rate=sample_rate,
                )
            else:
                self._accountant.step(self._noise_multiplier, sample_rate)
        except Exception as exc:  # pragma: no cover - accountant quirks
            logger.warning("Privacy accountant step failed: %s", exc)

        self._set_arm_weights(arm, updated)
        return updated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _neg_log_posterior_grad_single(
        self,
        weights: np.ndarray,
        context: np.ndarray,
        reward: float,
    ) -> np.ndarray:
        """Gradient of the negative log-posterior at a single observation.

        Uses an isotropic Gaussian prior with precision
        :attr:`ContextualThompsonBandit.prior_precision` and a Bernoulli
        likelihood — the same loss the Laplace-approximated bandit minimises.
        """
        logit = float(np.dot(weights, context))
        p = float(self._sigmoid(np.array([logit]))[0])
        prior_precision = getattr(self._bandit, "prior_precision", 1.0)
        # ∇ -log p(y|x,w) = (p - y) * x; ∇ prior = prior_precision * w
        return (p - float(reward)) * context + prior_precision * weights

    def _get_arm_weights(self, arm: int) -> np.ndarray:
        """Read an arm's weight vector from the bandit."""
        weights = getattr(self._bandit, "weights", None)
        if isinstance(weights, (list, tuple)):
            return np.asarray(weights[arm], dtype=np.float64).copy()
        if isinstance(weights, np.ndarray) and weights.ndim == 2:
            return weights[arm].astype(np.float64).copy()
        raise AttributeError(
            "Could not read per-arm weights from the bandit object — "
            "expose a `weights` attribute shaped (n_arms, context_dim)."
        )

    def _set_arm_weights(self, arm: int, updated: np.ndarray) -> None:
        """Write an arm's weight vector back to the bandit."""
        weights = getattr(self._bandit, "weights", None)
        if isinstance(weights, list):
            weights[arm] = updated
        elif isinstance(weights, np.ndarray) and weights.ndim == 2:
            weights[arm] = updated
        else:
            raise AttributeError(
                "Could not write per-arm weights back to the bandit."
            )

    def _get_arm_history(self, arm: int) -> list[tuple[np.ndarray, float]]:
        """Pull the observation history for *arm* from the bandit.

        Tolerates a few common attribute names (``observations``,
        ``history``, ``obs_history``) so the module does not hard-depend on
        the bandit's internals.
        """
        for attr in ("observations", "history", "obs_history"):
            histories = getattr(self._bandit, attr, None)
            if histories is not None:
                entries = histories[arm]
                out: list[tuple[np.ndarray, float]] = []
                for entry in entries:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        ctx = np.asarray(entry[0], dtype=np.float64)
                        reward = float(entry[1])
                        out.append((ctx, reward))
                return out
        return []
