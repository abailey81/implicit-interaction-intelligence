"""Contextual Thompson Sampling bandit with Laplace-approximated Bayesian
logistic regression.

This module implements the core decision engine for the Intelligent Router.
For each of the two arms (LOCAL_SLM and CLOUD_LLM), the bandit maintains:

1. A logistic-regression weight vector (posterior mean via MAP).
2. A posterior covariance matrix (Laplace approximation around the MAP).
3. A simple Beta-Bernoulli posterior as a cold-start fallback.

Thompson sampling works by:
    a) Sampling weight vectors from each arm's posterior.
    b) Computing the expected reward for each arm under the sampled weights.
    c) Selecting the arm with the highest sampled expected reward.

The Laplace approximation is refitted periodically (every ``refit_interval``
updates) using Newton-Raphson to find the MAP estimate, then computing the
Hessian at the MAP for the posterior covariance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
_LOGIT_CLIP = 10.0          # Clip logits to [-10, 10] for sigmoid stability
_EPSILON = 1e-6             # Small constant for numerical stability
_MIN_VARIANCE = 1e-8        # Floor for diagonal covariance entries
_NEWTON_ITERS = 8           # Newton-Raphson iterations for MAP
_COLD_START_PULLS = 5       # Use Beta fallback until this many pulls


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid function."""
    x_clipped = np.clip(x, -_LOGIT_CLIP, _LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class ContextualThompsonBandit:
    """Contextual Thompson Sampling with Laplace-approximated Bayesian
    logistic regression.

    Parameters:
        n_arms: Number of arms (default 2: LOCAL_SLM, CLOUD_LLM).
        context_dim: Dimensionality of the context vector (default 12).
        prior_precision: Precision (inverse variance) of the Gaussian prior
            on the logistic-regression weights.  Higher values shrink the
            weights toward zero more aggressively.
        exploration_bonus: Additive bonus to the logit before sigmoid, which
            gently encourages exploration.
        refit_interval: How often (in updates per arm) to refit the Laplace
            approximation.  Smaller values are more accurate but slower.

    Example::

        bandit = ContextualThompsonBandit(n_arms=2, context_dim=12)
        ctx = np.random.randn(12)
        arm, confidence = bandit.select_arm(ctx)
        bandit.update(arm, ctx, reward=1.0)
    """

    def __init__(
        self,
        n_arms: int = 2,
        context_dim: int = 12,
        prior_precision: float = 1.0,
        exploration_bonus: float = 0.1,
        refit_interval: int = 10,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if prior_precision <= 0:
            raise ValueError(f"prior_precision must be > 0, got {prior_precision}")

        self.n_arms: int = n_arms
        self.context_dim: int = context_dim
        self.prior_precision: float = prior_precision
        self.exploration_bonus: float = exploration_bonus
        self.refit_interval: int = refit_interval

        # --- Posterior parameters per arm ---
        # Logistic regression posterior (Laplace approximation)
        self.weight_means: list[np.ndarray] = [
            np.zeros(context_dim, dtype=np.float64) for _ in range(n_arms)
        ]
        self.weight_covs: list[np.ndarray] = [
            np.eye(context_dim, dtype=np.float64) / prior_precision
            for _ in range(n_arms)
        ]

        # Simple Beta-Bernoulli posteriors (cold-start fallback)
        self.alpha: list[float] = [1.0] * n_arms
        self.beta_param: list[float] = [1.0] * n_arms

        # --- Observation history per arm (for Laplace refitting) ---
        self.history: list[list[tuple[np.ndarray, float]]] = [
            [] for _ in range(n_arms)
        ]

        # --- Running statistics ---
        self.total_pulls: list[int] = [0] * n_arms
        self.total_rewards: list[float] = [0.0] * n_arms

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def select_arm(
        self, context: np.ndarray
    ) -> tuple[int, dict[str, float]]:
        """Sample from each arm's posterior and select the arm with the
        highest sampled expected reward.

        Args:
            context: A 1-D numpy array of shape ``(context_dim,)``.

        Returns:
            A tuple ``(arm_index, confidence)`` where ``confidence`` is a
            dict mapping ``"arm_0"``, ``"arm_1"``, ... to the normalised
            sampled probability for each arm.

        Raises:
            ValueError: If ``context`` has the wrong shape.
        """
        context = np.asarray(context, dtype=np.float64).ravel()
        if context.shape[0] != self.context_dim:
            raise ValueError(
                f"Expected context of dim {self.context_dim}, "
                f"got {context.shape[0]}"
            )

        sampled_rewards: list[float] = []

        for arm in range(self.n_arms):
            if self.total_pulls[arm] < _COLD_START_PULLS:
                # Cold start: draw from Beta-Bernoulli posterior
                sampled_p = float(
                    np.random.beta(self.alpha[arm], self.beta_param[arm])
                )
            else:
                # Sample weights from the multivariate-normal posterior
                try:
                    sampled_weights = np.random.multivariate_normal(
                        self.weight_means[arm], self.weight_covs[arm]
                    )
                except np.linalg.LinAlgError:
                    # Covariance not PSD (shouldn't happen, but be safe)
                    logger.warning(
                        "Covariance not PSD for arm %d; falling back to "
                        "diagonal sampling.",
                        arm,
                    )
                    std = np.sqrt(
                        np.maximum(np.diag(self.weight_covs[arm]), _MIN_VARIANCE)
                    )
                    sampled_weights = (
                        self.weight_means[arm]
                        + std * np.random.randn(self.context_dim)
                    )

                logit = float(context @ sampled_weights) + self.exploration_bonus
                sampled_p = float(_sigmoid(logit))

            sampled_rewards.append(sampled_p)

        chosen = int(np.argmax(sampled_rewards))

        # Build confidence dict (normalised sampled probabilities)
        total = sum(sampled_rewards)
        if total < _EPSILON:
            # Degenerate: all rewards near zero, uniform confidence
            confidence = {f"arm_{i}": 1.0 / self.n_arms for i in range(self.n_arms)}
        else:
            confidence = {
                f"arm_{i}": r / total for i, r in enumerate(sampled_rewards)
            }

        return chosen, confidence

    # ------------------------------------------------------------------
    # Posterior update
    # ------------------------------------------------------------------

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update the posterior for ``arm`` after observing ``reward``.

        Args:
            arm: Index of the arm that was pulled (0-indexed).
            context: The context vector that was used for the decision.
            reward: Observed reward signal, typically in [0, 1].
                Values > 0.5 are treated as successes for the Beta posterior.

        Raises:
            ValueError: If ``arm`` is out of range or ``context`` has the
                wrong shape.
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        context = np.asarray(context, dtype=np.float64).ravel()
        if context.shape[0] != self.context_dim:
            raise ValueError(
                f"Expected context of dim {self.context_dim}, "
                f"got {context.shape[0]}"
            )

        # 1. Update Beta-Bernoulli posterior
        if reward > 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta_param[arm] += 1.0

        # 2. Record observation for logistic regression
        self.history[arm].append((context.copy(), float(reward)))
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += float(reward)

        # 3. Refit Laplace approximation periodically
        if self.total_pulls[arm] % self.refit_interval == 0:
            self._refit_posterior(arm)

    # ------------------------------------------------------------------
    # Laplace approximation (Newton-Raphson MAP + Hessian)
    # ------------------------------------------------------------------

    def _refit_posterior(self, arm: int) -> None:
        """Refit the Laplace approximation for the given arm.

        Performs Newton-Raphson to find the MAP estimate of the logistic-
        regression weights under an isotropic Gaussian prior, then computes
        the Hessian at the MAP to form the posterior covariance.

        The logistic-regression model is:

            p(y=1 | x, w) = sigmoid(x^T w)

        with prior  w ~ N(0, (1/prior_precision) * I).

        The MAP objective (negative log-posterior) is:

            L(w) = -sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
                   + (prior_precision / 2) * ||w||^2

        Gradient:
            grad L = X^T (p - y) + prior_precision * w

        Hessian:
            H = X^T diag(p * (1-p)) X + prior_precision * I

        Posterior covariance = H^{-1}
        """
        observations = self.history[arm]
        n = len(observations)
        if n == 0:
            return

        # Stack observations into matrices
        X = np.zeros((n, self.context_dim), dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        for i, (ctx, reward) in enumerate(observations):
            X[i] = ctx
            # Binarise reward for logistic regression
            y[i] = 1.0 if reward > 0.5 else 0.0

        # Initialise weights at current posterior mean
        w = self.weight_means[arm].copy()

        # Newton-Raphson iterations
        prior_precision_mat = self.prior_precision * np.eye(
            self.context_dim, dtype=np.float64
        )

        for iteration in range(_NEWTON_ITERS):
            # Forward pass: compute predicted probabilities
            logits = X @ w  # (n,)
            p = _sigmoid(logits)  # (n,)

            # Clamp p away from 0 and 1 for numerical stability
            p = np.clip(p, _EPSILON, 1.0 - _EPSILON)

            # Gradient of negative log-posterior
            residuals = p - y  # (n,)
            grad = X.T @ residuals + self.prior_precision * w  # (d,)

            # Hessian of negative log-posterior
            s = p * (1.0 - p)  # (n,)  -- sigmoid derivative
            H = X.T @ (X * s[:, np.newaxis]) + prior_precision_mat  # (d, d)

            # Add small epsilon to diagonal for numerical stability
            H += _EPSILON * np.eye(self.context_dim, dtype=np.float64)

            # Newton step: w_new = w - H^{-1} @ grad
            try:
                # Use Cholesky for numerical stability (H is positive definite)
                L = np.linalg.cholesky(H)
                step = np.linalg.solve(L.T, np.linalg.solve(L, grad))
            except np.linalg.LinAlgError:
                # Fallback to standard solve if Cholesky fails
                logger.debug(
                    "Cholesky failed for arm %d iteration %d; using lstsq.",
                    arm,
                    iteration,
                )
                step, _, _, _ = np.linalg.lstsq(H, grad, rcond=None)

            w = w - step

            # Check for convergence
            if np.linalg.norm(step) < 1e-6:
                logger.debug(
                    "Newton-Raphson converged for arm %d at iteration %d.",
                    arm,
                    iteration,
                )
                break

        # Store MAP estimate
        self.weight_means[arm] = w

        # Compute posterior covariance = H^{-1} at the MAP
        # Recompute Hessian at final w
        logits = X @ w
        p = _sigmoid(logits)
        p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
        s = p * (1.0 - p)
        H = X.T @ (X * s[:, np.newaxis]) + prior_precision_mat
        H += _EPSILON * np.eye(self.context_dim, dtype=np.float64)

        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logger.warning(
                "Hessian inversion failed for arm %d; using pseudo-inverse.",
                arm,
            )
            cov = np.linalg.pinv(H)

        # Ensure covariance is symmetric and has positive diagonal
        cov = 0.5 * (cov + cov.T)
        np.fill_diagonal(cov, np.maximum(np.diag(cov), _MIN_VARIANCE))

        self.weight_covs[arm] = cov

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_arm_stats(self) -> dict[str, Any]:
        """Return summary statistics for each arm.

        Returns:
            A dict with keys ``"arms"`` (list of per-arm dicts) and
            ``"total_observations"`` (int).  Each arm dict contains
            ``"pulls"``, ``"total_reward"``, ``"mean_reward"``,
            ``"beta_alpha"``, and ``"beta_beta"``.
        """
        arms_stats: list[dict[str, Any]] = []
        for i in range(self.n_arms):
            pulls = self.total_pulls[i]
            total_reward = self.total_rewards[i]
            arms_stats.append(
                {
                    "arm": i,
                    "pulls": pulls,
                    "total_reward": total_reward,
                    "mean_reward": total_reward / pulls if pulls > 0 else 0.0,
                    "beta_alpha": self.alpha[i],
                    "beta_beta": self.beta_param[i],
                    "weight_norm": float(np.linalg.norm(self.weight_means[i])),
                }
            )
        return {
            "arms": arms_stats,
            "total_observations": sum(self.total_pulls),
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        """Serialise the bandit state to a JSON file.

        The observation history (context vectors and rewards) is included
        so that the Laplace approximation can be reconstructed on load.

        Args:
            path: Filesystem path for the output JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "prior_precision": self.prior_precision,
            "exploration_bonus": self.exploration_bonus,
            "refit_interval": self.refit_interval,
            "alpha": self.alpha,
            "beta_param": self.beta_param,
            "total_pulls": self.total_pulls,
            "total_rewards": self.total_rewards,
            "weight_means": [w.tolist() for w in self.weight_means],
            "weight_covs": [c.tolist() for c in self.weight_covs],
            "history": [
                [(ctx.tolist(), reward) for ctx, reward in arm_hist]
                for arm_hist in self.history
            ],
        }

        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)
        logger.info("Bandit state saved to %s", path)

    def load_state(self, path: str | Path) -> None:
        """Load bandit state from a JSON file created by :meth:`save_state`.

        All internal state (posteriors, history, statistics) is restored.

        Args:
            path: Filesystem path to the JSON state file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the loaded state has incompatible dimensions.
        """
        path = Path(path)
        with open(path) as fh:
            state: dict[str, Any] = json.load(fh)

        # Validate compatibility
        if state["n_arms"] != self.n_arms:
            raise ValueError(
                f"State has {state['n_arms']} arms but bandit has {self.n_arms}"
            )
        if state["context_dim"] != self.context_dim:
            raise ValueError(
                f"State has context_dim={state['context_dim']} but bandit "
                f"has {self.context_dim}"
            )

        self.prior_precision = state["prior_precision"]
        self.exploration_bonus = state["exploration_bonus"]
        self.refit_interval = state["refit_interval"]
        self.alpha = state["alpha"]
        self.beta_param = state["beta_param"]
        self.total_pulls = state["total_pulls"]
        self.total_rewards = state["total_rewards"]
        self.weight_means = [
            np.array(w, dtype=np.float64) for w in state["weight_means"]
        ]
        self.weight_covs = [
            np.array(c, dtype=np.float64) for c in state["weight_covs"]
        ]
        self.history = [
            [(np.array(ctx, dtype=np.float64), reward) for ctx, reward in arm_hist]
            for arm_hist in state["history"]
        ]
        logger.info(
            "Bandit state loaded from %s (%d total observations).",
            path,
            sum(self.total_pulls),
        )

    def reset(self) -> None:
        """Reset all internal state to the initial prior.

        Clears all observation history, resets posteriors to the prior,
        and zeroes all running statistics.
        """
        for arm in range(self.n_arms):
            self.weight_means[arm] = np.zeros(self.context_dim, dtype=np.float64)
            self.weight_covs[arm] = (
                np.eye(self.context_dim, dtype=np.float64) / self.prior_precision
            )
            self.alpha[arm] = 1.0
            self.beta_param[arm] = 1.0
            self.history[arm] = []
            self.total_pulls[arm] = 0
            self.total_rewards[arm] = 0.0
        logger.info("Bandit state reset to prior.")
