"""Reptile meta-learning (Nichol et al. 2018).

Reptile is a first-order meta-learning algorithm that, empirically,
matches or outperforms FO-MAML at a fraction of the implementation
complexity. The update is simply:

.. math::

    \\theta \\leftarrow \\theta + \\varepsilon (\\theta' - \\theta)

where :math:`\\theta'` is the result of :math:`k` SGD steps on a
support set. No ``create_graph`` needed; no second-order derivatives;
just pull the meta-parameters toward the adapted ones.

The trainer here wraps the same encoder + linear adaptation head used
by :class:`~i3.meta_learning.maml.MAMLTrainer` so the two algorithms
are drop-in comparable.

Reference
---------
Nichol, A., Achiam, J., & Schulman, J. (2018). *On First-Order
Meta-Learning Algorithms.* arXiv:1803.02999.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3.meta_learning.maml import (
    MetaBatch,
    MetaTask,
    _messages_to_sequence,
    _target_tensor,
)

logger = logging.getLogger(__name__)


class ReptileTrainer:
    """Reptile meta-trainer (Nichol et al. 2018).

    Args:
        model: The encoder to meta-train. Same signature requirements as
            :class:`~i3.meta_learning.maml.MAMLTrainer`.
        inner_lr: Step size for the inner-loop SGD updates. Must be
            positive.
        outer_lr: Reptile's step size :math:`\\varepsilon` in
            :math:`\\theta \\leftarrow \\theta + \\varepsilon (\\theta' - \\theta)`.
            Conventionally in ``(0, 1]``; Nichol 2018 uses
            ``0.1``-``1.0`` across their experiments. Must be positive.
        inner_steps: Number of inner-loop SGD steps per task. Must be at
            least one.
        embedding_dim: Encoder output dimensionality.
        adaptation_dim: Target dimensionality.

    Raises:
        ValueError: If any of the numeric hyperparameters are invalid.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.1,
        inner_steps: int = 5,
        embedding_dim: int = 64,
        adaptation_dim: int = 8,
    ) -> None:
        if inner_lr <= 0.0:
            raise ValueError(f"inner_lr must be positive, got {inner_lr!r}.")
        if outer_lr <= 0.0:
            raise ValueError(f"outer_lr must be positive, got {outer_lr!r}.")
        if inner_steps < 1:
            raise ValueError(
                f"inner_steps must be at least 1, got {inner_steps!r}."
            )
        if embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be at least 1, got {embedding_dim!r}."
            )
        if adaptation_dim < 1:
            raise ValueError(
                f"adaptation_dim must be at least 1, got {adaptation_dim!r}."
            )
        self.model = model
        self.inner_lr = float(inner_lr)
        self.outer_lr = float(outer_lr)
        self.inner_steps = int(inner_steps)
        self.embedding_dim = int(embedding_dim)
        self.adaptation_dim = int(adaptation_dim)
        self.head: nn.Linear = nn.Linear(self.embedding_dim, self.adaptation_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inner_loop(self, task: MetaTask) -> tuple[nn.Module, nn.Linear, float]:
        """Adapt a fresh copy of the model to a single task.

        Args:
            task: The :class:`MetaTask` to adapt to.

        Returns:
            A tuple ``(encoder_copy, head_copy, final_support_loss)``.
        """
        encoder_copy = copy.deepcopy(self.model)
        head_copy = copy.deepcopy(self.head)
        params = list(encoder_copy.parameters()) + list(head_copy.parameters())
        optimiser = torch.optim.SGD(params, lr=self.inner_lr)

        support_x = _messages_to_sequence(task.support_set)
        support_y = _target_tensor(task.target_adaptation)

        last_loss = 0.0
        for _ in range(self.inner_steps):
            optimiser.zero_grad(set_to_none=True)
            embedding = encoder_copy(support_x)
            pred = head_copy(embedding)
            loss = F.mse_loss(pred, support_y.expand_as(pred))
            loss.backward()
            optimiser.step()
            last_loss = float(loss.detach().item())
        return encoder_copy, head_copy, last_loss

    def outer_step(self, batch: MetaBatch) -> dict[str, float]:
        """Perform one Reptile outer update on a batch of tasks.

        For each task, compute adapted weights and collect the
        difference ``theta_prime - theta``. The accumulated differences
        are averaged and added to ``theta`` with step size
        ``outer_lr``.

        Args:
            batch: A non-empty :class:`MetaBatch`.

        Returns:
            A dict with ``"meta_loss"`` (mean support loss across
            tasks) and ``"update_norm"`` (L2 norm of the outer-loop
            delta).
        """
        # Accumulate parameter deltas.
        enc_deltas: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
        }
        head_deltas: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p)
            for name, p in self.head.named_parameters()
        }
        total_support = 0.0

        for task in batch.tasks:
            adapted_enc, adapted_head, support_loss = self.inner_loop(task)
            total_support += support_loss
            # Accumulate theta' - theta.
            for (name, p), (_, p_prime) in zip(
                self.model.named_parameters(),
                adapted_enc.named_parameters(),
            ):
                enc_deltas[name] = enc_deltas[name] + (
                    p_prime.detach() - p.detach()
                )
            for (name, p), (_, p_prime) in zip(
                self.head.named_parameters(),
                adapted_head.named_parameters(),
            ):
                head_deltas[name] = head_deltas[name] + (
                    p_prime.detach() - p.detach()
                )

        n_tasks = float(len(batch.tasks))
        # Apply averaged delta with Reptile step size.
        update_sq = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                delta = enc_deltas[name] / n_tasks
                p.add_(delta, alpha=self.outer_lr)
                update_sq += float((delta * self.outer_lr).pow(2).sum().item())
            for name, p in self.head.named_parameters():
                delta = head_deltas[name] / n_tasks
                p.add_(delta, alpha=self.outer_lr)
                update_sq += float((delta * self.outer_lr).pow(2).sum().item())

        return {
            "meta_loss": total_support / n_tasks,
            "update_norm": float(update_sq**0.5),
        }

    def meta_train(
        self,
        task_generator: Iterator[MetaBatch],
        n_outer_steps: int = 10_000,
        log_every: int = 100,
    ) -> list[dict[str, float]]:
        """Run the full Reptile meta-training schedule.

        Args:
            task_generator: Iterator yielding :class:`MetaBatch` objects.
            n_outer_steps: Number of outer-loop updates to perform. Must
                be non-negative.
            log_every: How often to log progress, in outer steps. Must
                be positive.

        Returns:
            Per-step history as a list of metric dicts.

        Raises:
            ValueError: If ``n_outer_steps`` is negative or
                ``log_every`` is not positive.
        """
        if n_outer_steps < 0:
            raise ValueError(
                f"n_outer_steps must be non-negative, got {n_outer_steps!r}."
            )
        if log_every < 1:
            raise ValueError(
                f"log_every must be positive, got {log_every!r}."
            )
        history: list[dict[str, float]] = []
        for step in range(n_outer_steps):
            batch = next(task_generator)
            stats = self.outer_step(batch)
            stats["step"] = float(step)
            history.append(stats)
            if (step + 1) % log_every == 0:
                logger.info(
                    "Reptile step %d/%d  meta_loss=%.4f  update_norm=%.4f",
                    step + 1,
                    n_outer_steps,
                    stats["meta_loss"],
                    stats["update_norm"],
                )
        return history

    def state_dict(self) -> dict[str, object]:
        """Return a serialisable state dict (encoder + head)."""
        return {
            "encoder": self.model.state_dict(),
            "head": self.head.state_dict(),
            "hyperparameters": {
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
                "inner_steps": self.inner_steps,
                "embedding_dim": self.embedding_dim,
                "adaptation_dim": self.adaptation_dim,
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load a previously-saved state dict.

        Raises:
            KeyError: If any required top-level key is missing.
        """
        for key in ("encoder", "head"):
            if key not in state:
                raise KeyError(f"Missing required state_dict key {key!r}.")
        self.model.load_state_dict(state["encoder"])  # type: ignore[arg-type]
        self.head.load_state_dict(state["head"])  # type: ignore[arg-type]


__all__: list[str] = ["ReptileTrainer"]
