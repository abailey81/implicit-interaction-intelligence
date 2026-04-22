"""Model-Agnostic Meta-Learning (MAML) for few-shot user adaptation.

This module implements the MAML algorithm of Finn, Abbeel & Levine (2017)
on top of the User-State Encoder TCN defined in
:mod:`i3.encoder.tcn`. The goal is to meta-train a set of initial
parameters :math:`\\theta` such that, for any new user sampled from the
task distribution, a small number of SGD steps on a handful of support
messages yields a model that generalises well to that user's query
messages.

Two flavours are supported:

* **Second-order MAML** (``first_order=False``): computes the outer-loop
  gradient through the inner-loop updates via
  :func:`torch.autograd.grad` with ``create_graph=True``. Exact but
  expensive in memory and compute.
* **First-order MAML (FO-MAML)** (``first_order=True``): drops the
  second-order term by using ``create_graph=False``. Much cheaper and,
  as reported by Finn 2017 §5.2 and Nichol 2018, often competitive.

The inner-loop loss is the **adaptation MSE** between the
mean-pooled encoder output and the task's ground-truth
:class:`~i3.adaptation.types.AdaptationVector`. This is the natural
objective for few-shot *user identification*: the encoder must place the
user's embedding near the archetype target after a couple of updates.

Example
-------
    >>> import torch
    >>> from i3.encoder.tcn import TemporalConvNet
    >>> from i3.meta_learning import MAMLTrainer, PersonaTaskGenerator
    >>> from i3.eval.simulation import ALL_PERSONAS
    >>> model = TemporalConvNet()
    >>> trainer = MAMLTrainer(model, inner_lr=0.01, outer_lr=1e-3,
    ...                       inner_steps=3, first_order=True)
    >>> gen = PersonaTaskGenerator(ALL_PERSONAS[:4], seed=0)
    >>> batch = gen.generate_batch(meta_batch_size=4)
    >>> stats = trainer.outer_step(batch)  # doctest: +SKIP

References
----------
* Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic
  Meta-Learning for Fast Adaptation of Deep Networks.* ICML, pp. 1126-1135.
* Nichol, A., Achiam, J., & Schulman, J. (2018). *On First-Order
  Meta-Learning Algorithms.* arXiv:1803.02999.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.interaction.types import InteractionFeatureVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MetaTask(BaseModel):
    """A single meta-learning task: support + query split for one user.

    A MAML task is drawn from the task distribution :math:`p(\\mathcal{T})`.
    In this project, each task corresponds to a fresh draw from a single
    :class:`~i3.eval.simulation.personas.HCIPersona` -- the support set
    is used to adapt the model inside the inner loop, and the query set
    is used to compute the outer-loop meta-loss.

    Attributes:
        persona_name: Machine-friendly identifier of the source persona.
        support_set: List of :class:`InteractionFeatureVector` objects
            used to compute the inner-loop gradient update.
        query_set: List of :class:`InteractionFeatureVector` objects
            used to evaluate the adapted model. Must be non-empty.
        target_adaptation: Ground-truth adaptation vector that both the
            support and query losses are regressed against.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    persona_name: str = Field(min_length=1)
    support_set: list[InteractionFeatureVector] = Field(min_length=1)
    query_set: list[InteractionFeatureVector] = Field(min_length=1)
    target_adaptation: AdaptationVector


class MetaBatch(BaseModel):
    """A collection of :class:`MetaTask` used for one outer-loop step.

    The outer-loop meta-gradient is averaged across all tasks in the
    batch, following Finn 2017 Algorithm 1.

    Attributes:
        tasks: List of :class:`MetaTask` -- must contain at least one
            task.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tasks: list[MetaTask] = Field(min_length=1)

    def __len__(self) -> int:
        """Number of tasks in the batch."""
        return len(self.tasks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages_to_sequence(messages: list[InteractionFeatureVector]) -> torch.Tensor:
    """Stack a list of feature vectors into a ``[1, T, 32]`` tensor.

    Args:
        messages: Non-empty list of :class:`InteractionFeatureVector`.

    Returns:
        A float32 tensor of shape ``[1, len(messages), 32]`` with a
        singleton batch dimension (one user = one session).

    Raises:
        ValueError: If ``messages`` is empty.
    """
    if not messages:
        raise ValueError("Cannot build a sequence from an empty message list.")
    rows = [m.to_tensor() for m in messages]
    seq = torch.stack(rows, dim=0).unsqueeze(0)
    return seq.float()


def _target_tensor(target: AdaptationVector) -> torch.Tensor:
    """Convert an :class:`AdaptationVector` to a ``[1, 8]`` tensor."""
    return target.to_tensor().unsqueeze(0)


def _functional_forward(
    model: nn.Module,
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Stateless forward pass using an explicit parameter dict.

    Uses :func:`torch.nn.utils.stateless.functional_call` (aliased via
    :func:`torch.func.functional_call` in newer PyTorch) to evaluate the
    model with a detached-from-module parameter set. This is the
    cornerstone of second-order MAML: it allows us to keep the
    computation graph on the adapted parameters.

    Args:
        model: The encoder module.
        params: Dict mapping parameter names to tensors (possibly
            requiring grad and tracking history through inner-loop
            updates).
        x: Input tensor.

    Returns:
        The model's output tensor under the supplied parameters.
    """
    # torch.func.functional_call is the modern API (PyTorch >= 2.0).
    # Fallback to torch.nn.utils.stateless for older versions.
    func_call = getattr(torch.func, "functional_call", None)
    if func_call is None:
        # pragma: no cover - legacy path
        from torch.nn.utils.stateless import functional_call as _fc

        func_call = _fc
    return func_call(model, params, (x,))


def _project_to_adaptation(
    embedding: torch.Tensor,
    head: nn.Linear,
    head_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Project a 64-dim embedding to an 8-dim adaptation vector."""
    return _functional_forward(head, head_params, embedding)


# ---------------------------------------------------------------------------
# MAMLTrainer
# ---------------------------------------------------------------------------


class MAMLTrainer:
    """Meta-trainer implementing MAML (Finn et al. 2017).

    The trainer adds a lightweight 64->8 linear *adaptation head* on top
    of whatever encoder is supplied, so that the inner-loop loss is a
    well-defined regression from the encoder's embedding space to the
    8-dim :class:`AdaptationVector` target. Both the encoder parameters
    **and** the head parameters participate in the inner- and outer-loop
    updates.

    Args:
        model: The encoder to meta-train. Must accept a
            ``[batch, seq_len, input_dim]`` input and return a
            ``[batch, embedding_dim]`` output. The TCN defined in
            :mod:`i3.encoder.tcn` is the canonical choice, but any
            ``nn.Module`` with the right signature works.
        inner_lr: Step size for the inner-loop SGD updates
            (:math:`\\alpha` in Finn 2017). Must be positive.
        outer_lr: Step size for the outer-loop Adam updates
            (:math:`\\beta` in Finn 2017). Must be positive.
        inner_steps: Number of inner-loop gradient steps per task. Must
            be at least one.
        first_order: If ``True``, use FO-MAML (no second-order
            derivatives); otherwise use full second-order MAML.
        embedding_dim: Output dimensionality of the supplied encoder.
            Defaults to 64 to match the TCN.
        adaptation_dim: Dimensionality of the adaptation target.
            Defaults to 8 to match :class:`AdaptationVector`.

    Raises:
        ValueError: If any of the numeric hyperparameters are invalid.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 1e-3,
        inner_steps: int = 5,
        first_order: bool = False,
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
        self.first_order = bool(first_order)
        self.embedding_dim = int(embedding_dim)
        self.adaptation_dim = int(adaptation_dim)

        # Adaptation head: 64 -> 8 linear regression. Its weights are
        # meta-learned alongside the encoder's.
        self.head: nn.Linear = nn.Linear(self.embedding_dim, self.adaptation_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Outer-loop optimiser owns both the encoder and head params.
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.head.parameters()),
            lr=self.outer_lr,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inner_loop(self, task: MetaTask) -> nn.Module:
        """Run ``inner_steps`` SGD updates on the task's support set.

        This method returns a **new** :class:`nn.Module` whose parameters
        are the adapted parameters. The returned model does not share
        storage with ``self.model`` -- mutating it will not affect the
        meta-parameters.

        Args:
            task: The :class:`MetaTask` to adapt to.

        Returns:
            An ``nn.Module`` with the same architecture as ``self.model``
            whose parameters have been updated by ``inner_steps`` SGD
            steps on the task's support set.
        """
        adapted_params, _ = self._adapt(task, create_graph=False)
        # Materialise the adapted parameters into a fresh copy of the
        # model. We use ``copy.deepcopy`` so the caller gets an
        # autograd-disconnected snapshot suitable for evaluation.
        import copy

        adapted_model = copy.deepcopy(self.model)
        # ``adapted_params`` contains entries for both the encoder and
        # the head, keyed with ``encoder.*`` / ``head.*`` prefixes. Split
        # them back out for assignment.
        encoder_state = {
            k[len("encoder.") :]: v.detach().clone()
            for k, v in adapted_params.items()
            if k.startswith("encoder.")
        }
        # Non-strict load: tolerate any persistent buffers in the model
        # whose values we inherit unchanged from the deep-copy.
        adapted_model.load_state_dict(encoder_state, strict=False)
        return adapted_model

    def outer_step(self, batch: MetaBatch) -> dict[str, float]:
        """Run one outer-loop meta-update on a :class:`MetaBatch`.

        For each task, compute adapted parameters via the inner loop
        (with ``create_graph = not first_order``), evaluate the query
        loss under those parameters, and accumulate the meta-gradient.

        * **Second-order MAML** (``first_order=False``): the adapted
          parameters retain their autograd graph back to the
          meta-parameters, so a single ``meta_loss.backward()`` flows
          gradients through the entire inner-loop unroll.
        * **First-order MAML** (``first_order=True``): the inner loop
          detaches its updates, so we instead compute the query-loss
          gradient w.r.t. the *adapted* parameters and assign that
          directly to the corresponding meta-parameter ``.grad`` slots.
          This is the Finn 2017 §5.2 FO-MAML approximation.

        Args:
            batch: A non-empty :class:`MetaBatch`.

        Returns:
            A dict with ``"meta_loss"``, ``"mean_support_loss"``, and
            ``"mean_query_loss"`` averaged across the tasks in the
            batch.
        """
        self.model.train()
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)

        create_graph = not self.first_order
        total_query = 0.0
        total_support = 0.0
        n_tasks = float(len(batch.tasks))

        meta_params_list = list(self._named_params().items())
        if not self.first_order:
            # Second-order path: sum the per-task query losses and do a
            # single backward call. The autograd engine will accumulate
            # gradients on the meta-parameters for us.
            meta_query = torch.zeros((), dtype=torch.float32)
            for task in batch.tasks:
                adapted_params, support_loss = self._adapt(
                    task, create_graph=True
                )
                query_loss = self._task_query_loss(task, adapted_params)
                meta_query = meta_query + query_loss
                total_support += float(support_loss.detach().item())
            meta_loss = meta_query / n_tasks
            meta_loss.backward()
            total_query = float(meta_loss.detach().item())
        else:
            # First-order path: per task, compute d(query_loss)/d(adapted)
            # and add it to the meta-parameter gradient slots.
            for task in batch.tasks:
                adapted_params, support_loss = self._adapt(
                    task, create_graph=False
                )
                query_loss = self._task_query_loss(task, adapted_params)
                total_support += float(support_loss.detach().item())
                total_query += float(query_loss.detach().item())
                adapted_keys = list(adapted_params.keys())
                adapted_values = [adapted_params[k] for k in adapted_keys]
                grads = torch.autograd.grad(
                    query_loss, adapted_values, allow_unused=True
                )
                # Accumulate into the meta-parameter .grad slots, keyed
                # by the same name.
                for (name, mp), g in zip(meta_params_list, grads):
                    if g is None:
                        continue
                    contribution = g / n_tasks
                    if mp.grad is None:
                        mp.grad = contribution.detach().clone()
                    else:
                        mp.grad = mp.grad + contribution.detach()

        # Gradient clipping stabilises second-order MAML in particular.
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.head.parameters()),
            10.0,
        )
        self.optimizer.step()

        if self.first_order:
            mean_query = total_query / n_tasks
        else:
            mean_query = total_query
        return {
            "meta_loss": mean_query,
            "mean_support_loss": total_support / n_tasks,
            "mean_query_loss": mean_query,
        }

    def meta_train(
        self,
        task_generator: Iterator[MetaBatch],
        n_outer_steps: int = 10_000,
        log_every: int = 100,
    ) -> list[dict[str, float]]:
        """Run the full outer-loop meta-training schedule.

        Args:
            task_generator: An iterator yielding :class:`MetaBatch`
                instances. Typically a :class:`PersonaTaskGenerator`.
            n_outer_steps: Number of outer-loop updates to perform. Must
                be non-negative.
            log_every: How often to log progress (in outer steps).

        Returns:
            The full history as a list of per-step metric dicts.

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
                    "MAML step %d/%d  meta_loss=%.4f",
                    step + 1,
                    n_outer_steps,
                    stats["meta_loss"],
                )
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _named_params(self) -> dict[str, torch.Tensor]:
        """Build a single dict of meta-parameters keyed with a prefix.

        The encoder's parameters are stored under ``encoder.*`` and the
        head's under ``head.*`` so that a single functional_call on
        either module can retrieve the right subset.
        """
        params: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            params[f"encoder.{name}"] = p
        for name, p in self.head.named_parameters():
            params[f"head.{name}"] = p
        return params

    @staticmethod
    def _split(
        params: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Split a combined parameter dict into encoder / head halves."""
        enc: dict[str, torch.Tensor] = {}
        head: dict[str, torch.Tensor] = {}
        for k, v in params.items():
            if k.startswith("encoder."):
                enc[k[len("encoder.") :]] = v
            elif k.startswith("head."):
                head[k[len("head.") :]] = v
        return enc, head

    def _forward(
        self, params: dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Full encoder -> head forward pass under ``params``."""
        enc_params, head_params = self._split(params)
        embedding = _functional_forward(self.model, enc_params, x)
        return _functional_forward(self.head, head_params, embedding)

    def _adapt(
        self, task: MetaTask, *, create_graph: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Compute adapted parameters by running the inner loop.

        Args:
            task: The task to adapt to.
            create_graph: If ``True``, keep the computation graph on the
                inner-loop updates so that second-order outer gradients
                can flow back through them.

        Returns:
            A pair ``(adapted_params, final_support_loss)``.
        """
        support_x = _messages_to_sequence(task.support_set)
        support_y = _target_tensor(task.target_adaptation)

        params = self._named_params()
        last_loss: torch.Tensor = torch.zeros((), dtype=torch.float32)
        for _ in range(self.inner_steps):
            pred = self._forward(params, support_x)
            # Broadcast target across the single-batch prediction.
            loss = F.mse_loss(pred, support_y.expand_as(pred))
            last_loss = loss
            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=create_graph,
                allow_unused=True,
            )
            new_params: dict[str, torch.Tensor] = {}
            for (name, p), g in zip(params.items(), grads):
                if g is None:
                    # Parameter did not participate in the forward -- keep
                    # it unchanged but re-flag as requiring grad in the
                    # first-order path so subsequent inner steps can
                    # still backprop through it.
                    new_params[name] = (
                        p if create_graph else p.detach().requires_grad_(True)
                    )
                else:
                    updated = p - self.inner_lr * g
                    if not create_graph:
                        # First-order MAML: cut the graph between inner
                        # steps so we don't accumulate memory, but keep
                        # requires_grad=True so the next step's grad call
                        # has something to differentiate.
                        updated = updated.detach().requires_grad_(True)
                    new_params[name] = updated
            params = new_params
        return params, last_loss

    def _task_query_loss(
        self, task: MetaTask, adapted_params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the query-set loss under the adapted parameters."""
        query_x = _messages_to_sequence(task.query_set)
        query_y = _target_tensor(task.target_adaptation)
        pred = self._forward(adapted_params, query_x)
        return F.mse_loss(pred, query_y.expand_as(pred))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        """Return a serialisable state dict (encoder + head + optimiser)."""
        return {
            "encoder": self.model.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparameters": {
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
                "inner_steps": self.inner_steps,
                "first_order": self.first_order,
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
                raise KeyError(
                    f"Missing required state_dict key {key!r}."
                )
        self.model.load_state_dict(state["encoder"])  # type: ignore[arg-type]
        self.head.load_state_dict(state["head"])  # type: ignore[arg-type]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]


__all__: list[str] = ["MAMLTrainer", "MetaBatch", "MetaTask"]
