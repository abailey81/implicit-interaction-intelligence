"""Elastic Weight Consolidation for the TCN encoder.

Implements the diagonal Fisher Information regulariser from Kirkpatrick
et al. 2017 PNAS ("Overcoming catastrophic forgetting in neural
networks") plus the running-Fisher variant from Schwarz et al. 2018
("Progress & Compress"). These primitives wrap an arbitrary
:class:`torch.nn.Module` such as :class:`~i3.encoder.tcn.TemporalConvNet`
and add a quadratic penalty ``(λ/2) · Σ_i F_i · (θ_i − θ*_i)²`` to the
training loss of subsequent tasks, where:

* ``F_i`` is the diagonal Fisher Information estimated on the completed
  task (``≈ E[(∂ log p(y|x;θ) / ∂ θ_i)²]``),
* ``θ*_i`` is the snapshot of the parameter value at the end of that
  task, and
* ``λ`` is a sensitivity constant (Kirkpatrick et al. 2017 used ``400``
  for MNIST-permutations and ``1000-5000`` for Atari).

References
----------
* Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in
  neural networks". *PNAS*, 114(13), 3521-3526.
* Schwarz, J., et al. (2018). "Progress & Compress: A scalable framework
  for continual learning". *ICML*.
* Aljundi, R., et al. (2018). "Memory Aware Synapses: Learning what (not)
  to forget". *ECCV*. (Complementary importance estimator; cited for
  future work.)
* arXiv 2511.20732 (2025). PA-EWC: "Parameter-Adaptive Elastic Weight
  Consolidation". (Adaptive λ -- future work.)

The module deliberately does NOT modify the wrapped model's forward or
training loop; instead it exposes :meth:`ElasticWeightConsolidation.
penalty_loss` which the caller adds to their task loss.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FisherDict = dict[str, torch.Tensor]
"""Diagonal Fisher Information Matrix keyed by parameter name."""

ParamDict = dict[str, torch.Tensor]
"""Snapshot of parameter values keyed by parameter name."""

LossClosure = Callable[[nn.Module, torch.Tensor], torch.Tensor]
"""Callable ``(model, batch) -> scalar loss`` used by the default Fisher
estimator. ``batch`` is whatever the dataloader yields."""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# SEC: Stability epsilon added to the diagonal Fisher to prevent the
# penalty from collapsing to zero on parameters the task never activated.
# Schwarz et al. (2018) use ``1e-3``; we expose it as a constructor arg.
_DEFAULT_FISHER_EPSILON: float = 1e-8

# SEC: Kirkpatrick 2017 used ``400`` for MNIST-permutations and much
# higher values (``5000``) for Atari. We default to ``1000`` (Schwarz
# 2018 default) and let callers override per-task.
_DEFAULT_LAMBDA_EWC: float = 1000.0

# SEC: Default number of minibatch samples used to estimate the Fisher
# diagonal. Kirkpatrick 2017 used the full dataset once. In streaming /
# on-device settings 200-400 samples is empirically sufficient.
_DEFAULT_FISHER_SAMPLES: int = 200

# SEC: Online EWC exponential decay (Schwarz 2018 equation 9). ``γ=1.0``
# recovers plain EWC with additive Fisher accumulation; ``γ<1`` forgets
# very old tasks and matches the Progress & Compress behaviour.
_DEFAULT_ONLINE_GAMMA: float = 0.95


# ---------------------------------------------------------------------------
# Default loss closure
# ---------------------------------------------------------------------------


def _default_loss_closure(model: nn.Module, batch: object) -> torch.Tensor:
    """Default loss closure used when the caller does not supply one.

    Treats the dataloader batch as a ``(input, target)`` tuple or dict
    with keys ``"input"`` and ``"target"``. The Fisher is estimated by
    (a) running the model on ``input``, (b) treating the output as a
    log-likelihood proxy via an L2 loss to ``target``.

    For contrastive / self-supervised encoders the caller should
    pass a custom :class:`LossClosure` that computes the real loss.

    Args:
        model: The wrapped module.
        batch: A tuple or mapping produced by the dataloader.

    Returns:
        A scalar tensor suitable for ``torch.autograd.grad``.

    Raises:
        TypeError: If ``batch`` is not a recognised structure.
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inp, target = batch[0], batch[1]
    elif isinstance(batch, dict) and "input" in batch and "target" in batch:
        inp, target = batch["input"], batch["target"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 1:
        # Self-supervised: use the encoder output as its own log-likelihood
        inp = batch[0]
        output = model(inp)
        return output.pow(2).sum() / max(1, output.shape[0])
    elif isinstance(batch, torch.Tensor):
        output = model(batch)
        return output.pow(2).sum() / max(1, output.shape[0])
    else:
        raise TypeError(
            "Default loss closure expects (input, target), "
            "{'input': ..., 'target': ...}, (input,), or a tensor. "
            f"Got {type(batch).__name__}."
        )

    output = model(inp)
    if output.shape != target.shape:
        # Average-pool output in case of encoder shape mismatch.
        target = target.reshape(output.shape)
    return torch.nn.functional.mse_loss(output, target)


# ---------------------------------------------------------------------------
# ElasticWeightConsolidation
# ---------------------------------------------------------------------------


class ElasticWeightConsolidation:
    """Diagonal-Fisher EWC regulariser around an arbitrary module.

    Wraps a :class:`torch.nn.Module` and exposes three pieces of lifecycle
    API suitable for sequential-task training:

    1. :meth:`estimate_fisher` -- compute the diagonal FIM on a dataset
       without mutating any state.
    2. :meth:`consolidate` -- finalise a task: store Fisher and a
       parameter snapshot so the next task's loss can include the
       quadratic penalty.
    3. :meth:`penalty_loss` -- scalar tensor to add to the training loss.

    The module deliberately does NOT override the model's
    :meth:`~torch.nn.Module.forward`; the caller's training loop remains
    in charge.

    Example::

        encoder = TemporalConvNet()
        ewc = ElasticWeightConsolidation(encoder, lambda_ewc=1000.0)

        # Task A
        train(encoder, loader_a)
        ewc.consolidate(loader_a)

        # Task B, with EWC penalty
        for batch in loader_b:
            loss = task_loss(encoder, batch) + ewc.penalty_loss()
            loss.backward()
            optimizer.step()

    Args:
        model: The module whose parameters are protected by EWC. Only
            parameters with ``requires_grad=True`` are tracked.
        lambda_ewc: Penalty strength ``λ``. Scales quadratically with
            parameter shift, so ``1000`` is typical and values above
            ``1e5`` effectively freeze the protected parameters.
        fisher_estimation_samples: Number of minibatches drawn from the
            dataloader when estimating the diagonal Fisher. The full
            epoch is used if the loader exhausts first.
        fisher_epsilon: Stability epsilon added to every Fisher entry so
            parameters never dropped out of the penalty entirely
            (Schwarz 2018 §3.1).
        loss_closure: Optional custom ``(model, batch) -> loss`` callable
            used during Fisher estimation. When ``None`` the default
            MSE-based closure is applied.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = _DEFAULT_LAMBDA_EWC,
        fisher_estimation_samples: int = _DEFAULT_FISHER_SAMPLES,
        *,
        fisher_epsilon: float = _DEFAULT_FISHER_EPSILON,
        loss_closure: LossClosure | None = None,
    ) -> None:
        if lambda_ewc < 0:
            raise ValueError(
                f"lambda_ewc must be >= 0, got {lambda_ewc}"
            )
        if fisher_estimation_samples < 1:
            raise ValueError(
                "fisher_estimation_samples must be >= 1, "
                f"got {fisher_estimation_samples}"
            )
        if fisher_epsilon < 0:
            raise ValueError(
                f"fisher_epsilon must be >= 0, got {fisher_epsilon}"
            )

        self._model = model
        self.lambda_ewc: float = float(lambda_ewc)
        self.fisher_estimation_samples: int = int(fisher_estimation_samples)
        self.fisher_epsilon: float = float(fisher_epsilon)
        self._loss_closure: LossClosure = (
            loss_closure if loss_closure is not None else _default_loss_closure
        )

        # SEC: Fisher and parameter snapshots are keyed by parameter name
        # so they survive checkpoint round-trips and can be persisted.
        self._fisher: FisherDict = {}
        self._star_params: ParamDict = {}
        # Number of consolidated tasks; used for logging + reset tracking.
        self._num_tasks: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """Return the wrapped module (read-only)."""
        return self._model

    @property
    def num_tasks_consolidated(self) -> int:
        """Number of tasks already consolidated."""
        return self._num_tasks

    @property
    def fisher(self) -> FisherDict:
        """Current diagonal Fisher Information Matrix (read-only view)."""
        return dict(self._fisher)

    @property
    def star_params(self) -> ParamDict:
        """Parameter snapshot from the last :meth:`consolidate` call."""
        return dict(self._star_params)

    # ------------------------------------------------------------------
    # Fisher estimation
    # ------------------------------------------------------------------

    def estimate_fisher(
        self,
        dataloader: Iterable[object],
        *,
        device: torch.device | None = None,
    ) -> FisherDict:
        """Estimate the diagonal Fisher Information on a dataset.

        Uses the empirical Fisher (Amari 1998; Kunstner 2019) which, for a
        log-likelihood ``log p(y|x; θ)``, is
        ``F_ii ≈ E[(∂ log p / ∂ θ_i)²]``. The expectation is taken over
        samples from ``dataloader``; we draw at most
        ``fisher_estimation_samples`` minibatches.

        The routine is non-mutating: it restores the model's ``training``
        flag and never updates gradients on the optimiser. It reuses the
        most recent estimate if ``dataloader`` yields no samples, which
        matches the "cache is re-used across tasks" contract.

        Args:
            dataloader: Iterable producing batches consumable by the
                configured loss closure.
            device: Optional explicit device for the returned tensors.
                When ``None`` each Fisher entry is placed on the same
                device as the parameter it corresponds to.

        Returns:
            A fresh :data:`FisherDict` (not aliased to the internal
            cache). All entries are non-negative.
        """
        was_training = self._model.training
        self._model.eval()

        # Initialise accumulators on the parameter's device.
        fisher: FisherDict = {}
        for name, param in self._iter_named_parameters():
            fisher[name] = torch.zeros_like(param, device=param.device)

        sample_count = 0
        try:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.fisher_estimation_samples:
                    break

                batch_on_device = self._move_to_device(batch, device)

                # SEC: Zero-grad every loop so gradients do not accumulate
                # across batches and inflate the Fisher.
                self._model.zero_grad(set_to_none=True)
                loss = self._loss_closure(self._model, batch_on_device)
                # NOTE: we allow scalar or per-sample losses; reduce to
                # scalar if necessary.
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()

                with torch.no_grad():
                    for name, param in self._iter_named_parameters():
                        if param.grad is None:
                            continue
                        # Empirical Fisher: E[(∂L/∂θ)²]. We square then
                        # running-average across batches.
                        fisher[name] += param.grad.detach() ** 2
                sample_count += 1
        finally:
            self._model.zero_grad(set_to_none=True)
            if was_training:
                self._model.train()

        if sample_count == 0:
            # SEC: Empty dataloader -- re-use prior estimate if available,
            # else return the zero-initialised dict unchanged (the caller
            # gets the diagnostic empty tensors).
            logger.warning(
                "estimate_fisher received an empty dataloader; "
                "returning prior estimate if available."
            )
            if self._fisher:
                return {k: v.detach().clone() for k, v in self._fisher.items()}
            return fisher

        # Normalise by the number of sampled batches.
        for name in fisher:
            fisher[name] = fisher[name] / float(sample_count)
            # SEC: Add stability epsilon so parameters the task barely
            # activated still receive a non-zero penalty (Schwarz 2018).
            fisher[name] = fisher[name] + self.fisher_epsilon

        if device is not None:
            fisher = {k: v.to(device) for k, v in fisher.items()}
        return fisher

    # ------------------------------------------------------------------
    # Consolidate / penalty / reset
    # ------------------------------------------------------------------

    def consolidate(
        self,
        dataloader: Iterable[object],
        *,
        device: torch.device | None = None,
    ) -> None:
        """Finalise a task: snapshot params and store Fisher.

        After this call, :meth:`penalty_loss` returns a non-zero scalar
        tensor when subsequent training drifts any parameter away from
        its snapshot.

        Args:
            dataloader: Dataset used to estimate the FIM diagonal for the
                just-completed task.
            device: Optional device for the cached tensors.
        """
        new_fisher = self.estimate_fisher(dataloader, device=device)

        # SEC: Detach and clone every tensor so subsequent training steps
        # cannot mutate the snapshot through aliasing. Tensors are stored
        # on the parameter's native device to avoid copy costs at
        # penalty-evaluation time.
        snapshot: ParamDict = {}
        for name, param in self._iter_named_parameters():
            snapshot[name] = param.detach().clone()

        self._fisher = {k: v.detach().clone() for k, v in new_fisher.items()}
        self._star_params = snapshot
        self._num_tasks += 1
        logger.info(
            "EWC consolidate: task=%d, params=%d, λ=%.2f",
            self._num_tasks,
            len(self._star_params),
            self.lambda_ewc,
        )

    def penalty_loss(self) -> torch.Tensor:
        """Scalar EWC penalty to add to the training loss.

        Formula (Kirkpatrick 2017 eq. 3)::

            L_EWC = (λ / 2) · Σ_i F_i · (θ_i − θ*_i)²

        When no task has been consolidated yet the penalty is zero (with
        gradients). The returned tensor lives on the same device as the
        first tracked parameter.

        Returns:
            Scalar tensor; ``torch.tensor(0.0)`` when no Fisher is
            cached.
        """
        if not self._fisher or not self._star_params:
            # SEC: return a zero scalar on the first tracked parameter's
            # device so backward() does not error on device mismatch.
            for _, param in self._iter_named_parameters():
                return torch.zeros((), device=param.device)
            return torch.zeros(())

        total: torch.Tensor | None = None
        for name, param in self._iter_named_parameters():
            if name not in self._fisher or name not in self._star_params:
                continue
            fisher = self._fisher[name]
            star = self._star_params[name]
            # SEC: cast Fisher/star to the parameter's device lazily so
            # the object survives moves of the underlying model between
            # devices.
            if fisher.device != param.device:
                fisher = fisher.to(param.device)
                self._fisher[name] = fisher
            if star.device != param.device:
                star = star.to(param.device)
                self._star_params[name] = star
            # SEC: (λ/2) · F · (θ − θ*)² summed across parameters.
            contribution = (fisher * (param - star).pow(2)).sum()
            total = contribution if total is None else total + contribution

        if total is None:
            for _, param in self._iter_named_parameters():
                return torch.zeros((), device=param.device)
            return torch.zeros(())
        return 0.5 * self.lambda_ewc * total

    def reset(self) -> None:
        """Drop all Fisher + parameter snapshots.

        Equivalent to a fresh instance. Intended for tests and for the
        rare case where a catastrophic reset is explicitly desired
        (e.g. user opts out of a fatigue baseline).
        """
        self._fisher = {}
        self._star_params = {}
        self._num_tasks = 0

    def state_dict(self) -> dict[str, FisherDict | ParamDict | int | float]:
        """Return a serialisable snapshot of Fisher + star params.

        Useful for persisting EWC state alongside a model checkpoint.
        """
        return {
            "fisher": {k: v.detach().clone() for k, v in self._fisher.items()},
            "star_params": {
                k: v.detach().clone() for k, v in self._star_params.items()
            },
            "num_tasks": self._num_tasks,
            "lambda_ewc": self.lambda_ewc,
        }

    def load_state_dict(
        self, state: dict[str, FisherDict | ParamDict | int | float]
    ) -> None:
        """Restore Fisher + star params from a :meth:`state_dict` payload."""
        fisher = state.get("fisher", {})
        stars = state.get("star_params", {})
        if not isinstance(fisher, dict) or not isinstance(stars, dict):
            raise TypeError("state_dict payload malformed")
        self._fisher = {k: v.detach().clone() for k, v in fisher.items()}
        self._star_params = {k: v.detach().clone() for k, v in stars.items()}
        num = state.get("num_tasks", 0)
        self._num_tasks = int(num) if isinstance(num, (int, float)) else 0
        lam = state.get("lambda_ewc", self.lambda_ewc)
        self.lambda_ewc = float(lam) if isinstance(lam, (int, float)) else self.lambda_ewc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        """Yield ``(name, param)`` for trainable parameters only."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                yield name, param

    @staticmethod
    def _move_to_device(
        batch: object, device: torch.device | None
    ) -> object:
        """Move tensor fields of *batch* to *device* without mutation."""
        if device is None:
            return batch
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            moved = [
                b.to(device) if isinstance(b, torch.Tensor) else b
                for b in batch
            ]
            return type(batch)(moved)
        if isinstance(batch, dict):
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        return batch


# ---------------------------------------------------------------------------
# Online EWC (Schwarz et al. 2018)
# ---------------------------------------------------------------------------


class OnlineEWC(ElasticWeightConsolidation):
    """Online EWC with a running Fisher estimate (Schwarz et al. 2018).

    In streaming or unbounded-task settings the vanilla EWC requires a
    discrete task boundary at which to call :meth:`consolidate`. Online
    EWC instead maintains a single running Fisher ``F̃`` that is updated
    with exponential decay::

        F̃_t = γ · F̃_{t-1} + F_t,   θ̃*_t = current params

    where ``γ ∈ (0, 1]`` controls how aggressively the old Fisher is
    forgotten. Setting ``γ=1`` recovers additive accumulation (closer to
    plain EWC).

    Args:
        model: The wrapped module.
        lambda_ewc: EWC penalty strength (see
            :class:`ElasticWeightConsolidation`).
        fisher_estimation_samples: Fisher estimation budget per update.
        gamma: Decay factor ``γ`` in ``(0, 1]``.
        fisher_epsilon: Stability epsilon.
        loss_closure: Optional custom loss closure.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = _DEFAULT_LAMBDA_EWC,
        fisher_estimation_samples: int = _DEFAULT_FISHER_SAMPLES,
        *,
        gamma: float = _DEFAULT_ONLINE_GAMMA,
        fisher_epsilon: float = _DEFAULT_FISHER_EPSILON,
        loss_closure: LossClosure | None = None,
    ) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        super().__init__(
            model,
            lambda_ewc=lambda_ewc,
            fisher_estimation_samples=fisher_estimation_samples,
            fisher_epsilon=fisher_epsilon,
            loss_closure=loss_closure,
        )
        self.gamma: float = float(gamma)

    # ------------------------------------------------------------------

    def consolidate(
        self,
        dataloader: Iterable[object],
        *,
        device: torch.device | None = None,
    ) -> None:
        """Update running Fisher + star params using exponential decay.

        The first call seeds the running Fisher directly; subsequent
        calls perform ``F̃ ← γ · F̃ + F_t``.
        """
        new_fisher = self.estimate_fisher(dataloader, device=device)

        if not self._fisher:
            # First consolidation -- seed directly.
            self._fisher = {k: v.detach().clone() for k, v in new_fisher.items()}
        else:
            merged: FisherDict = {}
            for name, new_val in new_fisher.items():
                prev = self._fisher.get(name)
                if prev is None or prev.shape != new_val.shape:
                    merged[name] = new_val.detach().clone()
                else:
                    if prev.device != new_val.device:
                        prev = prev.to(new_val.device)
                    merged[name] = self.gamma * prev + new_val
            self._fisher = merged

        # SEC: Star params always refresh to the current parameters --
        # Online EWC's protection reference is the "present" (Schwarz
        # 2018 §3.2), not the start of a task.
        snapshot: ParamDict = {}
        for name, param in self._iter_named_parameters():
            snapshot[name] = param.detach().clone()
        self._star_params = snapshot
        self._num_tasks += 1

        logger.info(
            "OnlineEWC consolidate: step=%d, γ=%.3f, λ=%.2f, params=%d",
            self._num_tasks,
            self.gamma,
            self.lambda_ewc,
            len(self._star_params),
        )


# ---------------------------------------------------------------------------
# Convenience: default-ctor helper that wires a DataLoader directly
# ---------------------------------------------------------------------------


def build_ewc_for_encoder(
    encoder: nn.Module,
    *,
    lambda_ewc: float = _DEFAULT_LAMBDA_EWC,
    online: bool = False,
    gamma: float = _DEFAULT_ONLINE_GAMMA,
    fisher_estimation_samples: int = _DEFAULT_FISHER_SAMPLES,
    loss_closure: LossClosure | None = None,
) -> ElasticWeightConsolidation:
    """Factory that returns either :class:`ElasticWeightConsolidation`
    or :class:`OnlineEWC` depending on ``online``.

    This keeps training scripts simple when the continual-learning mode
    is configured externally (YAML, CLI flag, ...)

    Args:
        encoder: Module to wrap.
        lambda_ewc: Penalty strength.
        online: Use :class:`OnlineEWC` when ``True``.
        gamma: Online EWC decay factor (ignored when ``online=False``).
        fisher_estimation_samples: Fisher sample budget.
        loss_closure: Optional custom loss closure.

    Returns:
        A concrete EWC instance.
    """
    if online:
        return OnlineEWC(
            encoder,
            lambda_ewc=lambda_ewc,
            gamma=gamma,
            fisher_estimation_samples=fisher_estimation_samples,
            loss_closure=loss_closure,
        )
    return ElasticWeightConsolidation(
        encoder,
        lambda_ewc=lambda_ewc,
        fisher_estimation_samples=fisher_estimation_samples,
        loss_closure=loss_closure,
    )


__all__ = [
    "ElasticWeightConsolidation",
    "FisherDict",
    "LossClosure",
    "OnlineEWC",
    "ParamDict",
    "build_ewc_for_encoder",
]
