"""Reservoir-sampling replay buffer for experience replay.

EWC regularises in *parameter space*; experience replay regularises in
*data space* by mixing a small, unbiased sample of prior observations
into the current minibatch. Chaudhry et al. 2019 ("On Tiny Episodic
Memories in Continual Learning") showed that a memory of a few hundred
past samples, drawn uniformly, is already enough to materially reduce
forgetting on standard continual-learning benchmarks. The two methods
are complementary: EWC prevents drift of the protected parameters, and
replay prevents drift of the data manifold the network was optimised on.

Uniform sampling from an unbounded stream is solved classically by
Vitter 1985 ("Random sampling with a reservoir", *ACM Trans. Math.
Softw.* 11 (1)). Algorithm R maintains a reservoir of size *k* with the
invariant that after ``n`` observations each has been stored with
probability ``k/n``.

References
----------
* Vitter, J. S. (1985). "Random sampling with a reservoir". *ACM
  Trans. Math. Softw.* 11 (1): 37-57.
* Chaudhry, A., et al. (2019). "On Tiny Episodic Memories in Continual
  Learning". *arXiv 1902.10486*.
* Rolnick, D., et al. (2019). "Experience Replay for Continual
  Learning". *NeurIPS*. (Complementary; cited for context.)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay sample payload
# ---------------------------------------------------------------------------


@dataclass
class ReplaySample:
    """A single replayable experience.

    Attributes:
        input_tensor: Model input, typically a sequence of interaction
            feature vectors shaped ``[seq_len, 32]`` for the TCN.
        target: Optional supervision signal. For self-supervised setups
            this may be ``None``.
        metadata: Optional dict of auxiliary information (persona name,
            session id, etc.). Deliberately dict-of-primitives to avoid
            pulling in :mod:`pydantic` here.
    """

    input_tensor: torch.Tensor
    target: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def detach_clone(self) -> ReplaySample:
        """Return a detached, cloned copy safe for storage in the buffer."""
        return ReplaySample(
            input_tensor=self.input_tensor.detach().clone(),
            target=(
                self.target.detach().clone()
                if isinstance(self.target, torch.Tensor)
                else self.target
            ),
            metadata=dict(self.metadata),
        )


# ---------------------------------------------------------------------------
# Reservoir sampler
# ---------------------------------------------------------------------------


class ReservoirReplayBuffer:
    """Vitter 1985 reservoir sampler for uniform past-sample retention.

    The reservoir is a list of at most ``capacity`` samples. After
    observing the ``n``-th item:

    * If ``n ≤ capacity`` the item is stored directly.
    * Otherwise a random index ``j`` in ``[0, n)`` is drawn and the item
      replaces ``reservoir[j]`` iff ``j < capacity``.

    This maintains the uniform-sampling invariant ``P(item_i in reservoir
    after n observations) = capacity / n`` for all ``i < n`` without
    ever revisiting the stream.

    Args:
        capacity: Maximum number of samples retained.
        seed: Optional deterministic seed for the Python RNG used to draw
            ``j``. Pass ``None`` (default) for non-deterministic sampling.

    Example::

        buf = ReservoirReplayBuffer(capacity=512, seed=0)
        for sample in stream:
            buf.add(sample)
        past_batch = buf.sample(32)
    """

    def __init__(self, capacity: int = 512, *, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity: int = int(capacity)
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._reservoir: list[ReplaySample] = []
        self._observed: int = 0

    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of samples retained."""
        return self._capacity

    @property
    def observed(self) -> int:
        """Total number of items ever passed to :meth:`add`."""
        return self._observed

    def __len__(self) -> int:
        return len(self._reservoir)

    def __iter__(self):
        return iter(self._reservoir)

    # ------------------------------------------------------------------

    def add(self, sample: ReplaySample) -> None:
        """Offer a new sample to the reservoir.

        Preserves Vitter's invariant that every observed item is in the
        reservoir with probability ``capacity / n``.

        Args:
            sample: The experience to consider. It is detached + cloned
                so the caller may continue using the source tensors.
        """
        self._observed += 1
        stored = sample.detach_clone()
        if len(self._reservoir) < self._capacity:
            self._reservoir.append(stored)
            return
        # Vitter's Algorithm R replacement step.
        j = self._rng.randrange(self._observed)
        if j < self._capacity:
            self._reservoir[j] = stored

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        """Convenience: offer an iterable of samples."""
        for s in samples:
            self.add(s)

    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> list[ReplaySample]:
        """Return a uniform random sample of the reservoir.

        If the reservoir contains fewer than ``batch_size`` items the
        entire reservoir is returned (with replacement is intentionally
        avoided to keep gradient estimates low-variance).

        Args:
            batch_size: Desired number of samples.

        Returns:
            A list of length ``min(batch_size, len(self))``.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        if batch_size == 0 or not self._reservoir:
            return []
        if batch_size >= len(self._reservoir):
            return list(self._reservoir)
        return self._rng.sample(self._reservoir, batch_size)

    def clear(self) -> None:
        """Drop every stored sample."""
        self._reservoir.clear()
        self._observed = 0


# ---------------------------------------------------------------------------
# Experience replay wrapper
# ---------------------------------------------------------------------------


class ExperienceReplay:
    """Glue that mixes reservoir samples into new-task training batches.

    A lightweight helper around :class:`ReservoirReplayBuffer`. The
    canonical usage is::

        replay = ExperienceReplay(buffer=ReservoirReplayBuffer(512))

        def combined_loss(batch):
            new_loss = task_loss_fn(batch)
            return replay.integrate_into_training(task_loss_fn, new_loss)

    The callable returns ``new_loss + α · replay_loss`` where
    ``replay_loss`` is averaged across the replay minibatch and
    ``α ∈ [0, 1]`` (``replay_weight``) controls the mixing ratio. Setting
    ``replay_weight = 0`` disables the contribution entirely.

    Args:
        buffer: The underlying reservoir.
        replay_batch_size: Number of past samples drawn per call.
        replay_weight: Mixing weight ``α`` applied to the replay loss.
    """

    def __init__(
        self,
        buffer: ReservoirReplayBuffer,
        *,
        replay_batch_size: int = 16,
        replay_weight: float = 1.0,
    ) -> None:
        if replay_batch_size < 0:
            raise ValueError(
                f"replay_batch_size must be >= 0, got {replay_batch_size}"
            )
        if replay_weight < 0:
            raise ValueError(
                f"replay_weight must be >= 0, got {replay_weight}"
            )
        self._buffer = buffer
        self.replay_batch_size: int = int(replay_batch_size)
        self.replay_weight: float = float(replay_weight)

    # ------------------------------------------------------------------

    @property
    def buffer(self) -> ReservoirReplayBuffer:
        """The underlying reservoir buffer."""
        return self._buffer

    def observe(self, sample: ReplaySample) -> None:
        """Offer a sample to the buffer. Convenience wrapper."""
        self._buffer.add(sample)

    def integrate_into_training(
        self,
        task_loss_fn: Callable[[ReplaySample], torch.Tensor],
        task_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``task_loss + α · replay_loss`` as a single scalar.

        The ``task_loss_fn`` callable takes a :class:`ReplaySample` and
        returns a scalar tensor. It is re-used both to compute the
        replay contribution (by applying it to each sampled past item)
        and as a signature constraint on the caller.

        If ``task_loss`` is ``None``, only the replay term is returned;
        useful when the caller wants the replay contribution alone (for
        diagnostics or a two-stage training loop).

        Args:
            task_loss_fn: Scalar loss closure evaluated on past samples.
            task_loss: Optional precomputed new-task loss to add to the
                replay term.

        Returns:
            Scalar tensor. When the buffer is empty, the replay term is
            a zero tensor placed on ``task_loss``'s device (or CPU if
            ``task_loss`` is ``None``).
        """
        device = (
            task_loss.device if isinstance(task_loss, torch.Tensor) else None
        )
        past = self._buffer.sample(self.replay_batch_size)
        if not past or self.replay_weight == 0.0:
            replay_loss = torch.zeros((), device=device)
        else:
            contributions = [task_loss_fn(p) for p in past]
            stacked = torch.stack([c if c.dim() == 0 else c.mean() for c in contributions])
            replay_loss = stacked.mean()

        weighted = self.replay_weight * replay_loss
        if task_loss is None:
            return weighted
        if weighted.device != task_loss.device:
            weighted = weighted.to(task_loss.device)
        return task_loss + weighted


__all__ = [
    "ExperienceReplay",
    "ReplaySample",
    "ReservoirReplayBuffer",
]
