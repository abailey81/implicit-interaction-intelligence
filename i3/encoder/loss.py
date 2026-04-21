"""NT-Xent (SimCLR) contrastive loss — standalone module.

This module extracts the contrastive objective originally inlined in
:mod:`i3.encoder.train` into a dedicated, reusable module. The behaviour is
faithful to the original implementation: a labelled variant of NT-Xent where
every sample sharing an anchor's label is treated as a positive.

The ``NTXentLoss`` ``nn.Module`` mirrors the more common two-view SimCLR
formulation (``z_a, z_b`` drawn from augmented views), while the free function
``nt_xent_loss`` provides the pure-function API. Both are paper-faithful.

Reference
---------
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
*A Simple Framework for Contrastive Learning of Visual Representations.*
arXiv:2002.05709, Eq. 1.

Example
-------
    >>> import torch
    >>> from i3.encoder.loss import NTXentLoss
    >>> loss_fn = NTXentLoss(temperature=0.1)
    >>> z_a = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
    >>> z_b = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
    >>> loss = loss_fn(z_a, z_b)
    >>> loss.backward()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["NTXentLoss", "nt_xent_loss"]


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    Implements the two-view SimCLR objective. Given two batches of embeddings
    ``z_a`` and ``z_b`` of shape ``[B, D]`` (assumed L2-normalised upstream),
    the module builds the ``[2B, 2B]`` cosine-similarity matrix, masks the
    diagonal, and computes cross-entropy against the positive partner
    (``z_a[i] <-> z_b[i]``).

    Reference:
        Chen et al. 2020 (SimCLR), arXiv:2002.05709, Eq. 1.

    Attributes:
        temperature: Softmax temperature. Lower = sharper similarity
            distribution. SimCLR default is ``0.5``; the I3 encoder uses
            ``0.1`` for tighter clusters on short sequences.
        reduction: One of ``"mean"``, ``"sum"``, or ``"none"``. Passed through
            to :func:`torch.nn.functional.cross_entropy`.
    """

    temperature: float
    reduction: str

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        """Initialise the NT-Xent loss module.

        Args:
            temperature: Softmax temperature for the similarity logits. Must
                be strictly positive.
            reduction: Reduction mode for the per-anchor cross-entropy losses.
                Accepts ``"mean"``, ``"sum"``, or ``"none"``.

        Raises:
            ValueError: If ``temperature`` is not strictly positive or
                ``reduction`` is not one of the accepted modes.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(
                f"NT-Xent temperature must be positive, got {temperature!r}."
            )
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}."
            )
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Compute the symmetric NT-Xent loss.

        Args:
            z_a: First view of the batch, shape ``[B, D]``, L2-normalised.
            z_b: Second view of the batch, shape ``[B, D]``, L2-normalised.

        Returns:
            Scalar loss tensor when ``reduction != "none"``; otherwise a
            ``[2B]`` tensor of per-anchor losses.

        Raises:
            ValueError: If ``z_a`` and ``z_b`` differ in shape or are not 2-D.
        """
        if z_a.dim() != 2 or z_b.dim() != 2:
            raise ValueError(
                f"NT-Xent expects 2-D tensors; got z_a.dim={z_a.dim()}, "
                f"z_b.dim={z_b.dim()}."
            )
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"NT-Xent expects z_a.shape == z_b.shape; got {tuple(z_a.shape)} "
                f"vs {tuple(z_b.shape)}."
            )

        batch_size = z_a.size(0)
        device = z_a.device

        # Concatenate views -> [2B, D]
        z = torch.cat([z_a, z_b], dim=0)

        # [2B, 2B] cosine similarity (embeddings assumed L2-normed upstream)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask the diagonal with a large negative value (not -inf, to keep
        # fp16 stable per §18.2 Day 6 numerical-stability rule).
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, -1e9)

        # For each anchor i in [0, 2B), its positive partner is i + B (mod 2B).
        targets = torch.arange(2 * batch_size, device=device)
        targets = (targets + batch_size) % (2 * batch_size)

        return F.cross_entropy(sim, targets, reduction=self.reduction)

    def extra_repr(self) -> str:
        """Pretty representation of the module's hyperparameters."""
        return f"temperature={self.temperature}, reduction={self.reduction!r}"


def nt_xent_loss(
    z_a: Tensor,
    z_b: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Functional form of :class:`NTXentLoss` with ``reduction="mean"``.

    Args:
        z_a: First view, shape ``[B, D]``, L2-normalised.
        z_b: Second view, shape ``[B, D]``, L2-normalised.
        temperature: Softmax temperature. Must be strictly positive.

    Returns:
        Scalar loss tensor (mean over the ``2B`` anchors).

    Reference:
        Chen et al. 2020 (SimCLR), arXiv:2002.05709, Eq. 1.
    """
    return NTXentLoss(temperature=temperature, reduction="mean")(z_a, z_b)
