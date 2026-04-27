"""Adaptive Computation Time (ACT) halting controller, conditioned on
adaptation state.

This module implements a per-token halting mechanism for a stack of
transformer layers, directly inspired by Graves (2016).  Where the original
ACT paper applied dynamic depth to an RNN, we apply it to a transformer
decoder: after each layer, every token emits a halting probability; once
the cumulative probability for a token exceeds ``halt_threshold`` the
token is considered "done" and its representation is frozen for the
remaining layers.

What makes this implementation specific to I3 is that the halting head
receives the 8-dimensional :class:`AdaptationVector` as an extra input.
The ``cognitive_load`` dimension (index 0 in the vector layout) is the
lever: high cognitive load is intended to bias the head toward higher
halting probabilities, which in turn causes the model to halt earlier
and emit shorter, simpler replies.  This is the direct mechanical
embodiment of the HMI story -- "when the user is cognitively loaded,
think less, say less".

For cost reasons we do **not** implement per-token dynamic depth at the
CUDA-kernel level (that would require sparse scheduling inside the
attention kernels).  Instead the companion v2 transformer uses a
*halting mask*: layers still run over the full sequence, but halted
tokens contribute only their frozen representation, and we report a
``ponder_loss`` auxiliary scalar that encourages early halting under
high load.

References
----------
* Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural
  Networks.* arXiv:1603.08983.

Only ``torch`` is imported -- no external dependency.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ACTController(nn.Module):
    """Adaptive Computation Time halting head for the I3 transformer.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the transformer layer output ``h``.
    adaptation_dim : int, optional
        Dimensionality of the :class:`AdaptationVector` (default 8).
    ponder_cost : float, optional
        Coefficient on the ponder loss (default 0.01).  Higher values
        encourage earlier halting at the cost of expressiveness.
    halt_threshold : float, optional
        Cumulative halting probability above which a token is treated as
        halted (default 0.99).  Matches the ``1 - epsilon`` formulation
        in Graves 2016.
    max_layers : int, optional
        Maximum number of transformer layers the controller will see.
        Used to pre-allocate / document expected depth (default 12); the
        controller itself is layer-agnostic at forward time.

    Attributes
    ----------
    head : nn.Sequential
        Small MLP producing per-token halting probabilities.  Inputs are
        ``[h || a_broadcast]`` of width ``d_model + adaptation_dim``; the
        output is a single logit passed through a sigmoid.
    """

    def __init__(
        self,
        d_model: int,
        adaptation_dim: int = 8,
        ponder_cost: float = 0.01,
        halt_threshold: float = 0.99,
        max_layers: int = 12,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not 0.0 < halt_threshold <= 1.0:
            raise ValueError(
                f"halt_threshold must be in (0, 1], got {halt_threshold}"
            )
        if ponder_cost < 0:
            raise ValueError(
                f"ponder_cost must be >= 0, got {ponder_cost}"
            )
        if max_layers <= 0:
            raise ValueError(f"max_layers must be positive, got {max_layers}")

        self.d_model: int = d_model
        self.adaptation_dim: int = adaptation_dim
        self.ponder_cost: float = float(ponder_cost)
        self.halt_threshold: float = float(halt_threshold)
        self.max_layers: int = max_layers

        # Small MLP head: project [h || a_broadcast] -> scalar halting logit.
        # Keeping the head small means the controller adds negligible
        # parameter overhead on top of the transformer stack.
        hidden: int = max(d_model // 4, 32)
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(d_model + adaptation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Xavier init for stability, zero bias except the final projection
        # which we bias NEGATIVELY so that early in training the model
        # prefers NOT to halt -- this gives the rest of the stack a chance
        # to learn before halting becomes sticky.
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Bias the final logit negative (sigmoid(-2) approx 0.12) so that
        # initial p_new is small.
        last_linear: nn.Linear = self.head[-1]  # type: ignore[assignment]
        nn.init.constant_(last_linear.bias, -2.0)

        # Running state used by the ponder_loss property -- populated at
        # each forward() call and read once per sequence by the parent
        # transformer.  Not a buffer because it is not part of the model's
        # persistent state.
        self._last_p_cum: torch.Tensor | None = None
        self._last_remainder: torch.Tensor | None = None
        self._last_halt_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        p_cum: torch.Tensor,
        adaptation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the ACT state by one transformer layer.

        Parameters
        ----------
        h : torch.Tensor
            ``[batch, seq, d_model]`` -- the current layer's output.
        p_cum : torch.Tensor
            ``[batch, seq]`` -- cumulative halting probability across
            layers seen so far.  Pass zeros for the first layer.
        adaptation : torch.Tensor
            ``[batch, adaptation_dim]`` -- per-sequence adaptation vector.
            Broadcast across the sequence dimension inside this method.

        Returns
        -------
        p_cum_new : torch.Tensor
            ``[batch, seq]`` -- the updated cumulative halting probability.
        halt_mask : torch.Tensor
            ``[batch, seq]`` boolean -- ``True`` for tokens that have just
            reached / passed ``halt_threshold`` after this layer.
        remainder : torch.Tensor
            ``[batch, seq]`` -- ``1 - p_cum_pre`` for tokens in
            ``halt_mask``, zero elsewhere.  Used by the parent transformer
            to weight the final layer's contribution for halted tokens,
            exactly as in Graves 2016 section 2.
        """
        if h.dim() != 3:
            raise ValueError(
                f"h must be 3D [batch, seq, d_model], got {tuple(h.shape)}"
            )
        if h.size(-1) != self.d_model:
            raise ValueError(
                f"h.shape[-1]={h.size(-1)} != d_model={self.d_model}"
            )
        if p_cum.shape != h.shape[:2]:
            raise ValueError(
                f"p_cum shape {tuple(p_cum.shape)} != h[:,:, 0] shape "
                f"{tuple(h.shape[:2])}"
            )
        if adaptation.dim() != 2 or adaptation.size(-1) != self.adaptation_dim:
            raise ValueError(
                f"adaptation must be [batch, {self.adaptation_dim}], got "
                f"{tuple(adaptation.shape)}"
            )

        batch, seq_len, _ = h.shape

        # Broadcast adaptation across the sequence: [B, A] -> [B, S, A].
        a_broadcast: torch.Tensor = adaptation.unsqueeze(1).expand(
            batch, seq_len, self.adaptation_dim
        )

        # Compute per-token halting probability p_new in [0, 1].
        head_in: torch.Tensor = torch.cat([h, a_broadcast], dim=-1)
        logits: torch.Tensor = self.head(head_in).squeeze(-1)  # [B, S]
        p_new: torch.Tensor = torch.sigmoid(logits)

        # Tokens that *were not yet halted* before this layer: mask out
        # already-halted tokens so their p_cum does not grow past
        # threshold.  Halted tokens contribute p_new = 0.
        not_halted_pre: torch.Tensor = (p_cum < self.halt_threshold).float()
        p_new_effective: torch.Tensor = p_new * not_halted_pre

        p_cum_new: torch.Tensor = p_cum + p_new_effective

        # Tokens that have *just* halted this step: were not-halted before,
        # are halted now.  Remainder = 1 - p_cum_pre (i.e. the mass that
        # this final layer had to absorb).
        halt_mask: torch.Tensor = (p_cum_new >= self.halt_threshold) & (
            p_cum < self.halt_threshold
        )
        remainder: torch.Tensor = torch.where(
            halt_mask,
            1.0 - p_cum,
            torch.zeros_like(p_cum),
        )

        # Stash for ponder_loss readers.
        self._last_p_cum = p_cum_new.detach()
        self._last_halt_mask = halt_mask.detach()
        self._last_remainder = remainder.detach()

        # Store grad-aware copies for the loss itself.
        self._ponder_p_cum = p_cum_new
        self._ponder_remainder = remainder
        self._ponder_halt_mask = halt_mask

        return p_cum_new, halt_mask, remainder

    # ------------------------------------------------------------------
    # ponder loss
    # ------------------------------------------------------------------

    @property
    def ponder_loss(self) -> torch.Tensor:
        """Auxiliary ponder loss encouraging early halting.

        Implements the standard ACT formulation:

        .. math::

            \\mathcal{L}_{\\text{ponder}} =
                \\text{ponder\\_cost} \\cdot
                \\operatorname{mean}(p_{cum} + R \\cdot M_{halt})

        where ``R`` is the remainder and ``M_halt`` masks tokens that
        halted on the current layer.  Minimising it shrinks the number
        of layers each token needs.  With the adaptation-conditioned
        head this becomes load-sensitive: under high ``cognitive_load``
        the head fires earlier, so ``p_cum`` climbs faster toward 1
        and the penalty is smaller with fewer layers -- exactly the
        behaviour we want.

        Returns
        -------
        torch.Tensor
            Scalar tensor.  If :meth:`forward` has not yet been called
            this call raises ``RuntimeError``.
        """
        if (
            not hasattr(self, "_ponder_p_cum")
            or self._ponder_p_cum is None
        ):
            raise RuntimeError(
                "ponder_loss requested before ACTController.forward was called"
            )
        return self.ponder_cost * (
            self._ponder_p_cum + self._ponder_remainder * self._ponder_halt_mask.float()
        ).mean()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear cached state from the last forward call.

        Call between sequences to avoid holding stale tensors on the GPU.
        """
        self._last_p_cum = None
        self._last_remainder = None
        self._last_halt_mask = None
        self._ponder_p_cum = None  # type: ignore[assignment]
        self._ponder_remainder = None  # type: ignore[assignment]
        self._ponder_halt_mask = None  # type: ignore[assignment]
