"""Mixture-of-Experts feed-forward layer gated by the adaptation vector.

This module implements a *conditional* feed-forward sub-layer for the I3
Adaptive Small Language Model.  Instead of a single position-wise MLP, we
maintain a small pool of independent "expert" MLPs and route every token
through a **soft mixture** of them.  The mixing weights are produced by a
lightweight gating network that takes the 8-dimensional :class:`AdaptationVector`
as input, so the user's observed cognitive / affective state directly
selects which experts contribute to the response at every layer.

Why MoE here (Huawei HMI rationale)
-----------------------------------
The pitch of I3 is that *implicit* behavioural signals should reshape the
response generation path -- not merely bias token sampling at the end.  MoE
gives us a clean, interpretable mechanism for this: each expert can learn to
specialise on a communication regime (e.g. "low-load / reassuring",
"high-load / technical", "neutral / concise", "elevated-arousal / gentle"),
and the gate reads the adaptation vector to blend them.  The resulting
``last_gate_weights`` tensor is directly useful to the demo UI -- it shows
which "mode" of the model is firing on any given turn.

Design notes
------------
* **Whole-sequence gating**: the adaptation vector is a *user-level* signal,
  not per-token, so the gate output is ``(batch, n_experts)`` and is applied
  identically across the sequence dimension.  This mirrors the "task-level"
  routing variant common in multi-task MoE rather than the per-token
  top-k routing used in Switch Transformers -- and is the right choice here
  because our conditioning changes on the order of seconds (typing-speed
  windows), not tokens.
* **Soft mixture (no top-k)**: with only a handful of experts and a stable
  gating signal, hard top-k routing is unnecessary and would hurt gradient
  flow through the gate.  We use a plain softmax and take a weighted sum.
* **Load-balancing loss**: even with a soft gate, we want to prevent
  collapse onto a single expert.  Following Shazeer et al. 2017 we expose
  ``load_balance_loss`` returning ``n_experts * mean(gate_weights ** 2)``,
  which is minimised when the gate output is uniform and grows as the
  distribution sharpens.  The training loop adds this as an auxiliary loss
  with a small coefficient (e.g. 0.01).

References
----------
* Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.
* Fedus, W., Zoph, B., Shazeer, N. (2022). *Switch Transformers: Scaling to
  Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.

No external dependencies beyond ``torch`` -- this is a from-scratch
implementation consistent with the Huawei R&D UK portfolio requirement.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _Expert(nn.Module):
    """A single expert: a two-layer position-wise MLP with GELU + dropout.

    Identical structure to the standard :class:`~i3.slm.attention.FeedForward`
    used in v1, but each expert is an independent ``nn.Module`` with its own
    parameters so that the gate can route between them.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: Literal["gelu", "relu"],
    ) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(d_model, d_ff)
        self.fc2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.act: nn.Module = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:  # pragma: no cover - guarded at parent level
            raise ValueError(f"Unsupported activation: {activation!r}")

        # Xavier uniform for the MLP weights, zero for biases: a sensible
        # default for a freshly-initialised from-scratch expert.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single expert MLP over ``x``.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq, d_model]``.

        Returns
        -------
        torch.Tensor
            ``[batch, seq, d_model]`` -- same shape as input.
        """
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward sub-layer.

    Replaces the standard position-wise FFN with ``n_experts`` parallel MLPs.
    A gating network consumes the :class:`AdaptationVector` and emits one
    softmax distribution per *sequence* (not per token), selecting how much
    each expert contributes.

    Parameters
    ----------
    d_model : int
        Model / hidden dimension -- same as the surrounding transformer.
    d_ff : int
        Inner (expanded) dimension of each expert (typically ``4 * d_model``).
    n_experts : int, optional
        Number of parallel experts (default 4).  Four experts map neatly
        onto the four adaptation quadrants (low/high cognitive load x
        low/high emotional arousal).
    adaptation_dim : int, optional
        Dimensionality of the adaptation vector (default 8).
    dropout : float, optional
        Dropout inside each expert's MLP (default 0.1).
    activation : str, optional
        Activation inside the expert MLP -- ``"gelu"`` (default) or
        ``"relu"``.

    Attributes
    ----------
    experts : nn.ModuleList[_Expert]
        The pool of ``n_experts`` independent MLPs.
    gate : nn.Linear
        The gating network: ``Linear(adaptation_dim, n_experts)``.
    last_gate_weights : torch.Tensor | None
        The most recent gate softmax (``[batch, n_experts]``) after a
        ``forward`` call -- useful for UI visualisation / debugging.

    Notes
    -----
    **Load-balancing auxiliary loss.**  Call
    :meth:`load_balance_loss(last_gate_weights)` in the training step and
    add it to the main cross-entropy with a small coefficient (0.01 is a
    reasonable starting point).  This encourages the gate distribution to
    stay close to uniform and prevents expert collapse.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        adaptation_dim: int = 8,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        if d_model <= 0 or d_ff <= 0:
            raise ValueError(
                f"d_model and d_ff must be positive, got {d_model} and {d_ff}"
            )
        if activation not in ("gelu", "relu"):
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {activation!r}"
            )

        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_experts: int = n_experts
        self.adaptation_dim: int = adaptation_dim

        self.experts: nn.ModuleList = nn.ModuleList(
            [
                _Expert(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_experts)
            ]
        )

        # Gating network: a single linear projection from the adaptation
        # vector to per-expert logits.  We deliberately keep this tiny --
        # the gate is not meant to be expressive; it just reads the
        # (already interpretable) adaptation vector and picks the mixing
        # weights.
        self.gate: nn.Linear = nn.Linear(adaptation_dim, n_experts)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Populated during forward; saved for introspection / UI.
        self.last_gate_weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        adaptation: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens through a soft mixture of experts.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq, d_model]`` -- the residual-stream input.
        adaptation : torch.Tensor
            ``[batch, adaptation_dim]`` -- the per-sequence
            :class:`AdaptationVector`.

        Returns
        -------
        torch.Tensor
            ``[batch, seq, d_model]`` -- mixture-of-experts output, same
            shape as ``x``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"MoEFeedForward expects x of shape [batch, seq, d_model], "
                f"got {tuple(x.shape)}"
            )
        if adaptation.dim() != 2:
            raise ValueError(
                f"adaptation must be 2D [batch, adaptation_dim], got "
                f"{tuple(adaptation.shape)}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"x.shape[-1]={x.size(-1)} != d_model={self.d_model}"
            )
        if adaptation.size(-1) != self.adaptation_dim:
            raise ValueError(
                f"adaptation.shape[-1]={adaptation.size(-1)} != "
                f"adaptation_dim={self.adaptation_dim}"
            )
        if x.size(0) != adaptation.size(0):
            raise ValueError(
                f"batch mismatch: x={x.size(0)}, adaptation={adaptation.size(0)}"
            )

        batch_size, seq_len, _ = x.shape

        # Gate softmax: [batch, n_experts]
        gate_logits: torch.Tensor = self.gate(adaptation)
        gate_weights: torch.Tensor = F.softmax(gate_logits, dim=-1)
        self.last_gate_weights = gate_weights.detach()

        # Run every expert on the full sequence.  Stacking along a new
        # dim (dim=2) yields [batch, seq, n_experts, d_model].
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )  # [B, S, E, D]

        # Broadcast gate weights over seq and d_model dims:
        #   gate_weights: [B, E] -> [B, 1, E, 1]
        gate_broadcast: torch.Tensor = gate_weights.unsqueeze(1).unsqueeze(-1)

        # Weighted sum across experts -> [B, S, D]
        mixed: torch.Tensor = (expert_outputs * gate_broadcast).sum(dim=2)

        return mixed

    # ------------------------------------------------------------------
    # auxiliary loss
    # ------------------------------------------------------------------

    @staticmethod
    def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
        """Shazeer-2017 load-balancing auxiliary loss.

        Defined as ``n_experts * mean(gate_weights ** 2)``.  This quantity
        equals ``1`` when the gate is perfectly uniform (each expert gets
        weight ``1/E``) and grows up to ``n_experts`` when the gate
        collapses onto a single expert.  Minimising it (scaled by a small
        coefficient) keeps experts balanced during training.

        Parameters
        ----------
        gate_weights : torch.Tensor
            ``[batch, n_experts]`` -- softmax gate output as produced by
            :meth:`forward` and cached on ``last_gate_weights``.

        Returns
        -------
        torch.Tensor
            Scalar auxiliary loss (requires-grad ``True`` if
            ``gate_weights`` does).
        """
        if gate_weights.dim() != 2:
            raise ValueError(
                f"gate_weights must be 2D [batch, n_experts], got "
                f"{tuple(gate_weights.shape)}"
            )
        n_experts: int = gate_weights.size(-1)
        return n_experts * gate_weights.pow(2).mean()
