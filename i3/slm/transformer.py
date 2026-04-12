"""
Pre-LayerNorm Transformer block with cross-attention conditioning for the I3
Small Language Model.

Built entirely from scratch in PyTorch -- no HuggingFace. Implements a
three-sub-layer Transformer block following the Pre-LN pattern
(Xiong et al., 2020, "On Layer Normalization in the Transformer Architecture")
with an additional cross-attention sub-layer that injects conditioning signals
(adaptation vector + user state) into the residual stream at every layer.

Module:
    AdaptiveTransformerBlock -- Pre-LN block with self-attention,
                                cross-attention, and feed-forward sub-layers.

Architecture per block::

    x --> LayerNorm --> Self-Attention  --> + residual --> x
    x --> LayerNorm --> Cross-Attention --> + residual --> x
    x --> LayerNorm --> Feed-Forward    --> + residual --> x

Pre-LN is preferred over Post-LN because:
    1. More stable training -- gradients flow cleanly through the residual
       path without being gated by normalisation.
    2. No need for careful learning-rate warmup schedules.
    3. Better for small models (6-8 M parameters) where training instability
       is especially costly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AdaptiveTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with cross-attention conditioning.

    Each block applies three sub-layers in sequence, each wrapped in the
    Pre-LN residual pattern ``x + Dropout(SubLayer(LayerNorm(x)))``:

    1. **Self-attention** -- standard causal multi-head self-attention over the
       input sequence, allowing each token to attend to past and present
       positions.
    2. **Cross-attention** -- multi-head cross-attention where queries come from
       the input sequence and keys/values come from the conditioning tokens
       produced by :class:`~src.slm.cross_attention.ConditioningProjector`.
       This is the mechanism through which adaptation signals (cognitive load,
       style preferences, emotional tone, accessibility needs) modulate the
       hidden representations at every layer.
    3. **Feed-forward** -- position-wise two-layer FFN with GELU activation.

    Parameters
    ----------
    d_model : int
        Model / embedding dimension.
    n_heads : int
        Number of heads for self-attention.
    d_ff : int
        Inner dimension of the feed-forward sub-layer (typically 2-4x
        ``d_model``).
    dropout : float
        Dropout probability applied after each sub-layer output, before
        the residual addition.
    n_cross_heads : int, optional
        Number of heads for the cross-attention sub-layer (default 2).
        Fewer heads than self-attention is typical because the conditioning
        sequence is short (4 tokens) and the information is dense.

    Attributes
    ----------
    self_attn : MultiHeadSelfAttention
        Self-attention sub-layer from :mod:`src.slm.attention`.
    cross_attn : MultiHeadCrossAttention
        Cross-attention sub-layer from :mod:`src.slm.cross_attention`.
    ff : FeedForward
        Position-wise feed-forward sub-layer from :mod:`src.slm.attention`.
    ln1, ln2, ln3 : nn.LayerNorm
        Layer normalisation applied *before* each sub-layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_cross_heads: int = 2,
    ) -> None:
        super().__init__()

        # Import sibling modules -- kept inside __init__ to avoid circular
        # imports at module-load time.
        from i3.slm.attention import MultiHeadSelfAttention, FeedForward
        from i3.slm.cross_attention import MultiHeadCrossAttention

        # ----- sub-layers -------------------------------------------------------
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_cross_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        # ----- layer norms (one per sub-layer, Pre-LN pattern) -----------------
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # ----- residual dropout ------------------------------------------------
        self.dropout = nn.Dropout(dropout)

    # ----- forward --------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[batch, seq_len, d_model]``.
        conditioning_tokens : torch.Tensor
            Conditioning context of shape ``[batch, n_cond, d_model]``
            produced by :class:`~src.slm.cross_attention.ConditioningProjector`.
            These serve as keys and values for the cross-attention sub-layer.
        causal_mask : torch.Tensor, optional
            Upper-triangular causal mask of shape
            ``[1, 1, seq_len, seq_len]`` with ``0`` for attend and
            ``-inf`` for mask-out.  Passed directly to self-attention.
        use_cache : bool, optional
            If ``True``, use / update the KV cache inside self-attention
            for efficient autoregressive inference (default ``False``).

        Returns
        -------
        output : torch.Tensor
            ``[batch, seq_len, d_model]`` -- the block output, same spatial
            shape as the input.
        attention_info : dict[str, torch.Tensor]
            Dictionary with two entries:

            - ``"self_attn"``  -- self-attention weights
              ``[batch, n_heads, seq_len, seq_len]``
            - ``"cross_attn"`` -- cross-attention weights
              ``[batch, n_cross_heads, seq_len, n_cond]``

            Useful for interpretability and visualisation.
        """
        # ----- Sub-layer 1: Pre-LN Self-Attention -----
        h = self.ln1(x)
        h, self_attn_weights = self.self_attn(h, mask=causal_mask, use_cache=use_cache)
        x = x + self.dropout(h)

        # ----- Sub-layer 2: Pre-LN Cross-Attention to conditioning -----
        h = self.ln2(x)
        h, cross_attn_weights = self.cross_attn(
            query=h,
            key=conditioning_tokens,
            value=conditioning_tokens,
        )
        x = x + self.dropout(h)

        # ----- Sub-layer 3: Pre-LN Feed-Forward -----
        h = self.ln3(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        attention_info: dict[str, torch.Tensor] = {
            "self_attn": self_attn_weights,
            "cross_attn": cross_attn_weights,
        }

        return x, attention_info

    # ----- cache management -----------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the KV cache in the self-attention sub-layer.

        Must be called between sequences during autoregressive inference.
        Cross-attention has no cache because the conditioning tokens are
        recomputed per sequence.
        """
        self.self_attn.clear_cache()
