"""
Multi-Head Self-Attention and Feed-Forward modules for the I3 Small Language Model.

Built entirely from scratch in PyTorch -- no HuggingFace or pre-built attention
libraries. Implements the core components described in "Attention Is All You Need"
(Vaswani et al., 2017) with modern refinements (GELU activation, optional KV
caching for autoregressive inference).

Modules:
    MultiHeadSelfAttention  -- scaled dot-product attention with multiple heads
    FeedForward             -- position-wise two-layer FFN with GELU
    create_causal_mask      -- upper-triangular -inf mask for autoregressive decoding
    create_padding_mask     -- masks out PAD token positions
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism built from scratch.

    Implements scaled dot-product attention with multiple parallel heads.

    Architecture::

        Input  [batch, seq_len, d_model]
            -> Project to Q, K, V via separate linear layers (no bias)
            -> Split into n_heads
            -> Scaled dot-product attention per head
            -> Concatenate heads
            -> Output projection

    Supports:
        - Causal masking for autoregressive generation
        - Padding masks
        - KV caching for efficient token-by-token inference

    Parameters
    ----------
    d_model : int
        Model / embedding dimension.
    n_heads : int
        Number of attention heads.  ``d_model`` must be divisible by ``n_heads``.
    dropout : float, optional
        Dropout probability applied to attention weights (default 0.1).

    Notes
    -----
    All projection weights are initialised with Xavier uniform
    (``gain = 1 / sqrt(2)``) which keeps the variance stable across
    the residual stream at initialisation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # ----- projection matrices (no bias, as per original transformer) -----
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # KV cache slots (populated during inference with use_cache=True)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

        # Initialise weights
        self._init_weights()

    # ----- initialisation ----------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all projection weights.

        Uses ``gain = 1 / sqrt(2)`` so that the variance contributed by the
        residual connection stays approximately constant across depth.
        """
        gain = 1.0 / math.sqrt(2.0)
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(proj.weight, gain=gain)

    # ----- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[batch, seq_len, d_model]``.
        mask : torch.Tensor, optional
            Attention mask broadcastable to
            ``[batch, 1, seq_len_q, seq_len_k]``.  Convention:

            * **0** for positions that *should* be attended to,
            * **-inf** for positions that should be masked out.

            Typical shapes: ``[1, 1, S, S]`` (causal) or
            ``[B, 1, 1, S]`` (padding).  Both can be summed and passed
            together.
        use_cache : bool, optional
            If ``True``, append current K / V to the internal cache and
            use the full cached K / V for attention.  Call
            :meth:`clear_cache` between sequences (default ``False``).

        Returns
        -------
        output : torch.Tensor
            ``[batch, seq_len, d_model]``
        attn_weights : torch.Tensor
            ``[batch, n_heads, seq_len_q, seq_len_k]`` -- useful for
            visualisation / interpretability.

        Notes
        -----
        * **Numerical stability**: ``F.softmax`` internally subtracts the
          row-wise maximum before exponentiation, so the softmax is
          numerically stable even with large logits or ``-inf`` mask
          entries.
        * **Edge cases**: a single-token sequence (``seq_len == 1``) and
          an empty sequence (``seq_len == 0``) are handled correctly by
          PyTorch's linear algebra ops; no special-casing is needed.
        """
        batch_size, seq_len, _ = x.shape

        # --- linear projections ----------------------------------------------
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # --- reshape to [batch, n_heads, seq_len, d_k] -----------------------
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # --- KV cache (autoregressive inference) -----------------------------
        if use_cache:
            if self._cache_k is not None:
                K = torch.cat([self._cache_k, K], dim=2)  # concat along seq dim
                V = torch.cat([self._cache_v, V], dim=2)
            self._cache_k = K.detach()
            self._cache_v = V.detach()

        # --- scaled dot-product attention ------------------------------------
        # scores: [batch, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask is broadcastable -- works for causal [1,1,S,S],
            # padding [B,1,1,S], or their sum.
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # numerically stable
        attn_weights = self.attn_dropout(attn_weights)

        # --- weighted sum of values ------------------------------------------
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len_q, d_k]

        # --- concatenate heads -----------------------------------------------
        attn_output = (
            attn_output
            .transpose(1, 2)                          # [batch, seq_len, n_heads, d_k]
            .contiguous()
            .view(batch_size, -1, self.d_model)       # [batch, seq_len, d_model]
        )

        # --- output projection -----------------------------------------------
        output = self.W_o(attn_output)                # [batch, seq_len, d_model]

        return output, attn_weights

    # ----- cache management --------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the KV cache.  Call between sequences during inference."""
        self._cache_k = None
        self._cache_v = None


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Two linear transformations with a GELU activation in between::

        FFN(x) = Linear_2( Dropout( GELU( Linear_1(x) ) ) )

    ``d_ff`` is typically 2-4x ``d_model``.

    Parameters
    ----------
    d_model : int
        Input and output dimension.
    d_ff : int
        Inner (hidden) dimension of the feed-forward block.
    dropout : float, optional
        Dropout probability applied after the activation (default 0.1).

    Notes
    -----
    Weights are initialised with Xavier uniform (``gain = 1``).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for both linear layers."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise FFN.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, seq_len, d_model]``

        Returns
        -------
        torch.Tensor
            ``[batch, seq_len, d_model]``
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a causal (autoregressive) attention mask.

    Returns a ``[1, 1, seq_len, seq_len]`` upper-triangular mask where:

    * ``mask[..., i, j] = 0``    if ``j <= i``  (attend -- past & present)
    * ``mask[..., i, j] = -inf`` if ``j > i``   (mask out -- future)

    The leading singleton dimensions allow broadcasting over batch and
    head dimensions.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    device : torch.device, optional
        Device to create the tensor on (default: CPU).

    Returns
    -------
    torch.Tensor
        ``[1, 1, seq_len, seq_len]``
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(
    token_ids: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Create a padding mask from token IDs.

    Returns a ``[batch, 1, 1, seq_len]`` mask where:

    * ``0``    for real tokens   (attend)
    * ``-inf`` for padding tokens (mask out)

    The shape broadcasts naturally with attention scores of shape
    ``[batch, n_heads, seq_len, seq_len]``.

    Parameters
    ----------
    token_ids : torch.Tensor
        ``[batch, seq_len]`` integer token IDs.
    pad_id : int, optional
        ID of the padding token (default 0).

    Returns
    -------
    torch.Tensor
        ``[batch, 1, 1, seq_len]``
    """
    # (token_ids == pad_id) -> bool [batch, seq_len]
    mask = (token_ids == pad_id).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(1).unsqueeze(2)
