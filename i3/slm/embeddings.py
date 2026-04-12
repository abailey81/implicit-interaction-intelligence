"""Embedding layers for the custom Small Language Model.

Implements token embeddings, sinusoidal positional encodings, and a combined
transformer embedding layer. All built from scratch in PyTorch -- no
HuggingFace, no pre-trained weights.

Mathematical references:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding with scaling.

    Maps discrete token IDs to dense vectors of dimension ``d_model``.
    Following Vaswani et al. (2017), the embeddings are scaled by
    ``sqrt(d_model)`` to maintain variance relative to the positional
    encodings that will be added downstream.

    .. math::

        \\text{Embed}(x) = E[x] \\cdot \\sqrt{d_{\\text{model}}}

    where :math:`E \\in \\mathbb{R}^{V \\times d_{\\text{model}}}` is the
    learned embedding matrix and :math:`V` is the vocabulary size.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Dimensionality of the embedding vectors.
        padding_idx: Index of the ``[PAD]`` token whose embedding is
                     fixed at zero and excluded from gradient updates.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        # SEC: Validate constructor args to fail-fast on malformed input
        # rather than producing a silently broken model.
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if not 0 <= padding_idx < vocab_size:
            raise ValueError(
                f"padding_idx ({padding_idx}) must be in [0, {vocab_size})"
            )
        self.d_model: int = d_model
        self.padding_idx: int = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # SEC: GPT-2-style N(0, 0.02) init keeps embedding magnitudes small
        # and matches the global init pass in AdaptiveSLM. Previously this
        # used std=1.0 which produced embeddings ~16x too large after the
        # sqrt(d_model) scaling, destabilising training and the residual
        # stream variance.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Re-zero the padding vector after init (Embedding respects
        # padding_idx during forward, but init may have set it non-zero).
        with torch.no_grad():
            self.embedding.weight[padding_idx].fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings and scale.

        Args:
            x: Integer tensor of token IDs with shape ``[batch, seq_len]``.

        Returns:
            Float tensor of shape ``[batch, seq_len, d_model]`` containing
            the scaled embedding vectors.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from *Attention Is All You Need*.

    Injects position information into token embeddings via additive
    sinusoidal signals of varying frequencies. The encoding is **not
    learned** -- it is computed once at initialisation and registered as a
    persistent buffer.

    .. math::

        PE(\\text{pos}, 2i)   &= \\sin\\!\\left(\\frac{\\text{pos}}{10000^{2i / d_{\\text{model}}}}\\right) \\\\
        PE(\\text{pos}, 2i+1) &= \\cos\\!\\left(\\frac{\\text{pos}}{10000^{2i / d_{\\text{model}}}}\\right)

    where :math:`\\text{pos}` is the absolute position index and :math:`i`
    is the dimension index.

    The resulting encoding matrix has shape ``[1, max_seq_len, d_model]``
    and is sliced to match the actual sequence length at runtime.

    Args:
        d_model: Dimensionality of the model (must be even).
        max_seq_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding positional
                 encoding.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for sinusoidal positional encoding, "
                f"got {d_model}"
            )

        self.dropout = nn.Dropout(p=dropout)

        # Precompute the full positional encoding table.
        pe = torch.zeros(max_seq_len, d_model)  # [max_seq_len, d_model]
        position = torch.arange(
            0, max_seq_len, dtype=torch.float
        ).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]

        # Register as a buffer so it moves with the model to GPU / is saved
        # in state_dict but is NOT a learnable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Float tensor of shape ``[batch, seq_len, d_model]``
               (typically the output of :class:`TokenEmbedding`).

        Returns:
            Float tensor of shape ``[batch, seq_len, d_model]`` with
            positional encoding added and dropout applied.

        Raises:
            ValueError: If ``seq_len`` exceeds the ``max_seq_len`` set at
                        initialisation.
        """
        seq_len = x.size(1)
        # SEC: Strict bound check — refuse over-length inputs rather than
        # silently truncating, which would corrupt position information.
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds the maximum "
                f"supported length ({self.pe.size(1)}). Increase "
                f"max_seq_len at construction time."
            )
        # SEC: Empty sequence — return as-is (dropout is a no-op on empty).
        if seq_len == 0:
            return self.dropout(x)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Combined token embedding and positional encoding layer.

    This is the standard input layer for a Transformer encoder or decoder.
    It first maps token IDs to dense vectors via :class:`TokenEmbedding`
    (scaled by :math:`\\sqrt{d_{\\text{model}}}`), then adds sinusoidal
    positional encoding via :class:`SinusoidalPositionalEncoding`.

    .. math::

        \\text{TransformerEmbed}(x) = \\text{Dropout}\\!\\left(
            E[x] \\cdot \\sqrt{d_{\\text{model}}} + PE_{[:, :L]}
        \\right)

    where :math:`L` is the input sequence length.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Dimensionality of the model.
        max_seq_len: Maximum sequence length for positional encoding.
        dropout: Dropout probability (applied after positional encoding).
        padding_idx: Index of the ``[PAD]`` token.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and add positional encoding.

        Args:
            x: Integer tensor of token IDs with shape ``[batch, seq_len]``.

        Returns:
            Float tensor of shape ``[batch, seq_len, d_model]``.
        """
        return self.positional_encoding(self.token_embedding(x))
