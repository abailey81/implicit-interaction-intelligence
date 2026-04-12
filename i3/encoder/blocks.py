"""Causal convolution building blocks for the Temporal Convolutional Network.

All components are built from scratch in PyTorch with no external libraries.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Causal 1-D convolution
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """1-D convolution with causal (left-only) zero-padding.

    The output has the same temporal length as the input, and position *t* in
    the output depends only on positions 0..t of the input -- no future
    leakage.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size:  Width of each convolutional filter.
        dilation:     Dilation factor for dilated / atrous convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.causal_padding: int = (kernel_size - 1) * dilation
        # Internal Conv1d has NO built-in padding -- we pad manually.
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self._init_weights()

    # -- Weight initialisation ------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for weights, zeros for biases."""
        init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    # -- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Tensor of shape ``[batch, channels, seq_len]``.

        Returns:
            Tensor of shape ``[batch, out_channels, seq_len]`` (same length).
        """
        # Left-pad along the temporal dimension (dim=-1).
        if self.causal_padding > 0:
            x = nn.functional.pad(x, (self.causal_padding, 0))
        out = self.conv(x)
        return out


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Residual block with two causal dilated convolutions.

    Architecture::

        x --> CausalConv1d --> LayerNorm --> GELU
          --> CausalConv1d --> LayerNorm --> GELU --> Dropout --> (+residual)

    If ``input_dim != output_dim`` a learnable 1x1 pointwise convolution is
    used for the skip / residual path so that the shapes match for addition.

    Args:
        input_dim:   Number of input channels.
        output_dim:  Number of output channels.
        kernel_size: Kernel width for both causal convolutions.
        dilation:    Dilation factor for both causal convolutions.
        dropout:     Dropout probability applied after the second activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # -- Main path --------------------------------------------------------
        self.conv1 = CausalConv1d(input_dim, output_dim, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(output_dim)
        self.act1 = nn.GELU()

        self.conv2 = CausalConv1d(output_dim, output_dim, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(output_dim)
        self.act2 = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        # -- Residual path ----------------------------------------------------
        self.skip: Optional[nn.Conv1d] = None
        if input_dim != output_dim:
            self.skip = nn.Conv1d(input_dim, output_dim, kernel_size=1)
            init.xavier_uniform_(self.skip.weight)
            if self.skip.bias is not None:
                init.zeros_(self.skip.bias)

    # -- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the residual block.

        Args:
            x: Tensor of shape ``[batch, channels, seq_len]``.

        Returns:
            Tensor of shape ``[batch, output_dim, seq_len]``.
        """
        residual = x if self.skip is None else self.skip(x)

        # First causal conv + LayerNorm + GELU
        out = self.conv1(x)
        # LayerNorm expects [..., C] so transpose C <-> T
        out = out.transpose(1, 2)        # [B, T, C]
        out = self.norm1(out)
        out = self.act1(out)
        out = out.transpose(1, 2)        # [B, C, T]

        # Second causal conv + LayerNorm + GELU
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = self.act2(out)
        out = out.transpose(1, 2)

        # Dropout + residual
        out = self.dropout(out)
        out = out + residual
        return out
