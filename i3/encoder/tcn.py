"""Temporal Convolutional Network encoder for user-state embeddings.

Transforms a sequence of 32-dim interaction feature vectors into a single
64-dim L2-normalised embedding suitable for contrastive learning on the unit
hypersphere.

Built entirely from scratch in PyTorch -- no external temporal-conv libraries.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3.encoder.blocks import ResidualBlock

logger = logging.getLogger(__name__)


class TemporalConvNet(nn.Module):
    """User State Encoder -- dilated causal TCN with global average pooling.

    Full architecture::

        Input:  [batch, seq_len, 32]
          --> Linear(32, 64)                          # input projection
          --> Transpose to [batch, 64, seq_len]
          --> ResidualBlock(64, 64, k=3, d=1)
          --> ResidualBlock(64, 64, k=3, d=2)
          --> ResidualBlock(64, 64, k=3, d=4)
          --> ResidualBlock(64, 64, k=3, d=8)
          --> Global Average Pooling  -> [batch, 64]
          --> Linear(64, 64)                          # output projection
          --> L2 normalise
        Output: [batch, 64] unit-norm embedding

    Args:
        input_dim:     Dimensionality of each timestep feature (default 32).
        hidden_dims:   Channel width for each residual block.  Length must
                       match ``dilations``.
        kernel_size:   Kernel width shared by all causal convolutions.
        dilations:     Per-block dilation factors.
        embedding_dim: Dimensionality of the output embedding (default 64).
        dropout:       Dropout probability inside each residual block.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list[int] | None = None,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        if dilations is None:
            dilations = [1, 2, 4, 8]
        if len(hidden_dims) != len(dilations):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dilations "
                f"({len(dilations)}) must have the same length."
            )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # -- Input projection --------------------------------------------------
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # -- Residual blocks ---------------------------------------------------
        blocks: list[nn.Module] = []
        in_ch = hidden_dims[0]
        for out_ch, d in zip(hidden_dims, dilations):
            blocks.append(
                ResidualBlock(
                    input_dim=in_ch,
                    output_dim=out_ch,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # -- Output projection -------------------------------------------------
        self.output_proj = nn.Linear(hidden_dims[-1], embedding_dim)

        # Log model size
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TemporalConvNet created: %d blocks, %s dilations, "
            "%d params, receptive field %d",
            len(self.blocks),
            dilations,
            n_params,
            self.get_receptive_field(),
        )

    # -- Receptive field computation ------------------------------------------

    def get_receptive_field(self) -> int:
        """Compute the effective receptive field of the TCN stack.

        For a stack of causal dilated convolutions each with kernel size *k*
        and dilations d_1, ..., d_L the receptive field is::

            1 + sum_i  (k - 1) * d_i    (over all conv layers)

        Each :class:`ResidualBlock` contains **two** causal convolutions that
        share the same dilation, so each block contributes ``2 * (k - 1) * d``.

        Returns:
            Integer number of input timesteps that influence each output step.
        """
        rf = 1
        for d in self.dilations:
            # Two causal conv layers per block, same dilation
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    # -- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of interaction-feature sequences.

        Args:
            x: Tensor of shape ``[batch, seq_len, input_dim]``.

        Returns:
            Tensor of shape ``[batch, embedding_dim]``, L2-normalised so that
            each embedding lies on the unit hypersphere.
        """
        # Input projection: [B, T, input_dim] -> [B, T, hidden]
        out = self.input_proj(x)

        # Transpose for Conv1d: [B, T, C] -> [B, C, T]
        out = out.transpose(1, 2)

        # Residual blocks
        for block in self.blocks:
            out = block(out)

        # Global average pooling: [B, C, T] -> [B, C]
        out = out.mean(dim=2)

        # Output projection
        out = self.output_proj(out)

        # L2 normalisation (unit sphere)
        out = F.normalize(out, p=2, dim=1)

        return out
