"""User State Encoder -- Temporal Convolutional Network for implicit interaction signals.

This package provides a TCN-based encoder that transforms sequences of 32-dim
:class:`~src.interaction.types.InteractionFeatureVector` observations into
64-dim user state embeddings on the unit hypersphere (L2-normalised).

Exports:
    TemporalConvNet:   The full TCN encoder model.
    CausalConvBlock:   A single causal dilated convolution.
    ResidualBlock:     Residual block with two causal convolutions.
    EncoderInference:  Stateful inference wrapper with per-user rolling windows.
    contrastive_loss:  NT-Xent (InfoNCE) contrastive loss for training.
"""

from i3.encoder.blocks import CausalConv1d as CausalConvBlock, ResidualBlock
from i3.encoder.inference import EncoderInference
from i3.encoder.tcn import TemporalConvNet
from i3.encoder.train import contrastive_loss

__all__ = [
    "CausalConvBlock",
    "EncoderInference",
    "ResidualBlock",
    "TemporalConvNet",
    "contrastive_loss",
]
