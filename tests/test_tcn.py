"""Tests for the Temporal Convolutional Network encoder.

Validates causal convolution, residual blocks, output shape, L2
normalisation, receptive field computation, and batch independence.
"""

from __future__ import annotations

import pytest
import torch

from i3.encoder.blocks import CausalConv1d, ResidualBlock
from i3.encoder.tcn import TemporalConvNet


# -------------------------------------------------------------------------
# CausalConv1d
# -------------------------------------------------------------------------

class TestCausalConv:
    """Tests for the causal (left-only) 1-D convolution layer."""

    @pytest.mark.parametrize("seq_len", [1, 8, 32, 64])
    def test_output_preserves_length(self, seq_len: int) -> None:
        """Output temporal dimension must equal input temporal dimension."""
        conv = CausalConv1d(in_channels=16, out_channels=16, kernel_size=3, dilation=1)
        x = torch.randn(2, 16, seq_len)
        out = conv(x)
        assert out.shape == (2, 16, seq_len), (
            f"Expected (2, 16, {seq_len}), got {out.shape}"
        )

    def test_causal_no_future_leakage(self) -> None:
        """Position t in the output must depend only on positions 0..t."""
        conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)

        # Two inputs identical up to position 3, differ at position 4
        x1 = torch.zeros(1, 1, 6)
        x2 = torch.zeros(1, 1, 6)
        x1[0, 0, 4] = 0.0
        x2[0, 0, 4] = 999.0

        out1 = conv(x1)
        out2 = conv(x2)

        # Outputs at positions 0..3 must be identical (no future leakage)
        torch.testing.assert_close(out1[:, :, :4], out2[:, :, :4])

        # Position 4 onwards may differ (the change is visible)
        # (We don't assert they differ, only that the past is unaffected.)

    @pytest.mark.parametrize("dilation", [1, 2, 4, 8])
    def test_different_dilations(self, dilation: int) -> None:
        """Dilated causal conv still preserves sequence length."""
        conv = CausalConv1d(
            in_channels=8, out_channels=8, kernel_size=3, dilation=dilation
        )
        x = torch.randn(2, 8, 32)
        out = conv(x)
        assert out.shape == (2, 8, 32)

    def test_causal_padding_value(self) -> None:
        """The causal_padding attribute should be (kernel_size - 1) * dilation."""
        conv = CausalConv1d(in_channels=4, out_channels=4, kernel_size=5, dilation=3)
        assert conv.causal_padding == (5 - 1) * 3

    def test_channel_change(self) -> None:
        """CausalConv1d should support different in/out channel counts."""
        conv = CausalConv1d(in_channels=8, out_channels=32, kernel_size=3, dilation=1)
        x = torch.randn(1, 8, 20)
        out = conv(x)
        assert out.shape == (1, 32, 20)


# -------------------------------------------------------------------------
# ResidualBlock
# -------------------------------------------------------------------------

class TestResidualBlock:
    """Tests for the residual block with skip connection."""

    def test_output_shape_same_channels(self) -> None:
        """Output shape should equal input shape when channels match."""
        block = ResidualBlock(input_dim=64, output_dim=64, kernel_size=3, dilation=2)
        x = torch.randn(4, 64, 20)
        out = block(x)
        assert out.shape == (4, 64, 20)

    def test_output_shape_channel_change(self) -> None:
        """Residual block with channel change uses 1x1 skip convolution."""
        block = ResidualBlock(input_dim=32, output_dim=64, kernel_size=3, dilation=1)
        x = torch.randn(2, 32, 16)
        out = block(x)
        assert out.shape == (2, 64, 16)
        # Should have a skip convolution
        assert block.skip is not None

    def test_skip_is_none_when_same_dim(self) -> None:
        """No skip convolution needed when input_dim == output_dim."""
        block = ResidualBlock(input_dim=64, output_dim=64)
        assert block.skip is None

    def test_residual_connection(self) -> None:
        """The residual connection should be active (output != main path alone)."""
        block = ResidualBlock(input_dim=64, output_dim=64, kernel_size=3, dilation=1)
        x = torch.randn(1, 64, 10)

        # Run main path manually (without residual)
        out_main = block.conv1(x)
        out_main = out_main.transpose(1, 2)
        out_main = block.norm1(out_main)
        out_main = block.act1(out_main)
        out_main = out_main.transpose(1, 2)
        out_main = block.conv2(out_main)
        out_main = out_main.transpose(1, 2)
        out_main = block.norm2(out_main)
        out_main = block.act2(out_main)
        out_main = out_main.transpose(1, 2)

        # Full forward (with residual)
        block.eval()
        with torch.no_grad():
            out_full = block(x)

        # They should differ because out_full = dropout(out_main) + x
        # (In eval mode dropout is identity, so out_full = out_main + x)
        assert not torch.allclose(out_main, out_full, atol=1e-6)


# -------------------------------------------------------------------------
# TemporalConvNet
# -------------------------------------------------------------------------

class TestTCN:
    """Tests for the full Temporal Convolutional Network encoder."""

    @pytest.fixture
    def tcn(self) -> TemporalConvNet:
        """Create a default-config TCN encoder."""
        return TemporalConvNet(
            input_dim=32,
            hidden_dims=[64, 64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            embedding_dim=64,
            dropout=0.1,
        )

    def test_output_shape(self, tcn: TemporalConvNet) -> None:
        """[batch, seq, 32] -> [batch, 64]."""
        x = torch.randn(4, 20, 32)
        out = tcn(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_l2_normalized(self, tcn: TemporalConvNet) -> None:
        """Output embeddings must lie on the unit sphere."""
        x = torch.randn(8, 15, 32)
        tcn.eval()
        with torch.no_grad():
            out = tcn(x)
        norms = torch.norm(out, p=2, dim=1)
        torch.testing.assert_close(
            norms,
            torch.ones(8),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_receptive_field(self, tcn: TemporalConvNet) -> None:
        """Verify the receptive field formula:
        RF = 1 + sum_i 2 * (k-1) * d_i for each block.
        """
        # k=3, dilations=[1,2,4,8]
        # RF = 1 + 2*(2)*1 + 2*(2)*2 + 2*(2)*4 + 2*(2)*8
        #    = 1 + 4 + 8 + 16 + 32 = 61
        assert tcn.get_receptive_field() == 61

    def test_receptive_field_custom(self) -> None:
        """Receptive field with custom dilations."""
        tcn = TemporalConvNet(
            hidden_dims=[64, 64],
            dilations=[1, 4],
            kernel_size=5,
        )
        # RF = 1 + 2*(4)*1 + 2*(4)*4 = 1 + 8 + 32 = 41
        assert tcn.get_receptive_field() == 41

    def test_batch_independence(self, tcn: TemporalConvNet) -> None:
        """Each sample in a batch must be processed independently."""
        tcn.eval()
        x = torch.randn(3, 10, 32)
        with torch.no_grad():
            out_batch = tcn(x)
            out_0 = tcn(x[0:1])
            out_1 = tcn(x[1:2])
            out_2 = tcn(x[2:3])

        torch.testing.assert_close(out_batch[0], out_0[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_batch[1], out_1[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_batch[2], out_2[0], atol=1e-5, rtol=1e-5)

    def test_single_timestep(self, tcn: TemporalConvNet) -> None:
        """Should handle a single-timestep input without errors."""
        x = torch.randn(1, 1, 32)
        out = tcn(x)
        assert out.shape == (1, 64)

    def test_mismatched_dims_raises(self) -> None:
        """hidden_dims and dilations must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            TemporalConvNet(hidden_dims=[64, 64], dilations=[1, 2, 4])

    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 5), (2, 10), (16, 50),
    ])
    def test_various_batch_seq(
        self, tcn: TemporalConvNet, batch_size: int, seq_len: int
    ) -> None:
        """TCN should handle various batch/sequence size combinations."""
        x = torch.randn(batch_size, seq_len, 32)
        out = tcn(x)
        assert out.shape == (batch_size, 64)
