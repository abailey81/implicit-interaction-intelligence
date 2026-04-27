"""Iter 74 — TemporalConvNet (TCN encoder) invariant tests.

Pins shape, L2 norm, gradient flow, and config validation contracts of
i3.encoder.tcn.TemporalConvNet — the from-scratch dilated TCN that
maps keystroke-feature sequences to the 64-D user-state embedding
fed into the SLM's cross-attention.
"""
from __future__ import annotations

import math

import pytest
import torch

from i3.encoder.tcn import TemporalConvNet


@pytest.fixture
def tcn():
    return TemporalConvNet()


# ---------------------------------------------------------------------------
# Output shape + L2 norm contract
# ---------------------------------------------------------------------------

def test_forward_returns_64d_per_batch(tcn):
    x = torch.randn(4, 32, 32)  # [batch, seq, feat]
    y = tcn(x)
    assert y.shape == (4, 64)
    assert y.dtype == torch.float32


def test_output_is_l2_normalised(tcn):
    x = torch.randn(8, 50, 32)
    y = tcn(x)
    norms = y.norm(dim=-1)
    # Should all be ~1.0 (unit hypersphere)
    for n in norms:
        assert math.isclose(float(n), 1.0, abs_tol=1e-4), \
            f"unit-norm violated: {float(n)}"


def test_handles_short_sequence(tcn):
    """Receptive field is k=3 × dilations [1,2,4,8] → effective ~16.
    Even shorter sequences must produce a valid embedding (causal
    conv handles padding)."""
    x = torch.randn(2, 5, 32)
    y = tcn(x)
    assert y.shape == (2, 64)
    assert torch.isfinite(y).all()


def test_handles_long_sequence(tcn):
    x = torch.randn(2, 1000, 32)
    y = tcn(x)
    assert y.shape == (2, 64)


def test_eval_mode_is_batch_independent(tcn):
    """In eval mode (BatchNorm uses running stats, no cross-batch
    coupling), two batches should produce identical embeddings for
    matching inputs."""
    tcn.eval()
    a = torch.randn(1, 20, 32)
    b = torch.randn(1, 20, 32)
    with torch.inference_mode():
        y_a_alone = tcn(a)
        combined = torch.cat([a, b], dim=0)
        y_a_in_batch = tcn(combined)[:1]
    assert torch.allclose(y_a_alone, y_a_in_batch, atol=1e-4), \
        "TCN eval-mode forward is not batch-independent"


# ---------------------------------------------------------------------------
# Gradients flow
# ---------------------------------------------------------------------------

def test_gradients_flow_to_input(tcn):
    x = torch.randn(2, 30, 32, requires_grad=True)
    y = tcn(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_mismatched_dims_dilations_raises():
    with pytest.raises(ValueError):
        TemporalConvNet(hidden_dims=[64, 64], dilations=[1, 2, 4])


def test_custom_dimensions_round_trip():
    enc = TemporalConvNet(input_dim=16, hidden_dims=[32, 32],
                          dilations=[1, 2], embedding_dim=48)
    x = torch.randn(3, 20, 16)
    y = enc(x)
    assert y.shape == (3, 48)
