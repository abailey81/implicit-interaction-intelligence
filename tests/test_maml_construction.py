"""Iter 127 — MAMLTrainer construction validation."""
from __future__ import annotations

import pytest
import torch.nn as nn

from i3.meta_learning.maml import MAMLTrainer


def _tiny_encoder(in_dim=8, hidden=16, out=64) -> nn.Module:
    """Minimal stand-in for the TCN encoder.  Accepts [batch, seq, in_dim]
    and returns [batch, out].  We use a 1-D average-pool + linear so the
    shape contract matches."""
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, out)

        def forward(self, x):
            # x: [batch, seq, in_dim] → mean-pool → [batch, in_dim] → [batch, out]
            return self.proj(x.mean(dim=1))

    return TinyEncoder()


def test_inner_lr_must_be_positive():
    enc = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(enc, inner_lr=0.0)


def test_outer_lr_must_be_positive():
    enc = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(enc, outer_lr=-1e-3)


def test_inner_steps_must_be_at_least_one():
    enc = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(enc, inner_steps=0)


def test_first_order_flag_accepted():
    enc = _tiny_encoder()
    t = MAMLTrainer(enc, first_order=True)
    assert t is not None


def test_default_construction():
    enc = _tiny_encoder()
    t = MAMLTrainer(enc)
    assert t is not None


def test_custom_dims_accepted():
    enc = _tiny_encoder(in_dim=8, out=32)
    t = MAMLTrainer(enc, embedding_dim=32, adaptation_dim=8)
    assert t is not None
