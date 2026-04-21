"""Smoke tests for the Lightning Fabric training wrapper.

These tests are deliberately minimal:

* they confirm that :mod:`training.train_encoder_fabric` imports even
  without ``lightning`` installed;
* when ``lightning`` *is* installed they run a single optimization step
  through a tiny TCN to confirm the Fabric plumbing works end-to-end on a
  single-CPU device.

The tests never assert on model quality; they are strictly smoke checks
meant to guard against import regressions and basic wiring breakage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

# ---------------------------------------------------------------------------
# Soft-check for optional deps
# ---------------------------------------------------------------------------

try:
    import lightning as _lightning  # type: ignore[import-not-found]  # noqa: F401

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False


if TYPE_CHECKING:
    from training import train_encoder_fabric  # noqa: F401


def test_fabric_module_imports() -> None:
    """Module must import regardless of whether lightning is installed."""
    from training import train_encoder_fabric

    assert hasattr(train_encoder_fabric, "run_fabric")
    assert hasattr(train_encoder_fabric, "main")


def test_fabric_availability_flag_matches_import() -> None:
    """_LIGHTNING_AVAILABLE must match the actual import state."""
    from training import train_encoder_fabric

    assert train_encoder_fabric._LIGHTNING_AVAILABLE == _HAS_LIGHTNING


@pytest.mark.skipif(not _HAS_LIGHTNING, reason="lightning not installed")
def test_fabric_one_step_cpu(tmp_path) -> None:
    """Run one Fabric step on CPU and confirm a checkpoint file is written.

    Args:
        tmp_path: Pytest tmp path fixture.
    """
    from lightning.fabric import Fabric  # type: ignore[import-not-found]

    from i3.encoder.tcn import TemporalConvNet
    from i3.encoder.train import contrastive_loss

    fabric = Fabric(accelerator="cpu", devices=1, strategy="auto", precision="32-true")
    fabric.launch()

    model = TemporalConvNet(
        input_dim=32,
        hidden_dims=[16, 16],
        kernel_size=3,
        dilations=[1, 2],
        embedding_dim=16,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = fabric.setup(model, optimizer)

    # Minimal batch: 8 sequences of length 32 with 4 labels.
    sequences = torch.randn(8, 32, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    embeddings = model(sequences)
    loss = contrastive_loss(embeddings, labels, temperature=0.1)
    optimizer.zero_grad(set_to_none=True)
    fabric.backward(loss)
    optimizer.step()

    ckpt = tmp_path / "smoke.pt"
    fabric.save(str(ckpt), {"model": model, "optimizer": optimizer})
    assert ckpt.exists()
    assert torch.isfinite(loss)
