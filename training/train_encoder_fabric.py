"""Fabric-based distributed training for the User State Encoder (TCN).

This module provides a PyTorch Lightning *Fabric* wrapper around the existing
NT-Xent contrastive training loop defined in :mod:`i3.encoder.train`. Fabric is
intentionally *lightweight*: it gives us multi-device launching, mixed
precision, and gradient scaling while keeping the training loop as plain
PyTorch. This lets the I3 encoder scale from one CPU box to a multi-GPU
cluster without rewriting the loop as a ``LightningModule``.

Design notes
------------

* ``lightning`` is *soft-imported*. If it is not installed the script falls
  back to a single-CPU training run that imports and calls
  :func:`training.train_encoder.main` directly, so the file is safe to drop
  into a stripped-down environment (CI, minimal Docker image).
* Checkpoints go to ``checkpoints/encoder/fabric/`` to avoid colliding with
  the non-Fabric path at ``models/encoder/``.
* The loop uses ``bf16-mixed`` precision by default; override with
  ``--precision`` if the accelerator does not support it.

Usage
-----

.. code-block:: bash

    # Single node, all visible GPUs
    python -m training.train_encoder_fabric --epochs 50

    # Two nodes, 8 GPUs each (launcher managed externally)
    lightning run model training/train_encoder_fabric.py \
        --strategy ddp --devices 8 --num-nodes 2

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.encoder.tcn import TemporalConvNet
from i3.encoder.train import contrastive_loss  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from lightning.fabric import Fabric  # noqa: F401

# ---------------------------------------------------------------------------
# Soft-import lightning
# ---------------------------------------------------------------------------

try:
    import lightning as _lightning  # type: ignore[import-not-found]

    _LIGHTNING_AVAILABLE = True
except ImportError:
    _lightning = None  # type: ignore[assignment]
    _LIGHTNING_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dict.

    Args:
        path: Path to a YAML file.

    Returns:
        The parsed config as a dict. Empty dict if the file is empty.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def _load_tensor_dataset(path: Path) -> TensorDataset:
    """Load a ``.pt`` split file into a TensorDataset.

    Args:
        path: Path to a ``.pt`` file with ``sequences`` and ``labels``.

    Returns:
        TensorDataset wrapping (sequences, labels).
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TensorDataset(data["sequences"], data["labels"])


# ---------------------------------------------------------------------------
# Fabric training loop
# ---------------------------------------------------------------------------


def _train_one_epoch_fabric(
    fabric: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    grad_clip: float,
    epoch: int,
) -> float:
    """Run one training epoch under Fabric.

    Args:
        fabric: The ``lightning.Fabric`` instance.
        model: The (already wrapped) TCN model.
        loader: The (already wrapped) train DataLoader.
        optimizer: The (already wrapped) optimizer.
        temperature: NT-Xent temperature.
        grad_clip: Max grad norm; ``<= 0`` disables clipping.
        epoch: The current epoch index (for logging).

    Returns:
        The mean loss over the epoch.
    """
    model.train()
    running = 0.0
    count = 0
    for step, batch in enumerate(loader):
        sequences, labels = batch
        embeddings = model(sequences)
        loss = contrastive_loss(embeddings, labels, temperature)
        optimizer.zero_grad(set_to_none=True)
        fabric.backward(loss)
        if grad_clip > 0:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
        optimizer.step()
        running += float(loss.detach())
        count += 1
        if step % 50 == 0:
            fabric.log("train/loss_step", float(loss.detach()), step=epoch * len(loader) + step)
    return running / max(count, 1)


@torch.no_grad()
def _validate_fabric(
    fabric: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    temperature: float,
) -> float:
    """Compute validation NT-Xent loss.

    Args:
        fabric: The ``lightning.Fabric`` instance (used for logging only).
        model: The wrapped TCN model.
        loader: The wrapped validation loader.
        temperature: NT-Xent temperature.

    Returns:
        The mean validation loss.
    """
    model.eval()
    running = 0.0
    count = 0
    for sequences, labels in loader:
        embeddings = model(sequences)
        loss = contrastive_loss(embeddings, labels, temperature)
        running += float(loss.detach())
        count += 1
    return running / max(count, 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_fabric(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Fabric-based training loop.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A dict with ``best_epoch`` and ``best_val_loss``.
    """
    if not _LIGHTNING_AVAILABLE:
        logger.warning(
            "lightning is not installed; falling back to single-CPU path. "
            "Install `poetry install --with distributed` to enable Fabric."
        )
        from training.train_encoder import main as cpu_main

        cpu_main()
        return {"best_epoch": -1, "best_val_loss": float("nan")}

    from lightning.fabric import Fabric  # type: ignore[import-not-found]

    fabric = Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
    )
    fabric.launch()

    # Load config
    cfg_path = Path(args.config).resolve()
    enc_cfg = load_config(cfg_path).get("encoder", {}) if cfg_path.exists() else {}

    # Seed
    fabric.seed_everything(args.seed)

    # Build model
    model = TemporalConvNet(
        input_dim=enc_cfg.get("input_dim", 32),
        hidden_dims=enc_cfg.get("hidden_dims", [64, 64, 64, 64]),
        kernel_size=enc_cfg.get("kernel_size", 3),
        dilations=enc_cfg.get("dilations", [1, 2, 4, 8]),
        embedding_dim=enc_cfg.get("embedding_dim", 64),
        dropout=enc_cfg.get("dropout", 0.1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Wrap for distributed training
    model, optimizer = fabric.setup(model, optimizer)

    # Data
    data_dir = Path(args.data_dir)
    train_ds = _load_tensor_dataset(data_dir / "train.pt")
    val_ds = _load_tensor_dataset(data_dir / "val.pt")
    train_loader = fabric.setup_dataloaders(
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    )
    val_loader = fabric.setup_dataloaders(
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    )

    # Checkpoint dir (Fabric-specific to avoid collision)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = _train_one_epoch_fabric(
            fabric, model, train_loader, optimizer, args.temperature, args.grad_clip, epoch
        )
        val_loss = _validate_fabric(fabric, model, val_loader, args.temperature)
        fabric.log_dict({"train/loss": train_loss, "val/loss": val_loss}, step=epoch)
        if fabric.is_global_zero:
            logger.info(
                "epoch %d  train_loss=%.4f  val_loss=%.4f", epoch, train_loss, val_loss
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            state = {"model": model, "optimizer": optimizer, "epoch": epoch, "val_loss": val_loss}
            fabric.save(str(ckpt_dir / "best_model.pt"), state)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if fabric.is_global_zero:
                    logger.info("Early stopping at epoch %d", epoch)
                break

    return {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Fabric trainer.

    Args:
        argv: Optional explicit argv; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Fabric-based TCN training for I3.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/encoder/fabric")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    return args


def main() -> None:
    """Entry point: parse args, configure logging, launch training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    args = _parse_args()
    results = run_fabric(args)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
