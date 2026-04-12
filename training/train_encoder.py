"""CLI script for training the User State Encoder (TCN).

Loads configuration from ``configs/default.yaml``, reads synthetic data from
``data/synthetic/``, instantiates :class:`TemporalConvNet`, runs the full
training loop, and saves the best checkpoint to ``models/encoder/``.

Usage::

    python -m training.train_encoder                         # defaults
    python -m training.train_encoder --epochs 200 --batch-size 128
    python training/train_encoder.py --config config/custom.yaml

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.encoder.tcn import TemporalConvNet
from i3.encoder.train import train

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return a YAML config file as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _merge(base: dict, overrides: dict) -> dict:
    """Shallow-merge *overrides* into *base* (non-recursive)."""
    merged = dict(base)
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(path: Path) -> TensorDataset:
    """Load a ``.pt`` split file produced by ``generate_synthetic.py``.

    Args:
        path: Path to the ``.pt`` file containing ``"sequences"`` and
              ``"labels"`` tensors.

    Returns:
        A :class:`TensorDataset` of ``(sequences, labels)``.
    """
    # Synthetic training splits are plain dict[str, Tensor] produced by
    # our own generator; weights_only=True is safe and blocks pickled
    # code execution.
    data = torch.load(path, map_location="cpu", weights_only=True)
    sequences = data["sequences"]  # [N, window, 32]
    labels = data["labels"]        # [N]
    logger.info("Loaded %s: %d samples, seq shape %s", path.name, len(sequences), list(sequences.shape))
    return TensorDataset(sequences, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, build model, train, and report results."""
    parser = argparse.ArgumentParser(
        description="Train the User State Encoder TCN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="NT-Xent temperature (default: 0.07).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Directory containing train.pt / val.pt / test.pt.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/encoder",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device override (cpu, cuda, mps). Auto-detects if not set.",
    )
    args = parser.parse_args()

    # SEC: bound-check numeric CLI arguments to catch obviously bad inputs
    # (e.g. negative epochs, zero batch size) before any heavy work runs.
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be a positive float")
    if args.temperature <= 0:
        parser.error("--temperature must be a positive float")
    if args.patience < 0:
        parser.error("--patience must be non-negative")

    # -- Logging --------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # -- Config ---------------------------------------------------------------
    # SEC: resolve the config path so log messages and downstream code see
    # the absolute path; YAML is parsed via yaml.safe_load (no code exec).
    config_path = Path(args.config).resolve()
    if config_path.exists() and config_path.is_file():
        cfg = load_config(config_path)
        enc_cfg = cfg.get("encoder", {})
    else:
        logger.warning("Config %s not found; using defaults.", config_path)
        enc_cfg = {}

    # -- Seed -----------------------------------------------------------------
    # SEC: seed every RNG framework we use (Python random, NumPy, torch)
    # so the data shuffling, augmentation noise, and weight init are all
    # reproducible from a single seed.
    seed = args.seed
    import random as _py_random

    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Device ---------------------------------------------------------------
    # Detection priority: explicit override > CUDA > MPS (Apple Silicon) > CPU.
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # -- Data -----------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_ds = load_split(data_dir / "train.pt")
    val_ds = load_split(data_dir / "val.pt")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # -- Model ----------------------------------------------------------------
    model = TemporalConvNet(
        input_dim=enc_cfg.get("input_dim", 32),
        hidden_dims=enc_cfg.get("hidden_dims", [64, 64, 64, 64]),
        kernel_size=enc_cfg.get("kernel_size", 3),
        dilations=enc_cfg.get("dilations", [1, 2, 4, 8]),
        embedding_dim=enc_cfg.get("embedding_dim", 64),
        dropout=enc_cfg.get("dropout", 0.1),
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)
    logger.info("Receptive field: %d timesteps", model.get_receptive_field())

    # -- Training hyperparams -------------------------------------------------
    epochs = args.epochs if args.epochs is not None else 100
    lr = args.lr if args.lr is not None else 1e-3

    # -- Train ----------------------------------------------------------------
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        temperature=args.temperature,
        grad_clip=1.0,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=10,
        patience=args.patience,
        device=device,
    )

    # -- Report ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Best epoch:        %d", results["best_epoch"])
    logger.info("Best val loss:     %.4f", results["best_val_loss"])

    final = results.get("final_metrics", {})
    logger.info("Silhouette score:  %.4f", final.get("silhouette_score", 0.0))
    logger.info("KNN accuracy:      %.4f", final.get("knn_accuracy", 0.0))
    logger.info("Checkpoint saved:  %s/best_model.pt", args.checkpoint_dir)

    # -- Evaluate on test set -------------------------------------------------
    test_path = data_dir / "test.pt"
    if test_path.exists():
        from i3.encoder.train import validate

        test_ds = load_split(test_path)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        test_metrics = validate(model, test_loader, args.temperature, device)
        logger.info("-" * 40)
        logger.info("TEST SET METRICS")
        logger.info("-" * 40)
        logger.info("Test loss:         %.4f", test_metrics["loss"])
        logger.info("Test silhouette:   %.4f", test_metrics["silhouette_score"])
        logger.info("Test KNN accuracy: %.4f", test_metrics["knn_accuracy"])
    else:
        logger.warning("No test.pt found at %s; skipping test evaluation.", test_path)


if __name__ == "__main__":
    main()
