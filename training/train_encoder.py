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
import os
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
from i3.runtime.device import enable_cuda_optimizations, pick_device

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
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker processes (default: min(4, cpu_count/2)). "
            "Set to 0 to disable prefetching."
        ),
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
        default="auto",
        help=(
            "Device override ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'). "
            "Default 'auto' picks CUDA when available, else MPS, else CPU."
        ),
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Mixed-precision training. 'auto' enables AMP on CUDA/MPS and "
            "disables it on CPU."
        ),
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        dest="compile_mode",
        help=(
            "torch.compile() JIT the model graph.  'auto' = on when CUDA "
            "is visible and the running torch supports it, else off.  "
            "Adds ~30 s warm-up on the first step but yields a 1.2-1.6x "
            "steady-state speedup on Ampere+ GPUs."
        ),
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

    # PERF: flip on cuDNN benchmark + TF32 fast matmul when CUDA is visible.
    # Safe no-op on CPU-only boxes, so it lives unconditionally here.
    enable_cuda_optimizations()

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
    # Detection priority lives in i3.runtime.device.pick_device: explicit
    # override > CUDA > MPS (Apple Silicon) > CPU.
    device = pick_device(args.device)
    # AMP is enabled on CUDA/MPS by default and skipped on CPU (where it
    # would be a slowdown / numeric change).
    if args.amp == "on":
        amp_enabled = device.type in {"cuda", "mps"}
    elif args.amp == "off":
        amp_enabled = False
    else:  # auto
        amp_enabled = device.type in {"cuda", "mps"}
    logger.info("Device: %s  (AMP: %s)", device, "on" if amp_enabled else "off")

    # -- Data -----------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_ds = load_split(data_dir / "train.pt")
    val_ds = load_split(data_dir / "val.pt")

    # PERF: DataLoader prefetching — ``num_workers`` spawns sidecar
    # Python processes that load+collate the next batch while the
    # current one is on the GPU/CPU.  Defaults:
    #   * ``--num-workers`` CLI override wins outright
    #   * On CUDA we want the GPU *never* idle, so we scale up to 8 or
    #     (cpu_count - 2) whichever is smaller — keeps 2 cores free for
    #     the main loop + Python overhead.
    #   * On CPU we keep it tight at ``min(4, cpu_count/2)`` — the CPU
    #     is *also* the trainer, contending with workers costs more
    #     than the overlap gains.
    #   * ``persistent_workers=True`` keeps the pool alive across
    #     epochs so fork/spawn cost is paid once per run, not per epoch
    #   * ``pin_memory`` only when CUDA is available (a no-op otherwise)
    #   * ``prefetch_factor`` bumped on CUDA — keeps 4 batches queued
    #     per worker so the GPU never waits on I/O
    import torch as _torch
    _cuda = _torch.cuda.is_available()
    _num_workers = getattr(args, "num_workers", None)
    if _num_workers is None:
        if _cuda:
            _num_workers = min(8, max(0, (os.cpu_count() or 4) - 2))
        else:
            _num_workers = max(0, min(4, (os.cpu_count() or 2) // 2))
    _pin = _cuda
    _prefetch = 4 if (_cuda and _num_workers > 0) else 2
    _dl_kwargs: dict[str, Any] = dict(
        num_workers=_num_workers,
        pin_memory=_pin,
    )
    if _num_workers > 0:
        _dl_kwargs.update(persistent_workers=True, prefetch_factor=_prefetch)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **_dl_kwargs,
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

    # PERF: torch.compile JITs the forward graph on first use and caches
    # the Inductor / NVIDIA kernels.  Only auto-enabled when Triton is
    # importable (Linux) — Windows has no official Triton wheel so we
    # silently skip compile and keep AMP + TF32 for the speedup we can
    # deliver.  Users can force-on with ``--compile on``.
    def _triton_available() -> bool:
        try:
            import triton  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return False
        return True

    _want_compile = args.compile_mode == "on" or (
        args.compile_mode == "auto"
        and device.type == "cuda"
        and _triton_available()
    )
    if _want_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            logger.info("torch.compile enabled (mode=default)")
        except Exception as exc:  # pragma: no cover - environment-specific
            logger.warning("torch.compile failed (%s); continuing uncompiled.", exc)
    elif args.compile_mode == "auto" and device.type == "cuda" and not _triton_available():
        logger.info(
            "torch.compile skipped: Triton not available (common on Windows). "
            "AMP + TF32 still active."
        )

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
        amp_enabled=amp_enabled,
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
            num_workers=_num_workers,
            persistent_workers=_num_workers > 0,
            pin_memory=_pin,
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
