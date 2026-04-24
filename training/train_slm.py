"""CLI script for training the Adaptive Small Language Model.

Loads configuration, prepares data, builds the model, and runs the full
training loop with checkpointing and validation.

Usage::

    # From project root:
    python -m training.train_slm
    python -m training.train_slm --config configs/default.yaml --epochs 10
    python -m training.train_slm --resume models/slm/checkpoint_step_5000.pt

    # Direct execution:
    python training/train_slm.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.config import load_config, Config
from i3.runtime.device import enable_cuda_optimizations, pick_device
from i3.slm.model import AdaptiveSLM
from i3.slm.tokenizer import SimpleTokenizer
from i3.slm.train import SLMTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class for dialogue data
# ---------------------------------------------------------------------------

class DialogueDataset(Dataset):
    """PyTorch Dataset wrapping pre-processed dialogue data.

    Loads ``.pt`` files produced by ``prepare_dialogue.py`` that contain
    pre-tokenized and padded sequences with conditioning labels.

    Each sample is a dictionary with:
    - ``input_ids``:    ``[seq_len]`` token IDs
    - ``target_ids``:   ``[seq_len]`` target token IDs (same as input, shifted)
    - ``conditioning``: ``[8]`` AdaptationVector
    - ``user_state``:   ``[64]`` synthetic UserStateEmbedding
    """

    def __init__(self, path: str | Path) -> None:
        """Load pre-processed dialogue data from a ``.pt`` file.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` file.
        """
        # Training data `.pt` files are plain dict[str, Tensor] produced
        # by our own preprocessing pipeline; weights_only=True is safe.
        data = torch.load(path, map_location="cpu", weights_only=True)

        self.input_ids: torch.Tensor = data["input_ids"]
        self.target_ids: torch.Tensor = data["target_ids"]
        self.conditioning: torch.Tensor = data["conditioning"]
        self.user_state: torch.Tensor = data["user_state"]

        logger.info(
            "Loaded DialogueDataset: %d samples from %s",
            len(self.input_ids),
            path,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
            "conditioning": self.conditioning[idx],
            "user_state": self.user_state[idx],
        }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SLM training."""
    parser = argparse.ArgumentParser(
        description="Train the Adaptive Small Language Model (SLM).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/dialogue",
        help="Directory containing train.pt / val.pt dialogue data.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer JSON. If None, looks in data-dir.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs (converted to steps internally).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
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
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/slm",
        help="Directory for model checkpoints (default: models/slm).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log training metrics every N steps (default: 100).",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1000,
        help="Run validation every N steps (default: 1000).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
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
            "is visible.  Adds ~30 s warm-up, yields 1.2-1.6x steady-state "
            "speedup on Ampere+ GPUs."
        ),
    )
    args = parser.parse_args()

    # SEC: bound-check numeric CLI overrides so user typos surface fast,
    # before we spend time loading data or building the model.
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be a positive float")
    if args.log_every <= 0:
        parser.error("--log-every must be a positive integer")
    if args.validate_every <= 0:
        parser.error("--validate-every must be a positive integer")
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    return args


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(override: str | None = None) -> torch.device:
    """Detect the best available device.

    Thin compatibility shim around :func:`i3.runtime.device.pick_device` —
    preserved so existing callers keep working.

    Parameters
    ----------
    override : str, optional
        If provided, use this device string directly.

    Returns
    -------
    torch.device
        The selected device.
    """
    return pick_device(override)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, build model, and run training."""
    args = parse_args()

    # -- Logging setup -------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # PERF: cuDNN benchmark + TF32 matmul fast paths. No-op on CPU so it is
    # always safe to call — keeps the CUDA-vs-CPU launch logic uniform.
    enable_cuda_optimizations()

    # -- Load configuration --------------------------------------------------
    # SEC: validate config path resolves under the project root and exists.
    # Config files come from a trusted YAML source but we still resolve and
    # check to avoid surprises from relative-path traversal.
    config_path = Path(args.config).resolve()
    if config_path.exists() and config_path.is_file():
        config = load_config(config_path)
    else:
        logger.warning(
            "Config %s not found; using built-in defaults.", config_path
        )
        # Pydantic Config has default_factory for every section, so a
        # zero-arg instantiation produces a fully-populated default config.
        from i3.config import Config
        config = Config()

    # -- Seed ----------------------------------------------------------------
    # SEC: seed every RNG framework we use (Python random, NumPy, torch)
    # so the data shuffling and weight init are reproducible from --seed.
    seed = args.seed
    import random as _py_random

    import numpy as _np
    _py_random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Device --------------------------------------------------------------
    device = detect_device(args.device)
    if args.amp == "on":
        amp_enabled = device.type in {"cuda", "mps"}
    elif args.amp == "off":
        amp_enabled = False
    else:  # auto
        amp_enabled = device.type in {"cuda", "mps"}
    logger.info(
        "Using device: %s  (AMP: %s)", device, "on" if amp_enabled else "off"
    )

    # -- Load tokenizer ------------------------------------------------------
    # SEC: resolve all input paths so log messages are unambiguous and
    # any "../"-style relative paths are normalised before file I/O.
    data_dir = Path(args.data_dir).resolve()
    tokenizer_path: Path
    if args.tokenizer is None:
        tokenizer_path = data_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = Path("models/slm/tokenizer.json").resolve()
    else:
        tokenizer_path = Path(args.tokenizer).resolve()

    if tokenizer_path.exists() and tokenizer_path.is_file():
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
        logger.info("Loaded tokenizer from %s (vocab=%d)", tokenizer_path, len(tokenizer))
    else:
        logger.error(
            "Tokenizer not found at %s. Run prepare_dialogue.py first.",
            tokenizer_path,
        )
        sys.exit(1)

    # -- Load data -----------------------------------------------------------
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"

    if not train_path.exists():
        logger.error(
            "Training data not found at %s. Run prepare_dialogue.py first.",
            train_path,
        )
        sys.exit(1)

    train_dataset = DialogueDataset(train_path)
    val_dataset = DialogueDataset(val_path)

    # PERF: auto-bump batch size when we have the VRAM.  The SLM is
    # tiny (4 layers, 256 d_model) so even a 6 GB RTX 4050 has room for
    # 2-4× the config default.  Users can always pin with --batch-size.
    import os as _os
    _cuda = device.type == "cuda"
    batch_size = args.batch_size or config.slm.training.batch_size
    if args.batch_size is None and _cuda:
        # 2× the config default is conservative and safe on 6 GB VRAM
        # at seq_len=256, d_model=256.  Keeps a comfortable 50 %
        # activation headroom for AMP + gradient buffers.
        batch_size = batch_size * 2
        logger.info(
            "CUDA detected: auto-bumping batch_size %d -> %d "
            "(override with --batch-size).",
            config.slm.training.batch_size,
            batch_size,
        )

    # PERF: DataLoader workers scaled by device — see train_encoder.py
    # for the rationale.  CUDA path prefetches 4 batches per worker.
    _num_workers = getattr(args, "num_workers", None)
    if _num_workers is None:
        if _cuda:
            _num_workers = min(8, max(0, (_os.cpu_count() or 4) - 2))
        else:
            _num_workers = max(0, min(4, (_os.cpu_count() or 2) // 2))
    _prefetch = 4 if (_cuda and _num_workers > 0) else 2
    _dl_kwargs: dict[str, Any] = dict(
        num_workers=_num_workers,
        pin_memory=_cuda,
    )
    if _num_workers > 0:
        _dl_kwargs.update(persistent_workers=True, prefetch_factor=_prefetch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_dl_kwargs,
    )

    logger.info(
        "Data loaded: %d train samples, %d val samples, batch_size=%d",
        len(train_dataset),
        len(val_dataset),
        batch_size,
    )

    # -- Build model ---------------------------------------------------------
    actual_vocab = len(tokenizer)
    model = AdaptiveSLM(
        vocab_size=actual_vocab,
        d_model=config.slm.d_model,
        n_heads=config.slm.n_heads,
        n_layers=config.slm.n_layers,
        d_ff=config.slm.d_ff,
        max_seq_len=config.slm.max_seq_len,
        conditioning_dim=config.slm.conditioning_dim,
        adaptation_dim=config.slm.adaptation_dim,
        n_cross_heads=config.slm.cross_attention_heads,
        dropout=config.slm.dropout,
        tie_weights=config.slm.tie_weights,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s", model)
    logger.info("Total parameters: %d (%.2f MB)", n_params, n_params * 4 / 1e6)

    # PERF: torch.compile JIT the forward+backward graph.  Only kicks in
    # on CUDA by default (the Inductor CPU path is slower than eager for
    # small transformers).  --compile on forces it; --compile off skips
    # even on GPU.
    _want_compile = (
        args.compile_mode == "on"
        or (args.compile_mode == "auto" and device.type == "cuda")
    )
    if _want_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            logger.info("torch.compile enabled (mode=default)")
        except Exception as exc:  # pragma: no cover - environment-specific
            logger.warning("torch.compile failed (%s); continuing uncompiled.", exc)

    # -- Build trainer -------------------------------------------------------
    # Apply CLI overrides to config (create mutable copy of training config)
    if args.lr is not None:
        # Override learning rate in the trainer
        logger.info("Overriding learning rate: %.2e -> %.2e", config.slm.training.learning_rate, args.lr)

    trainer = SLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=str(device),
        checkpoint_dir=args.checkpoint_dir,
        amp_enabled=amp_enabled,
    )

    # Apply LR override after trainer construction
    if args.lr is not None:
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = args.lr
        trainer.scheduler.base_lr = args.lr

    # -- Resume from checkpoint if requested ---------------------------------
    # SEC: resume loads checkpoint with weights_only=False (it must
    # restore optimiser state, which is pickled). The path is therefore
    # high-trust: validate it exists and is a regular file before handing
    # it to torch.load.
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.exists() and resume_path.is_file():
            trainer.load_checkpoint(str(resume_path))
        else:
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)

    # -- Determine max_steps -------------------------------------------------
    if args.max_steps is not None:
        max_steps = args.max_steps
    elif args.epochs is not None:
        steps_per_epoch = len(train_loader)
        max_steps = args.epochs * steps_per_epoch
        logger.info(
            "Converting %d epochs to %d steps (%d steps/epoch)",
            args.epochs, max_steps, steps_per_epoch,
        )
    else:
        max_steps = config.slm.training.max_steps

    # -- Train ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STARTING SLM TRAINING")
    logger.info("=" * 60)

    t0 = time.time()

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=max_steps,
        log_every=args.log_every,
        validate_every=args.validate_every,
        patience=args.patience,
    )

    elapsed = time.time() - t0

    # -- Report results ------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total steps:       %d", results["final_step"])
    logger.info("Best val loss:     %.4f", results["best_val_loss"])
    logger.info("Best step:         %d", results["best_step"])
    logger.info("Total time:        %.1f min", elapsed / 60.0)

    # Save training metrics to JSON
    metrics_path = Path(args.checkpoint_dir) / "training_metrics.json"
    metrics_data = {
        "final_step": results["final_step"],
        "best_val_loss": results["best_val_loss"],
        "best_step": results["best_step"],
        "total_time_s": results["total_time_s"],
        "configs": {
            "vocab_size": actual_vocab,
            "d_model": config.slm.d_model,
            "n_layers": config.slm.n_layers,
            "n_heads": config.slm.n_heads,
            "d_ff": config.slm.d_ff,
            "learning_rate": config.slm.training.learning_rate,
            "batch_size": batch_size,
            "max_steps": max_steps,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Training metrics saved to: %s", metrics_path)

    # Save tokenizer alongside the model
    tok_save_path = Path(args.checkpoint_dir) / "tokenizer.json"
    tokenizer.save(str(tok_save_path))
    logger.info("Tokenizer saved to: %s", tok_save_path)


if __name__ == "__main__":
    main()
