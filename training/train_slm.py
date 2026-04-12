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
        default=None,
        help="Device override (cpu, cuda, mps). Auto-detects if not set.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(override: str | None = None) -> torch.device:
    """Detect the best available device.

    Parameters
    ----------
    override : str, optional
        If provided, use this device string directly.

    Returns
    -------
    torch.device
        The selected device.
    """
    if override:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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

    # -- Load configuration --------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(
            "Config %s not found; using defaults.", config_path
        )
        config = load_config.__wrapped__ if hasattr(load_config, '__wrapped__') else None
        # Fallback: create a default Config
        from i3.config import Config
        config = Config()

    # -- Seed ----------------------------------------------------------------
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Device --------------------------------------------------------------
    device = detect_device(args.device)
    logger.info("Using device: %s", device)

    # -- Load tokenizer ------------------------------------------------------
    data_dir = Path(args.data_dir)
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = data_dir / "tokenizer.json"
        if not Path(tokenizer_path).exists():
            tokenizer_path = "models/slm/tokenizer.json"

    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
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

    batch_size = args.batch_size or config.slm.training.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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
    )

    # Apply LR override after trainer construction
    if args.lr is not None:
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = args.lr
        trainer.scheduler.base_lr = args.lr

    # -- Resume from checkpoint if requested ---------------------------------
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
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
