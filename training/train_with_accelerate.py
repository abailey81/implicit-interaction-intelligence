"""HuggingFace Accelerate-based distributed training for the I3 SLM.

This is a *parallel* implementation to :mod:`training.train_slm_fabric`: both
achieve the same goal (scale the Adaptive SLM across multiple devices) but via
different libraries. Users can pick whichever fits their deployment:

* **Accelerate** is the right choice when the rest of the team is already on
  a HuggingFace stack and the training script must interop with
  ``transformers``.
* **Fabric** is the right choice when we want to stay closer to vanilla
  PyTorch with fewer abstractions.

Soft-imports ``accelerate``. Without it this script still imports cleanly and
the CLI prints a friendly message.

Usage
-----

.. code-block:: bash

    accelerate config  # one-time
    accelerate launch training/train_with_accelerate.py --epochs 5

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.slm.model import AdaptiveSLM

# ---------------------------------------------------------------------------
# Soft-import accelerate
# ---------------------------------------------------------------------------

try:
    from accelerate import Accelerator  # type: ignore[import-not-found]

    _ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None  # type: ignore[assignment,misc]
    _ACCELERATE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (same layout as train_slm_fabric.py, duplicated to avoid a
# cross-script dependency)
# ---------------------------------------------------------------------------


class _AccelerateDialogueDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper for dialogue ``.pt`` files.

    Attributes:
        input_ids: Token IDs ``[N, seq_len]``.
        target_ids: Target IDs ``[N, seq_len]``.
        conditioning: AdaptationVector ``[N, 8]``.
        user_state: UserStateEmbedding ``[N, 64]``.
    """

    def __init__(self, path: Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.input_ids: torch.Tensor = data["input_ids"]
        self.target_ids: torch.Tensor = data["target_ids"]
        self.conditioning: torch.Tensor = data["conditioning"]
        self.user_state: torch.Tensor = data["user_state"]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
            "conditioning": self.conditioning[idx],
            "user_state": self.user_state[idx],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the cross-entropy loss for an SLM batch.

    Args:
        model: The accelerate-wrapped SLM.
        batch: Dict of input tensors.

    Returns:
        A scalar loss tensor.
    """
    logits = model(
        input_ids=batch["input_ids"],
        adaptation_vector=batch["conditioning"],
        user_state=batch["user_state"],
    )
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch["target_ids"].reshape(-1),
        ignore_index=-100,
    )


def run_with_accelerate(args: argparse.Namespace) -> dict[str, Any]:
    """Train the SLM using HuggingFace Accelerate.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A dict with ``best_epoch`` and ``best_val_loss``.
    """
    if not _ACCELERATE_AVAILABLE:
        logger.error(
            "accelerate is not installed. Install via "
            "`poetry install --with distributed`."
        )
        return {"best_epoch": -1, "best_val_loss": float("nan")}

    assert Accelerator is not None  # for mypy after guard
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    accelerator.print(f"Accelerate state: {accelerator.state}")

    # Model + optimizer
    model = AdaptiveSLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Data
    data_dir = Path(args.data_dir)
    train_ds = _AccelerateDialogueDataset(data_dir / "train.pt")
    val_ds = _AccelerateDialogueDataset(data_dir / "val.pt")
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # Distribute
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        for step_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss = _forward_loss(model, batch)
                accelerator.backward(loss)
                if args.grad_clip > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            count += 1
            if step_idx % 100 == 0 and accelerator.is_main_process:
                logger.info(
                    "epoch=%d step=%d loss=%.4f", epoch, step_idx, float(loss.detach())
                )
        train_loss = running / max(count, 1)

        model.eval()
        v_total = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                v_total += float(_forward_loss(model, batch).detach())
                v_n += 1
        val_loss = v_total / max(v_n, 1)

        if accelerator.is_main_process:
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            if accelerator.is_main_process:
                accelerator.save_state(str(ckpt_dir / f"epoch_{epoch:03d}"))

    return {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv override.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the I3 SLM with HuggingFace Accelerate."
    )
    parser.add_argument("--data-dir", type=str, default="data/dialogue")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/slm/accelerate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    return args


def main() -> None:
    """Script entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    args = _parse_args()
    results = run_with_accelerate(args)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
