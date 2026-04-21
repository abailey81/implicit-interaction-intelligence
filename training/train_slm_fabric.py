"""Fabric-based FSDP training for the Adaptive SLM.

Wraps the :class:`i3.slm.model.AdaptiveSLM` in ``lightning.Fabric`` with
``strategy='fsdp'`` so the 6-8 M-parameter model can be scaled, together with
its optimizer state, across multiple GPUs. The SLM itself is small enough to
fit on a single accelerator, but FSDP becomes essential when

* we retrain on a larger curated corpus (domain-specific conversational data);
* we experiment with wider or deeper variants;
* we colocate training with a larger companion model.

The training step is compiled with :func:`torch.compile` in
``mode='max-autotune'`` to squeeze the most out of the backend; compile
failures fall through to eager without aborting the run.

Lightning is soft-imported. Without it the script exits with a clear message
and does not raise a bare ``ImportError`` at import time, so downstream
tooling (docs builds, static analysis) keeps working.

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.slm.model import AdaptiveSLM

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
# Dataset helper
# ---------------------------------------------------------------------------


class _FabricDialogueDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Minimal dataset wrapper mirroring ``training.train_slm.DialogueDataset``.

    Attributes:
        input_ids: ``[N, seq_len]`` token IDs.
        target_ids: ``[N, seq_len]`` target IDs.
        conditioning: ``[N, 8]`` AdaptationVector.
        user_state: ``[N, 64]`` UserStateEmbedding.
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
# Training helpers
# ---------------------------------------------------------------------------


def _maybe_compile(model: torch.nn.Module, enable: bool) -> torch.nn.Module:
    """Apply ``torch.compile(mode='max-autotune')`` if requested.

    Args:
        model: The model to compile.
        enable: Whether compilation is requested.

    Returns:
        The (possibly compiled) model.
    """
    if not enable:
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable on this build; running eager.")
        return model
    try:
        return torch.compile(model, mode="max-autotune")  # type: ignore[return-value]
    except (RuntimeError, ValueError) as exc:
        logger.warning("torch.compile failed (%s); running eager.", exc)
        return model


def _step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run a single forward step and return the cross-entropy loss.

    Args:
        model: The SLM (wrapped under Fabric/FSDP).
        batch: Dict with ``input_ids``, ``target_ids``, ``conditioning``,
            ``user_state``.

    Returns:
        Scalar loss tensor.
    """
    logits = model(
        input_ids=batch["input_ids"],
        adaptation_vector=batch["conditioning"],
        user_state=batch["user_state"],
    )
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch["target_ids"].reshape(-1),
        ignore_index=-100,
    )
    return loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_fabric_slm(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Fabric+FSDP SLM training.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict with ``best_epoch`` and ``best_val_loss``.
    """
    if not _LIGHTNING_AVAILABLE:
        logger.error(
            "lightning is not installed; Fabric SLM training requires it. "
            "Install via `poetry install --with distributed`."
        )
        return {"best_epoch": -1, "best_val_loss": float("nan")}

    from lightning.fabric import Fabric  # type: ignore[import-not-found]

    fabric = Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
    )
    fabric.launch()
    fabric.seed_everything(args.seed)

    # Model
    model = AdaptiveSLM()
    model = _maybe_compile(model, enable=args.compile)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    model, optimizer = fabric.setup(model, optimizer)

    # Data
    data_dir = Path(args.data_dir)
    train_ds = _FabricDialogueDataset(data_dir / "train.pt")
    val_ds = _FabricDialogueDataset(data_dir / "val.pt")
    train_loader = fabric.setup_dataloaders(
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    )
    val_loader = fabric.setup_dataloaders(
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for step_idx, batch in enumerate(train_loader):
            loss = _step(model, batch)
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            if args.grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            n += 1
            if step_idx % 100 == 0 and fabric.is_global_zero:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step_idx, float(loss.detach()))

        train_loss = total / max(n, 1)

        # Validation
        model.eval()
        v_total = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                v_total += float(_step(model, batch).detach())
                v_n += 1
        val_loss = v_total / max(v_n, 1)

        fabric.log_dict({"train/loss": train_loss, "val/loss": val_loss}, step=epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            state = {"model": model, "optimizer": optimizer, "epoch": epoch}
            fabric.save(str(ckpt_dir / "best_model.pt"), state)

    return {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argv.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Fabric+FSDP training for the I3 SLM.")
    parser.add_argument("--data-dir", type=str, default="data/dialogue")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/slm/fabric")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="fsdp")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
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
    results = run_fabric_slm(args)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
