"""DeepSpeed ZeRO-3 training for the I3 SLM.

DeepSpeed adds ZeRO (Zero Redundancy Optimizer) on top of the standard
PyTorch DistributedDataParallel contract. Stage 3 shards optimizer state,
gradients, *and* parameters, which lets us train models much larger than any
single GPU's memory. For the I3 SLM (currently ~6-8M params) this is
overkill, but we include the wiring for the *planned* scale-up paths
discussed in ``docs/research/distributed_training.md``:

* wider SLM variants (~100M-1B params);
* joint training of the SLM and the planned long-term-memory compressor.

If a ZeRO-3 config is not found at ``configs/distributed/ds_config_zero3.json``
this script writes a sensible default at runtime so a fresh clone can launch
immediately.

Soft-imports ``deepspeed`` — without it the script exits cleanly.

Usage
-----

.. code-block:: bash

    deepspeed --num_gpus=4 training/train_with_deepspeed.py --epochs 3

"""

from __future__ import annotations

import argparse
import json
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
# Soft-import deepspeed
# ---------------------------------------------------------------------------

try:
    import deepspeed  # type: ignore[import-not-found]

    _DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None  # type: ignore[assignment]
    _DEEPSPEED_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DS_CONFIG: dict[str, Any] = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1_000_000_000,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1_000_000_000,
        "stage3_max_reuse_distance": 1_000_000_000,
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": False,
        "cpu_checkpointing": False,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-2},
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 500,
        },
    },
}


def ensure_ds_config(path: Path) -> Path:
    """Ensure a DeepSpeed config file exists, writing the default if missing.

    Args:
        path: Desired config path.

    Returns:
        The resolved path (existing or newly created).
    """
    path = path.resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_DEFAULT_DS_CONFIG, f, indent=2)
    logger.info("Wrote default DeepSpeed config to %s", path)
    return path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _DSDialogueDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper for dialogue ``.pt`` files.

    Attributes:
        input_ids: ``[N, seq_len]``.
        target_ids: ``[N, seq_len]``.
        conditioning: ``[N, 8]``.
        user_state: ``[N, 64]``.
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


def run_with_deepspeed(args: argparse.Namespace) -> dict[str, Any]:
    """Train the SLM using DeepSpeed ZeRO-3.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict with ``best_epoch`` and ``best_val_loss``.
    """
    if not _DEEPSPEED_AVAILABLE:
        logger.error(
            "deepspeed is not installed. Install via "
            "`poetry install --with distributed`."
        )
        return {"best_epoch": -1, "best_val_loss": float("nan")}

    assert deepspeed is not None
    ds_config_path = ensure_ds_config(Path(args.ds_config))

    model = AdaptiveSLM()
    model_engine, optimizer, train_loader_engine, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=str(ds_config_path),
    )

    data_dir = Path(args.data_dir)
    train_ds = _DSDialogueDataset(data_dir / "train.pt")
    val_ds = _DSDialogueDataset(data_dir / "val.pt")
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model_engine.train()
        running = 0.0
        n = 0
        for step_idx, batch in enumerate(train_loader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            logits = model_engine(
                input_ids=batch["input_ids"],
                adaptation_vector=batch["conditioning"],
                user_state=batch["user_state"],
            )
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch["target_ids"].reshape(-1),
                ignore_index=-100,
            )
            model_engine.backward(loss)
            model_engine.step()
            running += float(loss.detach())
            n += 1
            if step_idx % 100 == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step_idx, float(loss.detach()))
        train_loss = running / max(n, 1)

        # Validation
        model_engine.eval()
        v_total = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                logits = model_engine(
                    input_ids=batch["input_ids"],
                    adaptation_vector=batch["conditioning"],
                    user_state=batch["user_state"],
                )
                vloss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["target_ids"].reshape(-1),
                    ignore_index=-100,
                )
                v_total += float(vloss.detach())
                v_n += 1
        val_loss = v_total / max(v_n, 1)
        logger.info("epoch=%d train=%.4f val=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            model_engine.save_checkpoint(str(ckpt_dir), tag=f"epoch_{epoch:03d}")

    return {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments; accepts DeepSpeed extras via ``parse_known_args``.

    Args:
        argv: Optional argv.

    Returns:
        Parsed namespace (extras forwarded to DeepSpeed).
    """
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 training for the I3 SLM.")
    parser.add_argument("--data-dir", type=str, default="data/dialogue")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/slm/deepspeed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ds-config",
        type=str,
        default="configs/distributed/ds_config_zero3.json",
    )
    parser.add_argument("--local_rank", type=int, default=0)  # injected by deepspeed launcher
    args, _extras = parser.parse_known_args(argv)
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
    results = run_with_deepspeed(args)
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
