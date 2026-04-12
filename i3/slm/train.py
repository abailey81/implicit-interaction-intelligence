"""Training loop for the Adaptive SLM with conditioning.

Built from scratch -- no HuggingFace Trainer, Accelerate, or external
training frameworks. Implements a complete training pipeline with:

- Cosine learning rate schedule with linear warmup
- Gradient clipping
- Cross-entropy loss with padding masking
- Validation with perplexity tracking
- Periodic checkpointing
- Best-model tracking with early stopping

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer
    from i3.slm.train import SLMTrainer

    model = AdaptiveSLM()
    tokenizer = SimpleTokenizer.load("models/slm/tokenizer.json")
    trainer = SLMTrainer(model, tokenizer, config)
    results = trainer.train(train_loader, val_loader, max_steps=50000)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Cosine annealing learning rate schedule with linear warmup.

    During the warmup phase (steps 0 to ``warmup_steps``), the learning
    rate increases linearly from 0 to ``base_lr``. After warmup, it
    follows a cosine decay from ``base_lr`` to ``min_lr``.

    .. math::

        \\text{lr}(t) = \\begin{cases}
            \\text{base\\_lr} \\cdot \\frac{t}{\\text{warmup\\_steps}}
                & \\text{if } t < \\text{warmup\\_steps} \\\\
            \\text{min\\_lr} + \\frac{1}{2}(\\text{base\\_lr} - \\text{min\\_lr})
                \\left(1 + \\cos\\left(\\pi \\cdot \\frac{t - \\text{warmup\\_steps}}
                {\\text{max\\_steps} - \\text{warmup\\_steps}}\\right)\\right)
                & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate is controlled.
    warmup_steps : int
        Number of linear warmup steps.
    max_steps : int
        Total number of training steps for the cosine schedule.
    base_lr : float
        Peak learning rate reached after warmup.
    min_lr : float
        Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._step_count: int = 0
        self._last_lr: list[float] = [base_lr]

    def step(self) -> None:
        """Advance the schedule by one step and update the learning rate."""
        self._step_count += 1
        lr = self._compute_lr(self._step_count)
        self._last_lr = [lr]

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        """Compute the learning rate for a given step.

        Parameters
        ----------
        step : int
            Current training step (1-indexed after first ``step()`` call).

        Returns
        -------
        float
            The learning rate for this step.
        """
        if step < self.warmup_steps:
            # Linear warmup: 0 -> base_lr
            return self.base_lr * step / max(self.warmup_steps, 1)
        else:
            # Cosine decay: base_lr -> min_lr
            decay_steps = max(self.max_steps - self.warmup_steps, 1)
            progress = min(
                (step - self.warmup_steps) / decay_steps, 1.0
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def get_last_lr(self) -> list[float]:
        """Return the last computed learning rate.

        Returns
        -------
        list[float]
            Single-element list containing the current learning rate.
        """
        return self._last_lr


class SLMTrainer:
    """Training loop for the Adaptive SLM with conditioning.

    Handles the complete training pipeline including optimiser setup,
    learning rate scheduling, gradient clipping, loss computation,
    validation, checkpointing, and logging.

    Parameters
    ----------
    model : AdaptiveSLM
        The model to train.
    tokenizer : SimpleTokenizer
        Tokenizer (used for PAD_ID and vocab info).
    config : Config
        Full I3 configuration object.
    device : str
        Device to train on (default ``"cpu"``).
    checkpoint_dir : str
        Directory for saving checkpoints.

    Attributes
    ----------
    optimizer : torch.optim.AdamW
        AdamW optimiser with weight decay.
    scheduler : CosineWarmupScheduler
        Learning rate schedule with warmup.
    global_step : int
        Current training step.
    best_val_loss : float
        Best validation loss seen so far.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        device: str = "cpu",
        checkpoint_dir: str = "models/slm",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Training config shortcuts
        self.train_cfg = config.slm.training

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.train_cfg.warmup_steps,
            max_steps=self.train_cfg.max_steps,
            base_lr=self.train_cfg.learning_rate,
        )

        # Training state
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch dictionary with keys:

            - ``"input_ids"``    -- ``[batch, seq_len]`` token IDs
            - ``"target_ids"``   -- ``[batch, seq_len]`` target token IDs
            - ``"conditioning"`` -- ``[batch, 8]`` AdaptationVector
            - ``"user_state"``   -- ``[batch, 64]`` UserStateEmbedding
            - ``"attention_mask"`` -- ``[batch, seq_len]`` (optional)

        Returns
        -------
        dict[str, float]
            Dictionary with keys:

            - ``"loss"`` -- scalar cross-entropy loss
            - ``"lr"`` -- current learning rate
            - ``"grad_norm"`` -- gradient L2 norm before clipping
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        conditioning = batch["conditioning"].to(self.device)
        user_state = batch["user_state"].to(self.device)

        # Forward pass
        logits, _ = self.model(
            input_ids, conditioning, user_state, use_cache=False
        )

        # Cross-entropy loss on shifted targets, ignoring padding
        # logits[:, :-1] predicts target_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.model.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.tokenizer.PAD_ID,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm (before clipping)
        grad_norm = self._compute_grad_norm()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_cfg.gradient_clip,
        )

        # Optimiser step + LR schedule step
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Compute validation loss and perplexity.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader yielding batch dicts.

        Returns
        -------
        dict[str, float]
            Dictionary with keys:

            - ``"val_loss"`` -- average cross-entropy loss
            - ``"perplexity"`` -- exp(val_loss)
            - ``"num_batches"`` -- number of validation batches
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            conditioning = batch["conditioning"].to(self.device)
            user_state = batch["user_state"].to(self.device)

            logits, _ = self.model(
                input_ids, conditioning, user_state, use_cache=False
            )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.model.vocab_size),
                shift_targets.view(-1),
                ignore_index=self.tokenizer.PAD_ID,
            )

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 100.0))  # Cap to avoid overflow

        self.model.train()

        return {
            "val_loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": n_batches,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_steps: Optional[int] = None,
        log_every: int = 100,
        validate_every: int = 1000,
        patience: int = 10,
    ) -> dict[str, Any]:
        """Run the full training loop with validation and checkpointing.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding batch dicts.
        val_loader : DataLoader
            Validation data loader.
        max_steps : int, optional
            Maximum training steps. Defaults to ``config.slm.training.max_steps``.
        log_every : int
            Log training metrics every N steps (default 100).
        validate_every : int
            Run validation every N steps (default 1000).
        patience : int
            Number of validation rounds without improvement before early
            stopping (default 10).

        Returns
        -------
        dict[str, Any]
            Training results with keys:

            - ``"final_step"`` -- last training step
            - ``"best_val_loss"`` -- best validation loss
            - ``"best_step"`` -- step of best validation loss
            - ``"train_losses"`` -- list of (step, loss) tuples
            - ``"val_losses"`` -- list of (step, val_loss, ppl) tuples
            - ``"total_time_s"`` -- total training time in seconds
        """
        if max_steps is None:
            max_steps = self.train_cfg.max_steps

        checkpoint_every = self.train_cfg.checkpoint_every

        best_step = 0
        patience_counter = 0
        train_loss_log: list[tuple[int, float]] = []
        val_loss_log: list[tuple[int, float, float]] = []

        logger.info(
            "Starting SLM training: max_steps=%d, lr=%.2e, batch_size=%d",
            max_steps,
            self.train_cfg.learning_rate,
            self.train_cfg.batch_size,
        )

        t_start = time.time()
        data_iter = iter(train_loader)

        while self.global_step < max_steps:
            # Get next batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Train step
            metrics = self.train_step(batch)
            self.train_losses.append(metrics["loss"])
            train_loss_log.append((self.global_step, metrics["loss"]))

            # Logging
            if self.global_step % log_every == 0:
                avg_recent = sum(self.train_losses[-log_every:]) / min(
                    len(self.train_losses), log_every
                )
                logger.info(
                    "Step %d/%d  loss=%.4f  avg_loss=%.4f  lr=%.2e  grad_norm=%.4f",
                    self.global_step,
                    max_steps,
                    metrics["loss"],
                    avg_recent,
                    metrics["lr"],
                    metrics["grad_norm"],
                )

            # Validation
            if self.global_step % validate_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics["val_loss"]
                ppl = val_metrics["perplexity"]
                self.val_losses.append(val_loss)
                val_loss_log.append((self.global_step, val_loss, ppl))

                logger.info(
                    "  Validation: loss=%.4f  ppl=%.2f  (best=%.4f)",
                    val_loss,
                    ppl,
                    self.best_val_loss,
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_step = self.global_step
                    patience_counter = 0
                    self._save_checkpoint("best_model.pt")
                    logger.info("  New best model saved at step %d", best_step)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            "Early stopping at step %d (patience=%d)",
                            self.global_step,
                            patience,
                        )
                        break

            # Periodic checkpoint
            if (
                checkpoint_every > 0
                and self.global_step % checkpoint_every == 0
            ):
                self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        total_time = time.time() - t_start

        # Final checkpoint
        self._save_checkpoint("final_model.pt")

        results = {
            "final_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_step": best_step,
            "train_losses": train_loss_log,
            "val_losses": val_loss_log,
            "total_time_s": total_time,
        }

        logger.info(
            "Training complete: %d steps in %.1f min, best_loss=%.4f at step %d",
            self.global_step,
            total_time / 60.0,
            self.best_val_loss,
            best_step,
        )

        return results

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        filename : str
            Filename within ``self.checkpoint_dir``.
        """
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler._step_count,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses[-1000:],  # Keep last 1000
            "val_losses": self.val_losses[-100:],
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        """
        # SECURITY NOTE: Resuming training requires loading the optimizer
        # state, which is a pickled Python object, so weights_only=True
        # is not usable here.  This code path is intended for trusted
        # local checkpoints produced by *our own* trainer.  Do NOT point
        # it at untrusted files.  Inference-time loads elsewhere in the
        # codebase use weights_only=True.
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        # Restore scheduler state
        scheduler_steps = checkpoint.get("scheduler_step_count", self.global_step)
        self.scheduler._step_count = scheduler_steps

        logger.info(
            "Resumed from checkpoint: step=%d, best_val_loss=%.4f",
            self.global_step,
            self.best_val_loss,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _compute_grad_norm(self) -> float:
        """Compute the L2 norm of all parameter gradients.

        Returns
        -------
        float
            Total gradient L2 norm.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
