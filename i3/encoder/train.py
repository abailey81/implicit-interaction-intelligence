"""Training utilities for the User State Encoder TCN.

Provides the NT-Xent (InfoNCE) contrastive loss and a complete training loop
with AdamW, cosine annealing, gradient clipping, checkpointing, and early
stopping.  Validation metrics include silhouette score and k-NN accuracy.

All core logic is pure PyTorch; scikit-learn is used **only** for evaluation
metrics (silhouette_score, KNeighborsClassifier).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) contrastive loss
# ---------------------------------------------------------------------------


def contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent contrastive loss for user-state embedding learning.

    For each anchor embedding, other samples that share the same label are
    treated as positives and all remaining samples as negatives.  The loss
    pushes positive pairs together and negative pairs apart on the unit
    hypersphere.

    Args:
        embeddings: ``[batch, D]`` L2-normalised embeddings.
        labels:     ``[batch]`` integer state labels (e.g. 0-7 for 8 states).
        temperature: Softmax temperature scaling factor.

    Returns:
        Scalar loss tensor (mean over all valid positive pairs).
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # Pairwise cosine similarity (embeddings are already L2-normed)
    sim_matrix = torch.mm(embeddings, embeddings.t())  # [B, B]

    # Scale by temperature
    sim_matrix = sim_matrix / temperature

    # Mask: positive pairs share the same label, exclude self-pairs
    labels_col = labels.unsqueeze(1)  # [B, 1]
    labels_row = labels.unsqueeze(0)  # [1, B]
    positive_mask = (labels_col == labels_row).float()  # [B, B]
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask  # remove diagonal

    # If no positive pairs exist, return zero loss
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # For numerical stability, subtract max per row before exp
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Denominator: sum of exp over all pairs except self
    neg_mask = 1.0 - self_mask  # everything except diagonal
    exp_logits = torch.exp(logits) * neg_mask  # [B, B]
    log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # [B, 1]

    # Log-prob for each pair
    log_prob = logits - log_denom  # [B, B]

    # Average log-prob over positive pairs for each anchor
    # Sum of log-probs for positives / number of positives per anchor
    pos_per_anchor = positive_mask.sum(dim=1)  # [B]
    # Only consider anchors that have at least one positive
    valid = pos_per_anchor > 0

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (
        pos_per_anchor + 1e-12
    )

    # Average over valid anchors
    loss = -mean_log_prob_pos[valid].mean()
    return loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float = 0.07,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run a single training epoch.

    Args:
        model:       The TCN encoder (must return L2-normed embeddings).
        dataloader:  Yields ``(sequences, labels)`` batches.
        optimizer:   Optimiser instance (e.g. AdamW).
        temperature: NT-Xent temperature.
        grad_clip:   Max gradient norm for clipping.
        device:      Target device. ``None`` = auto-detect from model.

    Returns:
        Dict with keys ``"loss"`` (epoch mean) and ``"lr"`` (current LR).
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0
    n_batches = 0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        embeddings = model(sequences)
        loss = contrastive_loss(embeddings, labels, temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"loss": avg_loss, "lr": current_lr}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    temperature: float = 0.07,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate the encoder on a validation set.

    Computes contrastive loss, silhouette score, and k-NN accuracy (k=5).

    Args:
        model:       The TCN encoder.
        dataloader:  Yields ``(sequences, labels)`` batches.
        temperature: NT-Xent temperature.
        device:      Target device.

    Returns:
        Dict with ``"loss"``, ``"silhouette_score"``, and ``"knn_accuracy"``.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        embeddings = model(sequences)
        loss = contrastive_loss(embeddings, labels, temperature)

        total_loss += loss.item()
        n_batches += 1
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # Aggregate embeddings
    emb_np = torch.cat(all_embeddings, dim=0).numpy()
    lab_np = torch.cat(all_labels, dim=0).numpy()

    # Silhouette score (requires >= 2 distinct labels)
    n_unique = len(np.unique(lab_np))
    if n_unique >= 2 and len(lab_np) > n_unique:
        sil = float(silhouette_score(emb_np, lab_np))
    else:
        sil = 0.0

    # k-NN accuracy (k=5, leave-one-out style via cross-validation)
    if len(lab_np) >= 10:
        knn = KNeighborsClassifier(n_neighbors=min(5, len(lab_np) - 1))
        # Simple 50/50 split for speed
        mid = len(lab_np) // 2
        knn.fit(emb_np[:mid], lab_np[:mid])
        knn_acc = float(knn.score(emb_np[mid:], lab_np[mid:]))
    else:
        knn_acc = 0.0

    return {
        "loss": avg_loss,
        "silhouette_score": sil,
        "knn_accuracy": knn_acc,
    }


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    grad_clip: float = 1.0,
    checkpoint_dir: str | Path = "models/encoder",
    checkpoint_every: int = 10,
    patience: int = 15,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Full training procedure with scheduling, checkpointing, and early stop.

    Uses AdamW with CosineAnnealingLR.  Saves the best model (lowest
    validation loss) and periodic checkpoints.  Stops early if validation
    loss does not improve for ``patience`` consecutive epochs.

    Args:
        model:            TCN encoder instance.
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        epochs:           Maximum number of epochs.
        lr:               Peak learning rate.
        weight_decay:     AdamW weight-decay coefficient.
        temperature:      NT-Xent temperature.
        grad_clip:        Maximum gradient norm.
        checkpoint_dir:   Directory for model checkpoints.
        checkpoint_every: Save a checkpoint every N epochs.
        patience:         Early-stopping patience (epochs).
        device:           Target device (auto-detected if ``None``).

    Returns:
        Dict with ``"best_val_loss"``, ``"best_epoch"``, ``"final_metrics"``,
        and ``"history"`` (list of per-epoch metric dicts).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    logger.info(
        "Starting training: %d epochs, lr=%.1e, device=%s", epochs, lr, device
    )
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # -- Train ------------------------------------------------------------
        train_metrics = train_epoch(
            model, train_loader, optimizer, temperature, grad_clip, device
        )
        scheduler.step()

        # -- Validate ---------------------------------------------------------
        val_metrics = validate(model, val_loader, temperature, device)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "lr": train_metrics["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_info)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  sil=%.3f  knn=%.3f  lr=%.2e",
            epoch,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["silhouette_score"],
            val_metrics["knn_accuracy"],
            train_metrics["lr"],
        )

        # -- Best model -------------------------------------------------------
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                ckpt_path / "best_model.pt",
            )
            logger.info("  --> New best model saved (val_loss=%.4f)", best_val_loss)
        else:
            epochs_no_improve += 1

        # -- Periodic checkpoint ----------------------------------------------
        if epoch % checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                },
                ckpt_path / f"checkpoint_epoch{epoch:04d}.pt",
            )

        # -- Early stopping ---------------------------------------------------
        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch,
                patience,
            )
            break

    elapsed = time.time() - t0
    logger.info(
        "Training complete in %.1fs.  Best val_loss=%.4f at epoch %d.",
        elapsed,
        best_val_loss,
        best_epoch,
    )

    # Reload best model (weights_only=True: we only read model_state_dict
    # and val_metrics; this path never needs pickled Python objects).
    best_ckpt = torch.load(
        ckpt_path / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_metrics": best_ckpt.get("val_metrics", {}),
        "history": history,
    }
