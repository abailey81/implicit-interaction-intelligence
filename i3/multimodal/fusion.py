"""Modality fusion for the multi-modal extension of the I³ TCN encoder.

The existing encoder consumes a 32-dim :class:`InteractionFeatureVector` — four
groups of eight.  The keystroke-dynamics group is *one* modality; the other
three are derived from the same keystrokes.  This module shows how to extend
the pipeline cleanly to additional perceptual channels (voice, touch, gaze,
accelerometer) — *without* modifying the encoder itself.

The fused tensor layout is::

    [ keystroke(8) | message(8) | session(8) | deviation(8)
      | voice(8)   | touch(8)   | gaze(8)    | accel(8)     ]    -- 64 dims

The extra groups are **optional**: missing modalities contribute zeros plus a
binary mask bit so downstream layers can distinguish "measured zero" from
"not observed".  A learned :class:`ModalityEmbedding` conditions the
hidden state on *which* modalities are actually present.

References
----------
* Baltrušaitis, T., Ahuja, C., Morency, L.-P. (2019). *Multimodal machine
  learning: a survey and taxonomy.*  IEEE TPAMI 41(2).
* Bai, S., Kolter, J. Z., Koltun, V. (2018). *An empirical evaluation of
  generic convolutional and recurrent networks for sequence modelling.*
  arXiv:1803.01271.  (The modality-agnostic TCN is load-bearing here.)
* Ngiam, J. et al. (2011). *Multimodal deep learning.*  ICML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical modality ordering
# ---------------------------------------------------------------------------

MODALITY_INDEX: dict[str, int] = {
    "keystroke": 0,
    "message": 1,
    "session": 2,
    "deviation": 3,
    "voice": 4,
    "touch": 5,
    "gaze": 6,
    "accelerometer": 7,
}
"""Stable index assignment used by :class:`ModalityEmbedding` and fusion
tensor layout.  The first four modalities map to the native 32-dim
:class:`InteractionFeatureVector`; the last four are added by this module."""

NUM_MODALITIES: int = len(MODALITY_INDEX)
MODALITY_GROUP_DIM: int = 8


# ---------------------------------------------------------------------------
# Modality-embedding — learned per-modality bias for conditioning the TCN
# ---------------------------------------------------------------------------

class ModalityEmbedding(nn.Module):
    """Learned embedding conditioning the encoder on the present modalities.

    At each timestep, a boolean modality-mask vector indicates which groups
    carried real data.  The module returns a ``d_model``-wide vector — the
    **mean of the embeddings of the active modalities** — which callers add
    elementwise to the TCN's input projection.  This is conceptually the same
    trick Vaswani et al. (2017) use for token-type embeddings in BERT.

    Args:
        num_modalities: Number of supported modalities (default 8).
        d_model: Dimensionality of the embedding space.  Match the TCN's
            ``input_dim`` or hidden width.
    """

    def __init__(self, num_modalities: int = NUM_MODALITIES, d_model: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.num_modalities = num_modalities
        self.d_model = d_model

    def forward(self, modality_mask: torch.Tensor) -> torch.Tensor:
        """Compute the conditioning vector for a batch of modality masks.

        Args:
            modality_mask: ``bool`` / ``float32`` tensor of shape
                ``(batch, num_modalities)``.  ``True`` / ``1.0`` means the
                modality is present.

        Returns:
            ``float32`` tensor of shape ``(batch, d_model)`` — the mean of the
            active modalities' embeddings.  All-zero masks return the
            zero vector.
        """
        if modality_mask.shape[-1] != self.num_modalities:
            raise ValueError(
                f"Expected mask with {self.num_modalities} dims, "
                f"got {modality_mask.shape[-1]}"
            )
        mask = modality_mask.to(dtype=self.embedding.weight.dtype)
        # [batch, N] @ [N, d] -> [batch, d]
        weighted = mask @ self.embedding.weight
        # Normalise by the count of active modalities to stay on a stable scale.
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return weighted / count


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class FusedFeatureFrame:
    """A single timestep of fused multi-modal features.

    Attributes:
        features: ``float32`` tensor of shape ``(num_modalities * 8,)``.
        modality_mask: ``bool`` tensor of shape ``(num_modalities,)`` — True
            where that modality contributed real data.
    """

    features: torch.Tensor
    modality_mask: torch.Tensor


class ModalityFusion:
    """Concatenate modality-specific 8-dim groups into a fused frame.

    The fusion rule is "concatenate plus mask": present modalities' vectors
    flow in unchanged; absent ones are zero-filled, and the mask makes the
    distinction visible to downstream layers.

    The class is **stateless** in the statistical sense — it has no learned
    parameters.  Learned parameters live in :class:`ModalityEmbedding`, which
    fusion output plumbs into the TCN via elementwise addition.  Keeping the
    two responsibilities separated makes the module trivially testable.

    Example:
        >>> fusion = ModalityFusion()
        >>> frame = fusion.fuse(
        ...     keystroke=np.zeros(8, dtype=np.float32),
        ...     voice=np.ones(8, dtype=np.float32),
        ... )
        >>> frame.features.shape
        torch.Size([64])
    """

    def __init__(self) -> None:
        self.num_modalities = NUM_MODALITIES
        self.group_dim = MODALITY_GROUP_DIM
        self.total_dim = NUM_MODALITIES * MODALITY_GROUP_DIM

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fuse(
        self,
        keystroke: Optional[np.ndarray] = None,
        message: Optional[np.ndarray] = None,
        session: Optional[np.ndarray] = None,
        deviation: Optional[np.ndarray] = None,
        voice: Optional[np.ndarray] = None,
        touch: Optional[np.ndarray] = None,
        gaze: Optional[np.ndarray] = None,
        accelerometer: Optional[np.ndarray] = None,
    ) -> FusedFeatureFrame:
        """Fuse per-modality 8-dim vectors into a single :class:`FusedFeatureFrame`.

        Each argument is an optional 8-dim ``np.ndarray``.  Omitted or
        ``None`` modalities are zero-filled and flagged in the mask.

        Args:
            keystroke: Keystroke-dynamics group (8-dim).
            message: Message-content group (8-dim).
            session: Session-dynamics group (8-dim).
            deviation: Deviation-metrics group (8-dim).
            voice: Voice-prosody group (8-dim).
            touch: Touchscreen group (8-dim).
            gaze: Gaze group (8-dim).
            accelerometer: Wearable accelerometer group (8-dim).

        Returns:
            A :class:`FusedFeatureFrame`.

        Raises:
            ValueError: If any supplied modality does not have exactly 8
                elements.
        """
        supplied: dict[str, Optional[np.ndarray]] = {
            "keystroke": keystroke,
            "message": message,
            "session": session,
            "deviation": deviation,
            "voice": voice,
            "touch": touch,
            "gaze": gaze,
            "accelerometer": accelerometer,
        }

        buf = np.zeros(self.total_dim, dtype=np.float32)
        mask = np.zeros(self.num_modalities, dtype=np.bool_)

        for name, vec in supplied.items():
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (self.group_dim,):
                raise ValueError(
                    f"Modality '{name}' expected shape ({self.group_dim},), "
                    f"got {arr.shape}"
                )
            idx = MODALITY_INDEX[name]
            buf[idx * self.group_dim : (idx + 1) * self.group_dim] = arr
            mask[idx] = True

        return FusedFeatureFrame(
            features=torch.from_numpy(buf),
            modality_mask=torch.from_numpy(mask),
        )

    # ------------------------------------------------------------------
    # Sequence-level convenience
    # ------------------------------------------------------------------
    def fuse_sequence(self, frames: list[FusedFeatureFrame]) -> dict[str, torch.Tensor]:
        """Stack a list of per-timestep :class:`FusedFeatureFrame` into a
        sequence tensor suitable for TCN ingestion.

        Returns:
            A dict with keys ``features`` (shape ``(seq_len, total_dim)``)
            and ``modality_mask`` (shape ``(seq_len, num_modalities)``).
        """
        if not frames:
            raise ValueError("fuse_sequence requires at least one frame")
        features = torch.stack([f.features for f in frames], dim=0)
        masks = torch.stack([f.modality_mask for f in frames], dim=0)
        return {"features": features, "modality_mask": masks}
