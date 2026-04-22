"""Runnable multimodal late-fusion (Batch F-1).

This module upgrades the stub :class:`i3.multimodal.fusion.ModalityFusion`
into a **trainable** late-fusion head that converts per-modality 8-dim
feature vectors into the 64-dim user-state embedding consumed elsewhere in
the I3 stack.  Three fusion strategies are provided:

* ``late_concat`` — classical concatenation followed by a learned projection.
* ``late_gated`` — per-modality sigmoid gates multiplied onto each
  modality's contribution before concatenation.
* ``attention`` — multi-head attention over a small set of modality tokens,
  with a learned CLS token whose projected output is the fused embedding.

Missing modalities are handled by one of two policies:

* ``zero_fill`` — substitute a zero vector for the missing modality.
* ``mask_drop`` — drop the modality from the attention / gating weights
  entirely (concat strategy still zero-fills but renormalises the linear
  projection's contribution via the gate).

References
----------
* Baltrusaitis, T., Ahuja, C., Morency, L.-P. (2019). *Multimodal Machine
  Learning: A Survey and Taxonomy.*  IEEE TPAMI 41(2).
* Liang, P. P., Zadeh, A., Morency, L.-P. (2023). *Foundations and Trends in
  Multimodal Machine Learning: Principles, Challenges, and Open Questions.*
  arXiv:2209.03430 (survey of fusion architectures).
* Vaswani, A. et al. (2017). *Attention is all you need.*  NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Literal, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


FusionStrategy = Literal["late_concat", "late_gated", "attention"]
MissingPolicy = Literal["zero_fill", "mask_drop"]


# Canonical modality order used by all fusion strategies.  Users can supply a
# subset via ``modality_dim_map``; unlisted modalities are ignored.
_DEFAULT_MODALITY_ORDER: tuple[str, ...] = (
    "keystroke",
    "voice",
    "vision",
    "accelerometer",
)


def _as_tensor(
    x: np.ndarray | torch.Tensor | None,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coerce a modality feature into a ``(dim,)`` float tensor.

    Args:
        x: Numpy array, torch tensor, or ``None``.
        dim: Expected feature dimensionality.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tensor of shape ``(dim,)`` on *device* with *dtype*.  ``None`` is
        converted to a zero vector.

    Raises:
        ValueError: If *x* is supplied but has the wrong shape.
    """
    if x is None:
        return torch.zeros(dim, dtype=dtype, device=device)
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
        if arr.shape != (dim,):
            raise ValueError(
                f"Expected modality vector shape ({dim},), got {arr.shape}"
            )
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    if isinstance(x, torch.Tensor):
        if x.shape != (dim,):
            raise ValueError(
                f"Expected modality tensor shape ({dim},), got {tuple(x.shape)}"
            )
        return x.to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported modality input type: {type(x).__name__}")


class MultimodalFusion(nn.Module):
    """Late-fusion head producing a 64-dim user-state embedding.

    The class is an ``nn.Module`` so fusion weights can be trained jointly
    with the rest of the pipeline.  When ``tcn_encoder`` is provided, the
    concatenated modality vector is first projected to the encoder's input
    dim (32), passed through the encoder as a one-timestep sequence, and the
    resulting 64-dim embedding returned.  When the encoder is ``None``, a
    standalone linear head maps directly into 64 dims — useful for unit
    tests and for callers that do not yet have a trained TCN.

    Args:
        tcn_encoder: Optional :class:`i3.encoder.tcn.TemporalConvNet` (or
            any module with ``input_dim`` and a ``forward`` that accepts a
            ``(batch, seq_len, input_dim)`` tensor).
        modality_dim_map: Ordered mapping from modality name to feature
            dimensionality.  Insertion order is respected.
        fusion_strategy: ``"late_concat"``, ``"late_gated"``, or ``"attention"``.
        missing_modality_policy: ``"zero_fill"`` or ``"mask_drop"``.
        embedding_dim: Output embedding width (default 64).
        attention_num_heads: Heads used by the ``"attention"`` strategy.
        attention_hidden_dim: Per-modality token width in the attention path.
    """

    def __init__(
        self,
        tcn_encoder: nn.Module | None = None,
        modality_dim_map: Mapping[str, int] | None = None,
        fusion_strategy: FusionStrategy = "late_concat",
        missing_modality_policy: MissingPolicy = "zero_fill",
        embedding_dim: int = 64,
        attention_num_heads: int = 4,
        attention_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if fusion_strategy not in ("late_concat", "late_gated", "attention"):
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy!r}")
        if missing_modality_policy not in ("zero_fill", "mask_drop"):
            raise ValueError(
                f"Unknown missing_modality_policy: {missing_modality_policy!r}"
            )

        if modality_dim_map is None:
            modality_dim_map = {name: 8 for name in _DEFAULT_MODALITY_ORDER}
        # Preserve insertion order explicitly.
        self._modality_order: tuple[str, ...] = tuple(modality_dim_map.keys())
        self._modality_dims: dict[str, int] = dict(modality_dim_map)
        if len(self._modality_order) == 0:
            raise ValueError("modality_dim_map must contain at least one modality")

        self.fusion_strategy: FusionStrategy = fusion_strategy
        self.missing_modality_policy: MissingPolicy = missing_modality_policy
        self.embedding_dim = int(embedding_dim)
        self.tcn_encoder = tcn_encoder

        total_dim = sum(self._modality_dims.values())
        self._total_dim = total_dim

        encoder_input_dim = (
            int(getattr(tcn_encoder, "input_dim", 32))
            if tcn_encoder is not None
            else 32
        )
        self._encoder_input_dim = encoder_input_dim

        # -- Shared projection into the encoder input width ------------------
        self.concat_proj = nn.Linear(total_dim, encoder_input_dim)

        # -- Gated strategy --------------------------------------------------
        # One sigmoid gate per modality, conditioned on the concat of all
        # modality vectors.  Gate output shape == (num_modalities,).
        self.gate_head = nn.Sequential(
            nn.Linear(total_dim, max(total_dim // 2, len(self._modality_order))),
            nn.ReLU(),
            nn.Linear(
                max(total_dim // 2, len(self._modality_order)),
                len(self._modality_order),
            ),
            nn.Sigmoid(),
        )

        # -- Attention strategy ----------------------------------------------
        self.attention_hidden_dim = int(attention_hidden_dim)
        self.modality_projs = nn.ModuleDict(
            {
                name: nn.Linear(dim, self.attention_hidden_dim)
                for name, dim in self._modality_dims.items()
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.attention_hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_hidden_dim,
            num_heads=int(attention_num_heads),
            batch_first=True,
        )
        self.attention_out_proj = nn.Linear(self.attention_hidden_dim, encoder_input_dim)

        # -- Fallback head when no TCN encoder is attached -------------------
        self.standalone_head = nn.Linear(encoder_input_dim, self.embedding_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def fuse(
        self,
        keystroke_features: np.ndarray | torch.Tensor,
        voice_features: np.ndarray | torch.Tensor | None = None,
        vision_features: np.ndarray | torch.Tensor | None = None,
        accelerometer_features: np.ndarray | torch.Tensor | None = None,
        *,
        extra_features: Mapping[str, np.ndarray | torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Fuse per-modality features into a single embedding.

        This is an ``async`` method so it can be awaited from the rest of the
        async I3 stack; internally it performs only synchronous torch work.

        Args:
            keystroke_features: Keystroke-dynamics group.  Required.
            voice_features: Voice-prosody group or ``None`` if absent.
            vision_features: Facial-affect / gaze group or ``None``.
            accelerometer_features: Wearable accelerometer group or ``None``.
            extra_features: Optional mapping with additional modalities whose
                names appear in ``modality_dim_map``.

        Returns:
            A ``(embedding_dim,)`` tensor — the fused 64-dim user-state
            embedding (default).

        Raises:
            ValueError: If ``keystroke_features`` is ``None`` or any supplied
                tensor has the wrong shape.
        """
        if keystroke_features is None:
            raise ValueError("keystroke_features is required")

        named_inputs: dict[str, np.ndarray | torch.Tensor | None] = {
            "keystroke": keystroke_features,
            "voice": voice_features,
            "vision": vision_features,
            "accelerometer": accelerometer_features,
        }
        if extra_features is not None:
            for k, v in extra_features.items():
                if k in named_inputs and named_inputs[k] is not None:
                    logger.debug("fuse: duplicate modality %s, using extra_features", k)
                named_inputs[k] = v

        embedding = self._forward_sync(named_inputs)
        return embedding

    # ------------------------------------------------------------------
    def forward(
        self, named_inputs: Mapping[str, np.ndarray | torch.Tensor | None]
    ) -> torch.Tensor:
        """Synchronous forward — identical to :meth:`fuse` but usable in ``nn.Module`` training loops.

        Args:
            named_inputs: Mapping from modality name to its feature vector.

        Returns:
            The ``(embedding_dim,)`` fused embedding tensor.
        """
        return self._forward_sync(named_inputs)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------
    def _forward_sync(
        self, named_inputs: Mapping[str, np.ndarray | torch.Tensor | None]
    ) -> torch.Tensor:
        device = self.concat_proj.weight.device
        dtype = self.concat_proj.weight.dtype

        # -- Collect per-modality tensors + presence mask --------------------
        tensors: dict[str, torch.Tensor] = {}
        presence: dict[str, bool] = {}
        for name in self._modality_order:
            dim = self._modality_dims[name]
            vec = named_inputs.get(name)
            present = vec is not None
            tensor = _as_tensor(vec, dim, device=device, dtype=dtype)
            tensors[name] = tensor
            presence[name] = present

        present_count = sum(presence.values())
        if present_count == 0:
            raise ValueError("fuse() requires at least one non-None modality")

        if self.fusion_strategy == "late_concat":
            projected = self._fuse_concat(tensors, presence)
        elif self.fusion_strategy == "late_gated":
            projected = self._fuse_gated(tensors, presence)
        else:  # "attention"
            projected = self._fuse_attention(tensors, presence)

        return self._project_to_embedding(projected)

    # ------------------------------------------------------------------
    def _fuse_concat(
        self,
        tensors: Mapping[str, torch.Tensor],
        presence: Mapping[str, bool],
    ) -> torch.Tensor:
        """Late-concat strategy: zero-fill missing + linear projection."""
        ordered = [tensors[name] for name in self._modality_order]
        concat = torch.cat(ordered, dim=-1)
        if self.missing_modality_policy == "mask_drop":
            # Renormalise by the fraction of dims that were actually present,
            # so a 2-modality call does not get suppressed vs 4-modality.
            present_dims = sum(
                self._modality_dims[name]
                for name, is_present in presence.items()
                if is_present
            )
            total_dims = sum(self._modality_dims.values())
            scale = float(total_dims) / max(float(present_dims), 1.0)
            concat = concat * scale
        return self.concat_proj(concat)

    # ------------------------------------------------------------------
    def _fuse_gated(
        self,
        tensors: Mapping[str, torch.Tensor],
        presence: Mapping[str, bool],
    ) -> torch.Tensor:
        """Gated strategy: learned per-modality weights in [0, 1]."""
        ordered = [tensors[name] for name in self._modality_order]
        concat = torch.cat(ordered, dim=-1)
        gates = self.gate_head(concat)  # shape: (num_modalities,)

        if self.missing_modality_policy == "mask_drop":
            mask = torch.tensor(
                [1.0 if presence[name] else 0.0 for name in self._modality_order],
                dtype=gates.dtype,
                device=gates.device,
            )
            gates = gates * mask

        # Scale each modality contribution by its gate before concatenation.
        scaled: list[torch.Tensor] = []
        for i, name in enumerate(self._modality_order):
            scaled.append(tensors[name] * gates[i])
        gated_concat = torch.cat(scaled, dim=-1)
        return self.concat_proj(gated_concat)

    # ------------------------------------------------------------------
    def _fuse_attention(
        self,
        tensors: Mapping[str, torch.Tensor],
        presence: Mapping[str, bool],
    ) -> torch.Tensor:
        """Attention strategy: CLS token + modality tokens through MHA."""
        projected_tokens: list[torch.Tensor] = []
        key_padding_mask_entries: list[bool] = []  # True == ignore

        for name in self._modality_order:
            token = self.modality_projs[name](tensors[name])
            projected_tokens.append(token)
            if self.missing_modality_policy == "mask_drop" and not presence[name]:
                key_padding_mask_entries.append(True)
            else:
                key_padding_mask_entries.append(False)

        # shape: (1, num_modalities + 1, hidden)
        cls = self.cls_token  # (1, 1, hidden)
        tokens = torch.stack(projected_tokens, dim=0).unsqueeze(0)
        sequence = torch.cat([cls, tokens], dim=1)

        if self.missing_modality_policy == "mask_drop":
            pad_mask = torch.tensor(
                [[False] + key_padding_mask_entries],
                dtype=torch.bool,
                device=sequence.device,
            )
        else:
            pad_mask = None

        attended, _ = self.attention(
            sequence, sequence, sequence, key_padding_mask=pad_mask
        )
        cls_out = attended[:, 0, :].squeeze(0)
        return self.attention_out_proj(cls_out)

    # ------------------------------------------------------------------
    def _project_to_embedding(self, projected: torch.Tensor) -> torch.Tensor:
        """Run ``projected`` through the TCN (if any) or the standalone head."""
        if self.tcn_encoder is not None:
            # TCN expects (batch, seq_len, input_dim).  We make a length-1
            # sequence from the single fused vector.
            tcn_input = projected.unsqueeze(0).unsqueeze(0)
            embedding = self.tcn_encoder(tcn_input)
            return embedding.squeeze(0)
        return F.normalize(self.standalone_head(projected), p=2, dim=-1)


__all__ = [
    "FusionStrategy",
    "MissingPolicy",
    "MultimodalFusion",
]
