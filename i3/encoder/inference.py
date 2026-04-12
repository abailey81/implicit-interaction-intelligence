"""Inference wrapper for the trained User State Encoder.

Provides stateful per-user rolling windows and helper utilities for embedding
visualisation (2-D projection).  Supports both FP32 and INT8 quantised
models.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from i3.encoder.tcn import TemporalConvNet
from i3.interaction.types import InteractionFeatureVector

logger = logging.getLogger(__name__)


class EncoderInference:
    """Stateful inference wrapper around a trained :class:`TemporalConvNet`.

    Maintains a rolling window of :class:`InteractionFeatureVector` per user
    so that a single new observation can be encoded in context.

    Args:
        checkpoint_path: Path to a saved ``.pt`` checkpoint produced by the
            training loop (must contain ``"model_state_dict"``).
        window_size:     Number of timesteps in the rolling window.
        device:          Torch device string (``"cpu"``, ``"cuda"``).
        quantise_int8:   If ``True``, apply dynamic INT8 quantisation after
            loading the model weights.
        input_dim:       Feature dimensionality (default 32).
        hidden_dims:     Channel widths for TCN blocks.
        kernel_size:     Kernel width.
        dilations:       Per-block dilation factors.
        embedding_dim:   Output embedding dimensionality (default 64).
        dropout:         Dropout probability (unused at inference but needed
            for weight-shape compatibility).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        window_size: int = 10,
        device: str = "cpu",
        quantise_int8: bool = False,
        input_dim: int = 32,
        hidden_dims: Optional[list[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[list[int]] = None,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        self.window_size = window_size
        self.device = torch.device(device)

        # -- Build model and load weights -------------------------------------
        self.model = TemporalConvNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dilations=dilations,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # Security: weights_only=True forbids arbitrary code execution
        # during deserialization.  Inference paths must never load
        # pickled training state.
        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # -- Optional INT8 dynamic quantisation -------------------------------
        if quantise_int8:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d},
                dtype=torch.qint8,
            )
            logger.info("Model dynamically quantised to INT8.")

        # -- Per-user rolling windows -----------------------------------------
        self._windows: dict[str, list[torch.Tensor]] = defaultdict(list)

        # -- 2-D projection (fitted lazily) -----------------------------------
        self._proj_matrix: Optional[torch.Tensor] = None  # [64, 2]

        logger.info(
            "EncoderInference ready (window=%d, device=%s, quantised=%s).",
            window_size,
            self.device,
            quantise_int8,
        )

    # -- Core encoding --------------------------------------------------------

    @torch.no_grad()
    def encode(
        self, features: list[InteractionFeatureVector]
    ) -> torch.Tensor:
        """Encode a full sequence of feature vectors into a 64-dim embedding.

        If the sequence is shorter than ``window_size`` it is left-padded with
        zeros.  If longer, only the last ``window_size`` vectors are used.

        Args:
            features: List of :class:`InteractionFeatureVector` instances.

        Returns:
            1-D ``torch.Tensor`` of shape ``[64]`` (L2-normalised).
        """
        tensors = [fv.to_tensor() for fv in features]

        # Trim or pad to window_size
        if len(tensors) > self.window_size:
            tensors = tensors[-self.window_size :]
        elif len(tensors) < self.window_size:
            pad_count = self.window_size - len(tensors)
            pad = [torch.zeros(32) for _ in range(pad_count)]
            tensors = pad + tensors

        seq = torch.stack(tensors, dim=0)  # [window_size, 32]
        batch = seq.unsqueeze(0).to(self.device)  # [1, window_size, 32]
        embedding = self.model(batch)  # [1, 64]
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_single(
        self, feature: InteractionFeatureVector, user_id: str
    ) -> torch.Tensor:
        """Append one feature vector to the user's window and encode.

        Maintains a per-user rolling buffer of up to ``window_size`` vectors.

        Args:
            feature:  A single :class:`InteractionFeatureVector`.
            user_id:  Unique user identifier for the rolling buffer.

        Returns:
            1-D ``torch.Tensor`` of shape ``[64]`` (L2-normalised).
        """
        t = feature.to_tensor()
        window = self._windows[user_id]
        window.append(t)

        # Trim to window_size
        if len(window) > self.window_size:
            self._windows[user_id] = window[-self.window_size :]
            window = self._windows[user_id]

        # Pad if shorter
        if len(window) < self.window_size:
            pad_count = self.window_size - len(window)
            padded = [torch.zeros(32)] * pad_count + list(window)
        else:
            padded = list(window)

        seq = torch.stack(padded, dim=0).unsqueeze(0).to(self.device)
        embedding = self.model(seq)
        return embedding.squeeze(0).cpu()

    # -- 2-D projection for visualisation ------------------------------------

    def project_2d(self, embedding: torch.Tensor) -> tuple[float, float]:
        """Project a 64-dim embedding down to 2-D for visualisation.

        Uses a learned random linear projection (fixed once initialised).
        For production, replace with PCA fitted on a reference set.

        Args:
            embedding: 1-D tensor of shape ``[64]``.

        Returns:
            Tuple ``(x, y)`` coordinates.
        """
        if self._proj_matrix is None:
            # Initialise a fixed random projection (Gaussian)
            rng = torch.Generator()
            rng.manual_seed(42)
            self._proj_matrix = torch.randn(
                64, 2, generator=rng
            ) / np.sqrt(64)

        coords = embedding @ self._proj_matrix  # [2]
        return (float(coords[0]), float(coords[1]))

    def fit_pca_projection(self, embeddings: torch.Tensor) -> None:
        """Fit a PCA-based 2-D projection from a batch of embeddings.

        Replaces the random projection with the top-2 principal components
        computed via SVD.

        Args:
            embeddings: Tensor of shape ``[N, 64]``.
        """
        # Centre
        mean = embeddings.mean(dim=0, keepdim=True)
        centred = embeddings - mean
        # SVD
        _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
        self._proj_matrix = Vt[:2, :].t()  # [64, 2]
        logger.info("PCA projection fitted on %d samples.", embeddings.size(0))

    # -- Utilities ------------------------------------------------------------

    def clear_window(self, user_id: str) -> None:
        """Clear the rolling window for a specific user."""
        self._windows.pop(user_id, None)

    def clear_all_windows(self) -> None:
        """Clear all per-user rolling windows."""
        self._windows.clear()
