"""Inference wrapper for the trained User State Encoder.

Provides stateful per-user rolling windows and helper utilities for embedding
visualisation (2-D projection).  Supports both FP32 and INT8 quantised
models.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Optional

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
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        # SEC: validate checkpoint path -- resolve symlinks and confirm the
        # target is an existing regular file before handing it to torch.load.
        # Mitigates path-traversal / device-file abuse from caller-supplied
        # strings.
        ckpt_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not ckpt_path.is_file():
            raise ValueError(
                f"Checkpoint path is not a regular file: {ckpt_path}"
            )

        # SEC: dynamic INT8 quantisation only supports CPU; combining CUDA +
        # INT8 silently mis-runs.  Fail loudly here.
        if quantise_int8 and self.device.type != "cpu":
            raise ValueError(
                "Dynamic INT8 quantisation is CPU-only; "
                f"got device={self.device}."
            )

        # -- Build model and load weights -------------------------------------
        self.model = TemporalConvNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dilations=dilations,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # SEC: weights_only=True forbids arbitrary code execution during
        # deserialization (no pickle reduce / __reduce__ shenanigans).
        # Inference paths must never load pickled training state.
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=True,
        )
        if "model_state_dict" not in ckpt:
            raise KeyError(
                "Checkpoint missing required key 'model_state_dict'."
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
        # deque(maxlen=window_size) gives O(1) append + automatic eviction
        # and removes the rebind-list footgun the previous list-based
        # implementation had.
        self._windows: dict[str, Deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # SEC: protect _windows and _proj_matrix from concurrent encode/clear
        # calls.  Without this, two threads encoding the same user can race
        # on append/trim and corrupt the buffer.
        self._lock = threading.RLock()

        # -- 2-D projection (fitted lazily) -----------------------------------
        self._proj_matrix: Optional[torch.Tensor] = None  # [64, 2]
        self._proj_is_pca: bool = False

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

        If the sequence is empty, returns the zero-input embedding (a fully
        left-padded window).  If shorter than ``window_size`` it is left-padded
        with zeros.  If longer, only the last ``window_size`` vectors are used.

        Args:
            features: List of :class:`InteractionFeatureVector` instances.

        Returns:
            1-D ``torch.Tensor`` of shape ``[embedding_dim]`` (L2-normalised).
        """
        tensors = [fv.to_tensor() for fv in features]

        # Sanity-check feature dimensionality so a misshaped vector fails
        # loudly instead of broadcasting silently.
        for i, t in enumerate(tensors):
            if t.shape != (self.input_dim,):
                raise ValueError(
                    f"feature[{i}].to_tensor() returned shape {tuple(t.shape)},"
                    f" expected ({self.input_dim},)"
                )

        # Trim or pad to window_size
        if len(tensors) >= self.window_size:
            tensors = tensors[-self.window_size :]
        else:
            pad_count = self.window_size - len(tensors)
            pad = [
                torch.zeros(self.input_dim) for _ in range(pad_count)
            ]
            tensors = pad + tensors

        seq = torch.stack(tensors, dim=0)  # [window_size, input_dim]
        batch = seq.unsqueeze(0).to(self.device)  # [1, window_size, input_dim]
        embedding = self.model(batch)  # [1, embedding_dim]
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_single(
        self, feature: InteractionFeatureVector, user_id: str
    ) -> torch.Tensor:
        """Append one feature vector to the user's window and encode.

        Maintains a per-user rolling buffer of up to ``window_size`` vectors.
        Thread-safe via an internal RLock; concurrent calls for the same or
        different users will not corrupt window state.

        Args:
            feature:  A single :class:`InteractionFeatureVector`.
            user_id:  Unique user identifier for the rolling buffer.

        Returns:
            1-D ``torch.Tensor`` of shape ``[embedding_dim]`` (L2-normalised).
        """
        t = feature.to_tensor()
        if t.shape != (self.input_dim,):
            raise ValueError(
                f"feature.to_tensor() returned shape {tuple(t.shape)},"
                f" expected ({self.input_dim},)"
            )

        # SEC: lock window mutation against concurrent encode/clear callers.
        with self._lock:
            window = self._windows[user_id]
            window.append(t)  # deque(maxlen) auto-evicts oldest
            # Snapshot a list copy under the lock so the model forward
            # below is independent of any later mutation.
            snapshot = list(window)

        # Pad if shorter than window_size (deque is bounded above)
        if len(snapshot) < self.window_size:
            pad_count = self.window_size - len(snapshot)
            # Build distinct zero tensors (avoid alias-by-multiplication).
            padded = [
                torch.zeros(self.input_dim) for _ in range(pad_count)
            ] + snapshot
        else:
            padded = snapshot

        seq = torch.stack(padded, dim=0).unsqueeze(0).to(self.device)
        embedding = self.model(seq)
        return embedding.squeeze(0).cpu()

    # -- 2-D projection for visualisation ------------------------------------

    def project_2d(self, embedding: torch.Tensor) -> tuple[float, float]:
        """Project a ``embedding_dim``-dim embedding down to 2-D.

        If :meth:`fit_pca_projection` has been called the projection uses the
        top-2 principal components and is therefore meaningful for
        visualisation.  Otherwise a deterministic seeded random Gaussian
        projection is used as a fallback (and a one-time warning is logged so
        callers know they should fit PCA on a reference batch first).

        The PCA axes are sign-canonicalised in :meth:`fit_pca_projection` so
        repeated PCA fits on the same data give identical coordinates
        (avoiding the SVD sign-ambiguity).

        Args:
            embedding: 1-D tensor of shape ``[embedding_dim]``.

        Returns:
            Tuple ``(x, y)`` coordinates.
        """
        with self._lock:
            if self._proj_matrix is None:
                logger.warning(
                    "project_2d called before fit_pca_projection -- "
                    "falling back to a fixed random projection.  Coordinates"
                    " will not be PCA-meaningful."
                )
                rng = torch.Generator()
                rng.manual_seed(42)
                self._proj_matrix = torch.randn(
                    self.embedding_dim, 2, generator=rng
                ) / np.sqrt(self.embedding_dim)
                self._proj_is_pca = False
            proj = self._proj_matrix

        if embedding.dim() != 1 or embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"project_2d expected shape ({self.embedding_dim},), "
                f"got {tuple(embedding.shape)}"
            )

        coords = embedding.detach().cpu() @ proj  # [2]
        return (float(coords[0]), float(coords[1]))

    def fit_pca_projection(self, embeddings: torch.Tensor) -> None:
        """Fit a PCA-based 2-D projection from a batch of embeddings.

        Replaces the random projection with the top-2 principal components
        computed via SVD.  The sign of each principal component is
        canonicalised so that the entry with the largest absolute value is
        positive -- this removes SVD sign ambiguity and makes the projection
        deterministic across runs / hardware.

        Args:
            embeddings: Tensor of shape ``[N, embedding_dim]`` with ``N >= 2``.
        """
        if embeddings.dim() != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"fit_pca_projection expected shape (N, {self.embedding_dim}),"
                f" got {tuple(embeddings.shape)}"
            )
        if embeddings.size(0) < 2:
            raise ValueError(
                "fit_pca_projection needs at least 2 samples; got "
                f"{embeddings.size(0)}"
            )

        embeddings = embeddings.detach().cpu().float()
        # Centre
        mean = embeddings.mean(dim=0, keepdim=True)
        centred = embeddings - mean
        # SVD
        _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
        proj = Vt[:2, :].t().contiguous()  # [embedding_dim, 2]

        # Canonicalise sign per component for determinism.
        for k in range(proj.size(1)):
            col = proj[:, k]
            idx = int(torch.argmax(col.abs()))
            if col[idx].item() < 0:
                proj[:, k] = -col

        with self._lock:
            self._proj_matrix = proj
            self._proj_is_pca = True
        logger.info("PCA projection fitted on %d samples.", embeddings.size(0))

    # -- Utilities ------------------------------------------------------------

    def clear_window(self, user_id: str) -> None:
        """Clear the rolling window for a specific user."""
        with self._lock:
            self._windows.pop(user_id, None)

    def clear_all_windows(self) -> None:
        """Clear all per-user rolling windows."""
        with self._lock:
            self._windows.clear()
