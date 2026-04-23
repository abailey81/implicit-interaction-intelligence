"""Flower-compatible federated client for the I³ TCN encoder.

The client wraps the encoder **only** — the SLM is intentionally excluded.
Federated SLM training would require gigabytes per client and raises the
memorisation-attack surface dramatically; the encoder, at ~220 k parameters,
is a much better federated target.

This module soft-imports ``flwr`` and ``opacus``.  When ``flwr`` is not
installed the client falls back to a minimal stand-in class that surfaces
the install hint but still exposes ``fit`` / ``evaluate`` for unit-testable
dry runs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft imports — flwr, opacus
# ---------------------------------------------------------------------------

try:
    import flwr  # type: ignore[import-not-found]
    from flwr.client import NumPyClient  # type: ignore[import-not-found]

    _FLWR_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    flwr = None  # type: ignore[assignment]

    class NumPyClient:  # type: ignore[no-redef]
        """Lightweight stand-in when Flower is unavailable.

        Matches the public surface of ``flwr.client.NumPyClient`` so
        subclasses still type-check in isolation.
        """

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            raise NotImplementedError

        def fit(
            self, parameters: list[np.ndarray], config: dict[str, Any]
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            raise NotImplementedError

        def evaluate(
            self, parameters: list[np.ndarray], config: dict[str, Any]
        ) -> tuple[float, int, dict[str, Any]]:
            raise NotImplementedError

    _FLWR_AVAILABLE = False

try:
    from opacus import PrivacyEngine  # type: ignore[import-not-found]

    _OPACUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PrivacyEngine = None  # type: ignore[assignment]
    _OPACUS_AVAILABLE = False


_FLWR_INSTALL_HINT = (
    "flwr is not installed. Install the future-work group with "
    "`poetry install --with future-work` (pulls flwr + opacus + librosa)."
)


if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_parameters(model: nn.Module) -> list[np.ndarray]:
    """Serialise a model's state-dict to the NumPy list Flower expects."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def _load_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load a list of NumPy arrays into a model's state-dict."""
    state_dict = model.state_dict()
    for (k, _v), p in zip(state_dict.items(), parameters):
        state_dict[k] = torch.tensor(p, dtype=state_dict[k].dtype)
    model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class I3FederatedClient(NumPyClient):
    """Federated client wrapping the I³ TCN encoder only.

    Each client runs one or more local SGD passes on its own interaction
    data, applies Opacus-backed DP-SGD gradient clipping (if available),
    and ships the updated encoder weights back to the server.

    Args:
        model: The encoder to train.  Must be the same architecture across
            clients.
        train_loader: Local training ``DataLoader`` — yields ``(x, y)``
            tensors where ``x`` is the 32-dim feature sequence and ``y`` is
            whatever positive/negative pair signal the local trainer uses.
        loss_fn: Callable ``(model, batch) -> loss_tensor``.
        max_grad_norm: Per-sample gradient-norm clip (DP-SGD).  Matches
            Abadi et al. (2016) default.
        noise_multiplier: Gaussian-noise multiplier for DP-SGD.  None to
            disable DP entirely.
        local_epochs: Number of local passes per federated round.
        learning_rate: SGD learning rate used on this client.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        loss_fn: Callable[[nn.Module, Any], torch.Tensor],
        max_grad_norm: float = 1.0,
        noise_multiplier: float | None = None,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if not _FLWR_AVAILABLE:
            logger.warning(_FLWR_INSTALL_HINT)

        self.model = model.to(device)
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

        self._privacy_engine: Any = None

    # ------------------------------------------------------------------
    # NumPyClient API
    # ------------------------------------------------------------------
    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return the current encoder parameters as NumPy arrays."""
        return _model_to_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Run one federated round's local training.

        Args:
            parameters: Global encoder parameters at round start.
            config: Flower-supplied config (ignored except for
                ``local_epochs`` and ``lr`` overrides).

        Returns:
            Tuple ``(updated_parameters, num_examples, metrics)``.
        """
        _load_parameters(self.model, parameters)
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("lr", self.learning_rate))

        optimiser = torch.optim.SGD(self.model.parameters(), lr=lr)
        if self.noise_multiplier is not None and _OPACUS_AVAILABLE:
            # PrivacyEngine wraps model + optimiser + dataloader in place.
            self._privacy_engine = PrivacyEngine()
            self.model, optimiser, loader = self._privacy_engine.make_private(
                module=self.model,
                optimizer=optimiser,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            loader = self.train_loader

        self.model.train()
        seen = 0
        loss_sum = 0.0
        for _epoch in range(epochs):
            for batch in loader:
                optimiser.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model, batch)
                loss.backward()
                if self._privacy_engine is None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                optimiser.step()
                batch_size = _infer_batch_size(batch)
                loss_sum += float(loss.detach().cpu()) * batch_size
                seen += batch_size

        mean_loss = loss_sum / max(seen, 1)
        metrics: dict[str, Any] = {"train_loss": mean_loss}
        if self._privacy_engine is not None:
            eps = self._privacy_engine.get_epsilon(delta=1e-5)
            metrics["dp_epsilon"] = float(eps)
        return _model_to_parameters(self.model), seen, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the global model against local data.

        The eval loop here is deliberately minimal — it only reports the
        mean reconstruction / contrastive loss.  Richer eval (silhouette,
        KNN top-1 on local archetype labels) belongs in the real thing.
        """
        _load_parameters(self.model, parameters)
        self.model.eval()
        loss_sum = 0.0
        seen = 0
        with torch.no_grad():
            for batch in self.train_loader:
                loss = self.loss_fn(self.model, batch)
                batch_size = _infer_batch_size(batch)
                loss_sum += float(loss.detach().cpu()) * batch_size
                seen += batch_size
        mean_loss = loss_sum / max(seen, 1)
        return mean_loss, seen, {"eval_loss": mean_loss}


def _infer_batch_size(batch: Any) -> int:
    """Best-effort batch-size probe.

    Handles the common ``(x, y)`` tuple case as well as plain tensors.
    """
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        first = batch[0]
        if hasattr(first, "shape") and len(first.shape) > 0:
            return int(first.shape[0])
    if hasattr(batch, "shape") and len(batch.shape) > 0:
        return int(batch.shape[0])
    return 1
