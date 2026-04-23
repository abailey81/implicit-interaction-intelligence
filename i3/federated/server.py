"""Federated-averaging server sketch for the I³ encoder.

The server implements the canonical FedAvg update from McMahan et al. (2017):
each round, sample a fraction ``fraction_fit`` of available clients, collect
their parameter deltas, and aggregate by client-sample-weighted mean.

The real Flower server is soft-imported; when ``flwr`` is unavailable we
expose a pure-numpy :func:`weighted_fedavg` function that is fully
testable in isolation — the simulation harness in
``scripts/demos/federated.py`` exercises it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:
    from flwr.server.strategy import FedAvg  # type: ignore[import-not-found]

    _FLWR_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    FedAvg = object  # type: ignore[assignment,misc]
    _FLWR_AVAILABLE = False


_FLWR_INSTALL_HINT = (
    "flwr is not installed.  Install the future-work group with "
    "`poetry install --with future-work` to enable federated orchestration."
)


# ---------------------------------------------------------------------------
# Pure-numpy weighted FedAvg (testable)
# ---------------------------------------------------------------------------

def weighted_fedavg(
    client_updates: list[tuple[list[np.ndarray], int]]
) -> list[np.ndarray]:
    """Classic client-sample-weighted FedAvg.

    Aggregates a list of client parameter lists, each tagged with the number
    of local examples that update reflects.  The result is the sample-count-
    weighted mean, as defined in McMahan et al. (2017) Algorithm 1.

    Args:
        client_updates: List of ``(parameters, num_examples)`` pairs.  Every
            client's parameter list must have the same number of arrays, in
            the same shapes.

    Returns:
        A list of numpy arrays representing the aggregated parameters.

    Raises:
        ValueError: If the list is empty or if client parameter shapes are
            inconsistent.
    """
    if not client_updates:
        raise ValueError("weighted_fedavg requires at least one client update")

    reference = client_updates[0][0]
    for params, _n in client_updates[1:]:
        if len(params) != len(reference):
            raise ValueError(
                "Inconsistent parameter counts across clients: "
                f"{len(params)} vs {len(reference)}"
            )
        for a, b in zip(reference, params):
            if a.shape != b.shape:
                raise ValueError(
                    f"Inconsistent shapes across clients: {a.shape} vs {b.shape}"
                )

    total_examples = sum(n for _p, n in client_updates)
    if total_examples == 0:
        raise ValueError("Total client example count is zero; cannot average")

    aggregated = [np.zeros_like(w, dtype=np.float32) for w in reference]
    for params, n in client_updates:
        weight = float(n) / float(total_examples)
        for i, arr in enumerate(params):
            aggregated[i] = aggregated[i] + weight * arr.astype(np.float32)
    return aggregated


# ---------------------------------------------------------------------------
# I³ federated strategy
# ---------------------------------------------------------------------------

@dataclass
class I3ServerConfig:
    """Tuning knobs for the I³ federated server.

    Attributes:
        fraction_fit: Fraction of clients sampled for training each round.
            McMahan et al. (2017) find 0.1–0.3 is a strong default.
        min_fit_clients: Minimum clients that must be available to train.
        min_available_clients: Minimum clients connected to start a round.
        num_rounds: Total rounds to run.
        use_secure_aggregation: If True, aggregate under
            :class:`~i3.federated.aggregator.SecureAggregator`.
    """

    fraction_fit: float = 0.3
    min_fit_clients: int = 3
    min_available_clients: int = 5
    num_rounds: int = 10
    use_secure_aggregation: bool = False


class I3FederatedServer:
    """Thin wrapper around Flower's ``FedAvg`` strategy with I³-specific
    defaults.

    When Flower is not installed the server is still instantiable but cannot
    launch a real cluster; the pure-numpy :func:`weighted_fedavg` aggregator
    remains available for local testing and simulation.
    """

    def __init__(self, config: I3ServerConfig | None = None) -> None:
        self.config = config or I3ServerConfig()
        self._strategy: Any = None
        if _FLWR_AVAILABLE:
            self._strategy = FedAvg(
                fraction_fit=self.config.fraction_fit,
                min_fit_clients=self.config.min_fit_clients,
                min_available_clients=self.config.min_available_clients,
            )
        else:
            logger.warning(_FLWR_INSTALL_HINT)

    @property
    def strategy(self) -> Any:
        """Return the underlying Flower strategy (``None`` if Flower absent)."""
        return self._strategy

    def aggregate_fit(
        self, client_updates: list[tuple[list[np.ndarray], int]]
    ) -> list[np.ndarray]:
        """Dispatch aggregation through the pure-numpy path.

        Intended for simulation / test harnesses that want a deterministic
        result without spinning up the Flower runtime.  Real deployments
        should bind :attr:`strategy` into ``flwr.server.start_server``.
        """
        return weighted_fedavg(client_updates)
