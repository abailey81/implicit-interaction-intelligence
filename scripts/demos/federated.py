#!/usr/bin/env python3
"""Run a local 3-client federated-learning simulation of the I³ encoder.

This is a **sketch-level demo**.  It exercises the
:class:`~i3.federated.client.I3FederatedClient`,
:class:`~i3.federated.server.I3FederatedServer`, and the pure-numpy
:func:`~i3.federated.server.weighted_fedavg` aggregator, with optional
integration against ``flwr.simulation.start_simulation`` when Flower is
installed.

Usage::

    python scripts/run_federated_demo.py --num-clients 3 --num-rounds 5

If ``flwr`` is not installed the script prints an install hint and exits
with status 2 rather than crashing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from i3.federated.client import I3FederatedClient
from i3.federated.server import I3FederatedServer, I3ServerConfig, weighted_fedavg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("federated_demo")


_INSTALL_HINT = (
    "flwr is not installed.  Install the future-work group with "
    "`poetry install --with future-work` to run the Flower-backed "
    "simulation.  Falling back to the pure-numpy aggregator."
)


# ---------------------------------------------------------------------------
# Minimal toy model shaped like the TCN's input projection — enough to show
# the shape of a real federated pipeline without requiring the full encoder.
# ---------------------------------------------------------------------------

class _ToyEncoderHead(nn.Module):
    """A trivial linear head for the demo only."""

    def __init__(self, input_dim: int = 32, embedding_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.linear(x), dim=-1)


def _make_fake_dataloader(rng: np.random.Generator, n: int = 128) -> DataLoader[Any]:
    """Create a tiny synthetic dataset of 32-dim feature vectors."""
    x = torch.tensor(rng.standard_normal((n, 32)).astype(np.float32))
    y = torch.tensor(rng.integers(0, 8, size=(n,)).astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def _loss_fn(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Stand-in contrastive loss — cosine similarity to a fixed class anchor.

    Good enough to produce non-trivial gradients for the demo.
    """
    x, y = batch
    emb = model(x)
    # Anchor = mean embedding for each class in the current batch.
    anchors = torch.zeros_like(emb)
    for c in torch.unique(y):
        mask = (y == c)
        anchors[mask] = emb[mask].mean(dim=0, keepdim=True)
    return (1.0 - (emb * anchors).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_pure_numpy(num_clients: int, num_rounds: int, seed: int) -> None:
    """Run the fallback aggregator-only simulation.

    Each client fits once locally on their synthetic data; the server
    aggregates by :func:`weighted_fedavg`.  No Flower required.
    """
    rng = np.random.default_rng(seed)
    clients: list[I3FederatedClient] = []
    for i in range(num_clients):
        model = _ToyEncoderHead()
        loader = _make_fake_dataloader(rng)
        clients.append(I3FederatedClient(model, loader, _loss_fn, local_epochs=1))

    # Round 0 snapshot.
    params = [p.detach().cpu().numpy() for p in clients[0].model.state_dict().values()]

    for rnd in range(num_rounds):
        updates: list[tuple[list[np.ndarray], int]] = []
        for c in clients:
            new_params, n, metrics = c.fit(params, {})
            updates.append((new_params, n))
            logger.info(
                "round=%d client=%d examples=%d loss=%.4f",
                rnd,
                clients.index(c),
                n,
                metrics["train_loss"],
            )
        params = weighted_fedavg(updates)
    logger.info("pure-numpy simulation completed (%d rounds)", num_rounds)


def run_flower_simulation(num_clients: int, num_rounds: int, seed: int) -> int:
    """Run the real Flower simulation if Flower is installed."""
    try:
        import flwr  # type: ignore[import-not-found]
        from flwr.simulation import start_simulation  # type: ignore[import-not-found]
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    rng = np.random.default_rng(seed)

    def client_fn(cid: str) -> I3FederatedClient:
        model = _ToyEncoderHead()
        loader = _make_fake_dataloader(rng)
        return I3FederatedClient(model, loader, _loss_fn, local_epochs=1)

    server = I3FederatedServer(
        I3ServerConfig(
            fraction_fit=0.3,
            min_fit_clients=max(1, num_clients // 3),
            min_available_clients=num_clients,
            num_rounds=num_rounds,
        )
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=server.strategy,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="I³ federated-learning sketch demo")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=("auto", "flower", "numpy"),
        default="auto",
        help="auto picks flower if available, else pure-numpy aggregator.",
    )
    args = parser.parse_args()

    if args.backend == "flower":
        return run_flower_simulation(args.num_clients, args.num_rounds, args.seed)
    if args.backend == "numpy":
        run_pure_numpy(args.num_clients, args.num_rounds, args.seed)
        return 0

    try:
        import flwr  # noqa: F401  -- presence probe
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        run_pure_numpy(args.num_clients, args.num_rounds, args.seed)
        return 0
    return run_flower_simulation(args.num_clients, args.num_rounds, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
