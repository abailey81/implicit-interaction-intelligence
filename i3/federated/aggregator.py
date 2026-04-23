"""Secure-aggregation sketch for federated updates.

.. warning::
    **This is a pedagogical sketch, not production cryptography.**  The XOR
    additive-mask protocol implemented here has well-known weaknesses against
    malicious-server collusion and does not handle dropouts — both of which
    are addressed in the Bonawitz et al. (2017) production protocol.  It is
    included because a *federated-learning sketch without a secure-aggregation
    surface* would be misleading.  For production, integrate with
    `flwr-secure-aggregation` or similar audited implementations.

References
----------
* Bonawitz, K. et al. (2017). *Practical secure aggregation for privacy-
  preserving machine learning.*  CCS.
* Dwork, C., Roth, A. (2014). *The algorithmic foundations of differential
  privacy.*  Foundations and Trends in Theoretical Computer Science 9(3-4).
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask generation helpers
# ---------------------------------------------------------------------------

def _generate_pairwise_mask(
    seed: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Generate a deterministic pseudo-random mask keyed by *seed*.

    In the real Bonawitz protocol this is replaced by a keyed PRG over a
    Diffie–Hellman-exchanged shared secret.  Here it is ``numpy.random``
    seeded on a shared integer — **cryptographically inadequate** but
    sufficient to demonstrate the additive-mask cancellation property.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class MaskedUpdate:
    """A client parameter update obscured by pairwise additive masks.

    Attributes:
        client_id: Unique identifier of the contributing client.
        masked_parameters: Parameters with pairwise masks applied.
        num_examples: Number of local training examples this update reflects.
        peer_seeds: Shared PRG seeds — one per peer-client pair.  In the
            real protocol these would be computed from ECDH exchanges; here
            they are cleartext integers.
    """

    client_id: int
    masked_parameters: list[np.ndarray]
    num_examples: int
    peer_seeds: dict[int, int] = field(default_factory=dict)


class SecureAggregator:
    """Pedagogical additive-mask secure aggregator.

    The protocol: each pair of clients *(i, j)* shares a seed *s_ij*.  Client
    *i* adds ``+mask(s_ij)`` to their update if ``i < j`` and subtracts it
    otherwise.  When the server sums all masked updates, every mask cancels
    pairwise, leaving the unmasked sum — but no *individual* client update
    is visible to the server in clear form.

    Again: this is **not** production crypto.  Read the docstring at the top
    of the module before using this for anything real.
    """

    def __init__(self) -> None:
        self._pending: list[MaskedUpdate] = []

    # ------------------------------------------------------------------
    # Client-side helpers
    # ------------------------------------------------------------------
    @staticmethod
    def mask_update(
        client_id: int,
        parameters: list[np.ndarray],
        peer_seeds: dict[int, int],
    ) -> list[np.ndarray]:
        """Apply pairwise additive masks to *parameters*.

        Args:
            client_id: This client's integer id.
            parameters: The plaintext parameter list to mask.
            peer_seeds: Map from peer client id to shared PRG seed.  The
                caller is responsible for filling this with the correct
                seeds before invocation.

        Returns:
            The masked parameter list (same shapes as input).
        """
        masked = [p.copy() for p in parameters]
        for peer_id, seed in peer_seeds.items():
            sign = 1.0 if client_id < peer_id else -1.0
            for i, p in enumerate(masked):
                mask = _generate_pairwise_mask(seed, p.shape)
                masked[i] = p + sign * mask
        return masked

    # ------------------------------------------------------------------
    # Server-side state
    # ------------------------------------------------------------------
    def submit(self, update: MaskedUpdate) -> None:
        """Submit a client's masked update to the server buffer."""
        self._pending.append(update)

    def clear(self) -> None:
        """Clear all buffered updates (end-of-round cleanup)."""
        self._pending.clear()

    def aggregate(
        self, dropped_clients: set[int] | None = None
    ) -> list[np.ndarray]:
        """Aggregate the buffered masked updates into the plaintext sum.

        When clients drop out after their seeds have been distributed, the
        server requires the surviving clients to disclose the dropped
        clients' pairwise masks so they can be subtracted post-hoc — this
        sketch handles the "no dropouts" happy path only; dropouts raise.

        Args:
            dropped_clients: Clients that failed to submit.  **Must be
                empty** in this sketch; supplied for forward-compatibility.

        Returns:
            The sum of unmasked parameter lists, sample-weighted.

        Raises:
            NotImplementedError: If ``dropped_clients`` is non-empty.
            ValueError: If no updates have been submitted.
        """
        if dropped_clients:
            raise NotImplementedError(
                "Dropout recovery is not implemented in this sketch; "
                "see Bonawitz et al. (2017) §3.3."
            )
        if not self._pending:
            raise ValueError("SecureAggregator.aggregate: no pending updates")

        total_examples = sum(u.num_examples for u in self._pending)
        if total_examples == 0:
            raise ValueError("All pending updates carry zero examples")

        reference = self._pending[0].masked_parameters
        aggregated = [np.zeros_like(w, dtype=np.float32) for w in reference]
        for update in self._pending:
            weight = float(update.num_examples) / float(total_examples)
            for i, arr in enumerate(update.masked_parameters):
                aggregated[i] = aggregated[i] + weight * arr.astype(np.float32)
        return aggregated


# ---------------------------------------------------------------------------
# Convenience: generate a shared seed for a pair of clients
# ---------------------------------------------------------------------------

def generate_shared_seed() -> int:
    """Generate a cryptographically random seed in the range ``[0, 2**31)``.

    The seed is used by :func:`_generate_pairwise_mask`.  In production this
    would be replaced by a Diffie–Hellman-exchanged shared secret.
    """
    return int.from_bytes(secrets.token_bytes(4), "big") & 0x7FFFFFFF
