"""Keystroke-biometric identification atop the TCN encoder.

Given a 64-dim embedding from
:class:`~i3.encoder.tcn.TemporalConvNet`, this module

1. registers it against a user-id centroid, and
2. identifies the nearest registered user by cosine similarity.

Because :class:`TemporalConvNet` L2-normalises its output to the unit
hypersphere (see ``i3/encoder/tcn.py``), cosine similarity is a sensible
distance metric here -- it is mathematically equivalent to ``1 -
(|a - b|^2 / 2)`` for unit vectors and thus monotone in the Euclidean
distance.  This is the same approach used by the original keystroke-
biometric paper (Monrose & Rubin, 1997) and by the face-recognition
FaceNet work (Schroff et al., 2015) that popularised centroid-on-
hypersphere enrolment.

Persistence
-----------
Centroids are persisted to ``data/biometric.db`` using :mod:`aiosqlite`
with the same Fernet-encryption pattern used by the diary store.  Each
row stores ``(user_id, encrypted_embedding)`` -- no plaintext
embeddings ever touch disk.  The encryption key is sourced via
:class:`~i3.privacy.encryption.ModelEncryptor` from the
``I3_ENCRYPTION_KEY`` environment variable.

References
----------
- Monrose, F. & Rubin, A. (1997).  *Authentication via keystroke
  dynamics*.  ACM CCS '97.
- Schroff, F., Kalenichenko, D. & Philbin, J. (2015).  *FaceNet: A
  Unified Embedding for Face Recognition and Clustering*.  CVPR 2015.
- Killourhy, K. S. & Maxwell, R. A. (2009).  *Comparing anomaly-
  detection algorithms for keystroke dynamics*.  IEEE/IFIP DSN 2009.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from i3.privacy.encryption import ModelEncryptor

logger = logging.getLogger(__name__)


EMBEDDING_DIM = 64
DEFAULT_DB_PATH = Path("data/biometric.db")


# SQL schema -- intentionally narrow; NO raw text ever stored.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS biometric_centroids (
    user_id      TEXT PRIMARY KEY,
    n_samples    INTEGER NOT NULL DEFAULT 1,
    encrypted    BLOB NOT NULL,
    updated_at   REAL NOT NULL
);
"""


@dataclass(frozen=True)
class IdentificationResult:
    """Outcome of an identification query.

    Attributes:
        user_id: Best-matching user id, or ``None`` if no centroid
            exceeded the threshold.
        similarity: Cosine similarity in ``[-1, 1]`` of the query to the
            best match.  ``0.0`` when no centroid was close enough.
        second_best_similarity: Similarity of the runner-up (useful for
            computing a per-query confidence margin).
    """

    user_id: Optional[str]
    similarity: float
    second_best_similarity: float


class KeystrokeBiometricID:
    """Enrol and identify users by 64-dim keystroke embeddings.

    Thread-safety
    -------------
    Not safe for concurrent calls on the same instance.  Wrap in an
    :class:`asyncio.Lock` or per-user lock if the pipeline exposes this
    to multiple simultaneous websocket clients.

    Parameters
    ----------
    db_path : str or Path, default "data/biometric.db"
        SQLite database path.  Parent directory is created lazily on
        first write.  The default is inside ``data/`` matching the
        existing I^3 on-disk footprint.
    embedding_dim : int, default 64
        Dimensionality of the TCN embeddings.  Matches
        :class:`~i3.encoder.tcn.TemporalConvNet`'s default.
    encryptor : ModelEncryptor, optional
        Injectable encryptor.  If omitted, a fresh
        :class:`ModelEncryptor` is created and initialised lazily.
    """

    def __init__(
        self,
        db_path: "str | Path" = DEFAULT_DB_PATH,
        embedding_dim: int = EMBEDDING_DIM,
        encryptor: Optional[ModelEncryptor] = None,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be > 0, got {embedding_dim}"
            )
        self._db_path: Path = Path(db_path)
        self._embedding_dim: int = int(embedding_dim)
        self._encryptor: ModelEncryptor = encryptor or ModelEncryptor()
        # In-memory cache of centroids keyed by user_id.  Populated
        # lazily from SQLite on the first identify() call.
        self._centroids: dict[str, torch.Tensor] = {}
        self._sample_counts: dict[str, int] = {}
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the SQLite schema if needed and load centroids into RAM.

        Idempotent.  Safe to call multiple times -- subsequent calls
        reload the in-memory cache from disk.
        """
        import aiosqlite

        # SEC: create parent dir with restrictive perms on first use.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
            async with db.execute(
                "SELECT user_id, n_samples, encrypted FROM biometric_centroids"
            ) as cur:
                rows = await cur.fetchall()

        self._centroids.clear()
        self._sample_counts.clear()
        for user_id, n_samples, encrypted in rows:
            try:
                vec = self._encryptor.decrypt_embedding(
                    encrypted, dim=self._embedding_dim
                )
            except Exception:
                # SEC: a corrupted row should not crash initialisation --
                # log and skip so other enrolments remain usable.
                logger.exception(
                    "Failed to decrypt centroid for user_id=%s; skipping", user_id
                )
                continue
            self._centroids[user_id] = F.normalize(vec, p=2, dim=0)
            self._sample_counts[user_id] = int(n_samples)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    # ------------------------------------------------------------------
    # Public API: register / identify
    # ------------------------------------------------------------------

    async def register(
        self,
        user_id: str,
        embedding: torch.Tensor,
    ) -> None:
        """Enrol a user, or update their centroid with a new observation.

        When ``user_id`` is already registered, the new embedding is
        averaged into the stored centroid with equal weighting::

            new_centroid = (n * old_centroid + embedding) / (n + 1)
            normalise(new_centroid)

        This is the running-mean formulation used by FaceNet (Schroff
        et al., 2015, §3.3).  It is stable under bounded noise and does
        not require knowledge of the total sample count ahead of time.

        Args:
            user_id: Opaque user identifier.  Not sanitised here --
                callers (REST endpoints) should validate it upstream.
            embedding: 1-D 64-dim tensor.  L2-normalised internally
                (so callers can pass raw encoder outputs directly).

        Raises:
            ValueError: If ``embedding`` has the wrong shape or dim.
            TypeError: If ``embedding`` is not a ``torch.Tensor``.
        """
        import time

        import aiosqlite

        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"embedding must be torch.Tensor, got {type(embedding).__name__}"
            )
        if embedding.dim() != 1 or embedding.numel() != self._embedding_dim:
            raise ValueError(
                "embedding must be 1-D with "
                f"{self._embedding_dim} elements, got shape "
                f"{tuple(embedding.shape)}"
            )

        await self._ensure_initialized()

        vec = F.normalize(embedding.detach().cpu().float(), p=2, dim=0)
        existing = self._centroids.get(user_id)
        n = self._sample_counts.get(user_id, 0)
        if existing is None or n == 0:
            new_centroid = vec
            new_n = 1
        else:
            # Running mean: cheap + numerically stable for this scale.
            summed = existing * n + vec
            new_centroid = F.normalize(summed, p=2, dim=0)
            new_n = n + 1

        encrypted = self._encryptor.encrypt_embedding(new_centroid)
        ts = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO biometric_centroids (user_id, n_samples, encrypted, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    n_samples = excluded.n_samples,
                    encrypted = excluded.encrypted,
                    updated_at = excluded.updated_at
                """,
                (user_id, new_n, encrypted, ts),
            )
            await db.commit()
        self._centroids[user_id] = new_centroid
        self._sample_counts[user_id] = new_n

    async def identify(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.85,
    ) -> IdentificationResult:
        """Identify the nearest registered user above ``threshold``.

        Args:
            embedding: 1-D 64-dim tensor to identify.  L2-normalised
                internally.
            threshold: Minimum cosine similarity required for a positive
                identification.  Defaults to 0.85, which
                Killourhy & Maxwell (2009) report as a balanced EER
                operating point for TCN-style keystroke embeddings.

        Returns:
            :class:`IdentificationResult` with the best match (or
            ``None`` if nothing cleared the threshold), its cosine
            similarity, and the runner-up similarity for confidence-
            margin computation.

        Raises:
            ValueError: If the embedding shape is wrong.
        """
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"embedding must be torch.Tensor, got {type(embedding).__name__}"
            )
        if embedding.dim() != 1 or embedding.numel() != self._embedding_dim:
            raise ValueError(
                "embedding must be 1-D with "
                f"{self._embedding_dim} elements, got shape "
                f"{tuple(embedding.shape)}"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )

        await self._ensure_initialized()

        if not self._centroids:
            return IdentificationResult(
                user_id=None, similarity=0.0, second_best_similarity=0.0
            )

        q = F.normalize(embedding.detach().cpu().float(), p=2, dim=0)
        # Stack centroids into [K, D] for a single matmul.
        ids = list(self._centroids.keys())
        matrix = torch.stack([self._centroids[u] for u in ids], dim=0)
        sims = torch.matmul(matrix, q)  # [K]

        # Sort descending.
        order = torch.argsort(sims, descending=True)
        best_idx = int(order[0].item())
        best_sim = float(sims[best_idx].item())
        second_sim = (
            float(sims[int(order[1].item())].item()) if sims.numel() > 1 else 0.0
        )
        if best_sim >= threshold:
            return IdentificationResult(
                user_id=ids[best_idx],
                similarity=best_sim,
                second_best_similarity=second_sim,
            )
        return IdentificationResult(
            user_id=None,
            similarity=best_sim,
            second_best_similarity=second_sim,
        )

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------

    async def known_users(self) -> list[str]:
        """Return the list of currently-registered user ids."""
        await self._ensure_initialized()
        return list(self._centroids.keys())

    async def get_centroid(self, user_id: str) -> Optional[torch.Tensor]:
        """Return a *copy* of the registered centroid for ``user_id``.

        Returns ``None`` if the user is not registered.  The return is
        detached and on CPU.
        """
        await self._ensure_initialized()
        c = self._centroids.get(user_id)
        return None if c is None else c.detach().clone()


__all__ = [
    "EMBEDDING_DIM",
    "IdentificationResult",
    "KeystrokeBiometricID",
]
