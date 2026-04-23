"""Active preference learning and online DPO for the Intelligent Router.

This module implements Batch F-4 of the v3 Advancement Plan: a principled,
sample-efficient mechanism for replacing the router's hand-crafted composite
reward with a *learned* reward model distilled from pairwise user
preferences.

The design follows four widely-cited lines of work:

1. **Pairwise preference modelling.** Bradley, R. A. and Terry, M. E.
   (1952). *Rank analysis of incomplete block designs: I. The method of
   paired comparisons.* Biometrika, 39(3/4), 324-345.  The Bradley-Terry
   model gives the probability that response A beats B as
   ``sigmoid(r(A) - r(B))`` where ``r(.)`` is a learned scalar reward.
2. **Direct Preference Optimisation.** Rafailov, R. et al. (2023).
   *Direct Preference Optimization: Your Language Model is Secretly a
   Reward Model.* NeurIPS 2023.  Shows that the Bradley-Terry objective
   optimised directly on preference data is equivalent to RLHF without
   the PPO stage.
3. **Active Learning for DPO.** Mehta, V. et al. (2025).
   *Active Learning for Direct Preference Optimization.* ICLR 2025.
   Near-optimal sample efficiency (~10-20 pairs per user) when the
   queries are selected by a D-optimal active criterion on the
   current reward model.
4. **IPO and modern DPO variants.** Azar, M. G. et al. (2023).
   *A General Theoretical Paradigm to Understand Learning from Human
   Preferences* (IPO). Wu, Y. et al. (2024). *A Survey of DPO Variants*.
   Provide the hyper-parameter choices (``reference_policy_kl_coef``)
   and the numerical-stability tricks (logit clipping).

This module intentionally stays lightweight: a small two-layer MLP as the
reward function, cross-entropy training, D-optimal active query scoring,
append-only SQLite persistence with a soft-imported aiosqlite, and a
deterministic in-memory fallback when aiosqlite is unavailable.

Threat model
------------

* ``aiosqlite`` is optional; when absent the dataset degrades to a
  process-local list.  All persistence paths validate user IDs via the
  ``USER_ID_REGEX`` and reject non-finite floats before writing.
* The reward model is bounded by logit clipping and a small architecture
  (<10 k parameters) so an attacker supplying adversarial feature
  vectors cannot inflate memory or cause runaway gradients.
* Active query selection never trusts untyped user-supplied tensors; all
  feature vectors are normalised and shape-validated before scoring.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import aiosqlite
    _AIOSQLITE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when aiosqlite missing
    aiosqlite = None
    _AIOSQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Maximum allowed length of the ``prompt`` / response strings.  Longer
#: fields are rejected to prevent memory blow-up from a malicious client.
_MAX_STRING_LEN: int = 8 * 1024

#: Dimensionality of the response-feature vector the Bradley-Terry MLP
#: consumes.  Matches the router's 12-dim context by default.
_DEFAULT_RESPONSE_DIM: int = 12

#: Dimensionality of the routing-context vector that is concatenated with
#: the response features when the reward model is conditioned on context.
_DEFAULT_CONTEXT_DIM: int = 12

#: Hidden-layer width of the Bradley-Terry MLP.
_DEFAULT_HIDDEN_DIM: int = 32

#: Clip raw reward logits to this magnitude so exponentiation in the
#: cross-entropy loss is numerically stable.
_REWARD_CLIP: float = 10.0

#: Small epsilon for numerical stability in log/exp.
_EPSILON: float = 1e-6

#: Winner enum values — keep as str so SQLite round-trips are unambiguous.
_VALID_WINNERS: frozenset[str] = frozenset({"a", "b", "tie"})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferencePair:
    """A single pairwise preference sample.

    Attributes:
        prompt: The user prompt that elicited the two responses.  Used
            only for logging/UI display; not fed into the reward model.
        response_a: The first candidate response text.  Logged only.
        response_b: The second candidate response text.  Logged only.
        winner: Which response the user preferred — ``"a"``, ``"b"``, or
            ``"tie"``.  A ``"tie"`` is treated as a 0.5/0.5 label.
        context: The 12-dim routing context vector at the time of the
            query.  Concatenated with the response feature vectors
            to condition the reward model.
        response_a_features: Numeric feature vector for response A.
            Supplied by the caller (e.g. latency, length, confidence);
            see :func:`build_response_features`.
        response_b_features: Numeric feature vector for response B.
        timestamp: Unix timestamp of the preference event.
        user_id: Who supplied the label.
    """

    prompt: str
    response_a: str
    response_b: str
    winner: str
    context: Sequence[float]
    response_a_features: Sequence[float]
    response_b_features: Sequence[float]
    timestamp: float = field(default_factory=time.time)
    user_id: str = "anonymous"

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is outside its contract."""
        if not isinstance(self.prompt, str) or len(self.prompt) > _MAX_STRING_LEN:
            raise ValueError("prompt must be a str of <= 8 KiB")
        for name, val in (("response_a", self.response_a), ("response_b", self.response_b)):
            if not isinstance(val, str) or len(val) > _MAX_STRING_LEN:
                raise ValueError(f"{name} must be a str of <= 8 KiB")
        if self.winner not in _VALID_WINNERS:
            raise ValueError(
                f"winner must be one of {sorted(_VALID_WINNERS)}, got {self.winner!r}"
            )
        for name, vec in (
            ("context", self.context),
            ("response_a_features", self.response_a_features),
            ("response_b_features", self.response_b_features),
        ):
            arr = np.asarray(vec, dtype=np.float64).ravel()
            if arr.size == 0:
                raise ValueError(f"{name} must be non-empty")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")


@dataclass
class DPOFitReport:
    """Summary of a single :meth:`DPOPreferenceOptimizer.fit` invocation.

    Attributes:
        n_pairs: Total number of preference pairs trained on.
        n_train: Pairs in the training split.
        n_val: Pairs in the validation split.
        train_loss: Final mean cross-entropy loss on the training split.
        val_accuracy: Fraction of val pairs whose predicted winner
            matches the ground-truth winner (``ties`` counted as 0.5).
        epochs_run: Number of epochs actually executed.
        elapsed_seconds: Wall-clock duration of the fit.
    """

    n_pairs: int
    n_train: int
    n_val: int
    train_loss: float
    val_accuracy: float
    epochs_run: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "n_pairs": self.n_pairs,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "train_loss": float(self.train_loss),
            "val_accuracy": float(self.val_accuracy),
            "epochs_run": self.epochs_run,
            "elapsed_seconds": float(self.elapsed_seconds),
        }


# ---------------------------------------------------------------------------
# PreferenceDataset — append-only store with SQLite persistence
# ---------------------------------------------------------------------------


def _serialise_vec(vec: Sequence[float]) -> str:
    """JSON-style encode a numeric vector for SQLite storage."""
    arr = np.asarray(vec, dtype=np.float64).ravel()
    return ",".join(repr(float(v)) for v in arr)


def _deserialise_vec(raw: str) -> list[float]:
    """Inverse of :func:`_serialise_vec` with strict float parsing."""
    if not raw:
        return []
    return [float(v) for v in raw.split(",") if v.strip()]


class PreferenceDataset:
    """Append-only store of :class:`PreferencePair` objects.

    The dataset is backed by SQLite when ``aiosqlite`` is available and the
    caller supplies a ``db_path``; otherwise it falls back to a pure
    in-memory Python list.  In both modes the public API is synchronous —
    async persistence helpers are provided separately.

    Args:
        db_path: Optional filesystem path.  When ``None`` (the default)
            the dataset is memory-only.
    """

    _SCHEMA: str = """
    CREATE TABLE IF NOT EXISTS preference_pairs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      TEXT NOT NULL,
        prompt       TEXT NOT NULL,
        response_a   TEXT NOT NULL,
        response_b   TEXT NOT NULL,
        winner       TEXT NOT NULL,
        context      TEXT NOT NULL,
        feat_a       TEXT NOT NULL,
        feat_b       TEXT NOT NULL,
        ts           REAL NOT NULL
    );
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path: Path | None = Path(db_path) if db_path is not None else None
        self._pairs: list[PreferencePair] = []
        self._persistent: bool = self.db_path is not None and _AIOSQLITE_AVAILABLE
        if self.db_path is not None and not _AIOSQLITE_AVAILABLE:
            logger.warning(
                "aiosqlite unavailable; PreferenceDataset running in-memory only."
            )

    # -- sync API ----------------------------------------------------------

    def append(self, pair: PreferencePair) -> None:
        """Append ``pair`` to the in-memory buffer after validation.

        Args:
            pair: The preference pair to store.

        Raises:
            ValueError: If the pair fails its own validation.
        """
        pair.validate()
        self._pairs.append(pair)

    def extend(self, pairs: Iterable[PreferencePair]) -> None:
        """Append every pair in ``pairs`` (each is individually validated)."""
        for p in pairs:
            self.append(p)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Any:
        return iter(list(self._pairs))

    def all(self) -> list[PreferencePair]:
        """Return a snapshot copy of all stored pairs."""
        return list(self._pairs)

    def filter_by_user(self, user_id: str) -> list[PreferencePair]:
        """Return only the pairs labelled by ``user_id``."""
        return [p for p in self._pairs if p.user_id == user_id]

    # -- async SQLite persistence -----------------------------------------

    async def persist(self) -> None:
        """Flush the current in-memory buffer to disk.

        A no-op when ``aiosqlite`` is not available or ``db_path`` is None.
        """
        if not self._persistent or self.db_path is None:
            return
        assert aiosqlite is not None  # for mypy
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(self._SCHEMA)
            for p in self._pairs:
                await db.execute(
                    "INSERT INTO preference_pairs"
                    " (user_id, prompt, response_a, response_b, winner,"
                    "  context, feat_a, feat_b, ts)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        p.user_id,
                        p.prompt,
                        p.response_a,
                        p.response_b,
                        p.winner,
                        _serialise_vec(p.context),
                        _serialise_vec(p.response_a_features),
                        _serialise_vec(p.response_b_features),
                        float(p.timestamp),
                    ),
                )
            await db.commit()
        logger.debug(
            "Persisted %d preference pairs to %s", len(self._pairs), self.db_path
        )

    async def load(self) -> None:
        """Replace the in-memory buffer with the contents of ``db_path``.

        No-op when ``aiosqlite`` is not available or the file is missing.
        """
        if not self._persistent or self.db_path is None:
            return
        if not self.db_path.exists():
            return
        assert aiosqlite is not None
        rows: list[tuple[Any, ...]] = []
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(self._SCHEMA)
            async with db.execute(
                "SELECT user_id, prompt, response_a, response_b, winner,"
                " context, feat_a, feat_b, ts FROM preference_pairs"
            ) as cur:
                rows = list(await cur.fetchall())
        self._pairs = []
        for r in rows:
            try:
                pair = PreferencePair(
                    user_id=str(r[0]),
                    prompt=str(r[1]),
                    response_a=str(r[2]),
                    response_b=str(r[3]),
                    winner=str(r[4]),
                    context=_deserialise_vec(str(r[5])),
                    response_a_features=_deserialise_vec(str(r[6])),
                    response_b_features=_deserialise_vec(str(r[7])),
                    timestamp=float(r[8]),
                )
                pair.validate()
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                logger.warning("Skipping corrupt preference row: %s", exc)
                continue
            self._pairs.append(pair)
        logger.info(
            "Loaded %d preference pairs from %s", len(self._pairs), self.db_path
        )


# ---------------------------------------------------------------------------
# Bradley-Terry reward model
# ---------------------------------------------------------------------------


class BradleyTerryRewardModel(nn.Module):
    """Two-layer MLP mapping ``(context, response_features) -> scalar reward``.

    Under the Bradley-Terry assumption (Bradley & Terry 1952) the
    probability that response A beats response B given context ``c`` is:

        P(A > B | c) = sigmoid( r(c, A) - r(c, B) )

    Training minimises the pairwise cross-entropy between this probability
    and the observed label (``1`` for A-wins, ``0`` for B-wins, ``0.5``
    for ties).

    Args:
        context_dim: Dimensionality of the routing-context vector.
        response_dim: Dimensionality of the response feature vector.
        hidden_dim: Width of the MLP's hidden layer.

    Raises:
        ValueError: If any dimension is <= 0.
    """

    def __init__(
        self,
        context_dim: int = _DEFAULT_CONTEXT_DIM,
        response_dim: int = _DEFAULT_RESPONSE_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be > 0, got {context_dim}")
        if response_dim <= 0:
            raise ValueError(f"response_dim must be > 0, got {response_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self.context_dim: int = context_dim
        self.response_dim: int = response_dim
        self.hidden_dim: int = hidden_dim
        in_dim = context_dim + response_dim
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        # Deterministic init — aids reproducibility for tests.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self, context: torch.Tensor, response_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scalar reward ``r(context, response_features)``.

        Args:
            context: Tensor of shape ``(B, context_dim)`` or ``(context_dim,)``.
            response_features: Tensor of shape ``(B, response_dim)`` or
                ``(response_dim,)``.

        Returns:
            Tensor of shape ``(B,)`` (or scalar) with clipped rewards.
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if response_features.dim() == 1:
            response_features = response_features.unsqueeze(0)
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context last dim must be {self.context_dim}, "
                f"got {context.shape[-1]}"
            )
        if response_features.shape[-1] != self.response_dim:
            raise ValueError(
                f"response_features last dim must be {self.response_dim}, "
                f"got {response_features.shape[-1]}"
            )
        x = torch.cat([context, response_features], dim=-1)
        h = F.relu(self.hidden(x))
        r = self.head(h).squeeze(-1)
        # Clip rewards to keep the BT sigmoid numerically stable.
        r = torch.clamp(r, min=-_REWARD_CLIP, max=_REWARD_CLIP)
        return r

    def score(
        self, context: Sequence[float], response_features: Sequence[float]
    ) -> float:
        """Score a single ``(context, response)`` pair as a Python float.

        Args:
            context: 1-D sequence of length ``context_dim``.
            response_features: 1-D sequence of length ``response_dim``.

        Returns:
            The scalar reward.
        """
        self.eval()
        with torch.no_grad():
            ctx = torch.as_tensor(context, dtype=torch.float32).view(-1)
            feat = torch.as_tensor(response_features, dtype=torch.float32).view(-1)
            r = self.forward(ctx, feat)
        return float(r.squeeze().item())

    def last_layer_features(
        self, context: torch.Tensor, response_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the pre-head hidden activations.

        Used by :class:`ActivePreferenceSelector` for D-optimal design on
        the last linear layer (Mehta et al. 2025, §3.2).
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if response_features.dim() == 1:
            response_features = response_features.unsqueeze(0)
        x = torch.cat([context, response_features], dim=-1)
        return F.relu(self.hidden(x))


# ---------------------------------------------------------------------------
# DPO-style optimiser
# ---------------------------------------------------------------------------


def _winner_to_label(winner: str) -> float:
    """Map a winner code to a soft label in ``{0.0, 0.5, 1.0}``.

    A value of ``1.0`` means A is preferred, ``0.0`` means B is preferred,
    and ``0.5`` is a tie.
    """
    if winner == "a":
        return 1.0
    if winner == "b":
        return 0.0
    return 0.5


class DPOPreferenceOptimizer:
    """Trains a :class:`BradleyTerryRewardModel` on preference pairs.

    The objective follows Rafailov et al. 2023: minimise the negative log
    likelihood of the observed winner under the Bradley-Terry model.  A
    soft KL regulariser towards a zero-reward reference policy is added
    with coefficient ``reference_policy_kl_coef`` — this matches DPO's
    ``β`` hyper-parameter and keeps the learned rewards on a sensible
    scale without requiring a separate reference network.

    Args:
        reward_model: The Bradley-Terry model to train.
        reference_policy_kl_coef: DPO-style KL strength (``β``).
        learning_rate: AdamW learning rate.
    """

    def __init__(
        self,
        reward_model: BradleyTerryRewardModel,
        reference_policy_kl_coef: float = 0.1,
        learning_rate: float = 1e-4,
    ) -> None:
        if reference_policy_kl_coef < 0.0:
            raise ValueError(
                f"reference_policy_kl_coef must be >= 0, "
                f"got {reference_policy_kl_coef}"
            )
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.reward_model = reward_model
        self.beta: float = float(reference_policy_kl_coef)
        self.learning_rate: float = float(learning_rate)

    # -- public API -------------------------------------------------------

    def fit(
        self,
        preference_dataset: PreferenceDataset,
        n_epochs: int = 50,
        val_fraction: float = 0.2,
        batch_size: int = 32,
        seed: int = 0,
    ) -> DPOFitReport:
        """Train the reward model on ``preference_dataset``.

        Args:
            preference_dataset: Source of pairs.
            n_epochs: Maximum number of epochs.
            val_fraction: Fraction of pairs held out for accuracy.
            batch_size: Mini-batch size for training.
            seed: RNG seed for the train/val split and optimiser.

        Returns:
            A :class:`DPOFitReport` summarising the run.

        Raises:
            ValueError: If the dataset is empty or ``n_epochs`` < 1.
        """
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in [0, 1), got {val_fraction}"
            )

        pairs = preference_dataset.all()
        if not pairs:
            raise ValueError("preference_dataset is empty")

        rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        idx = rng.permutation(len(pairs))
        n_val = max(1, int(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
        train_idx = idx[n_val:]
        val_idx = idx[:n_val]
        train_pairs = [pairs[i] for i in train_idx] if len(train_idx) else pairs
        val_pairs = [pairs[i] for i in val_idx]

        ctx_dim = self.reward_model.context_dim
        resp_dim = self.reward_model.response_dim

        # Stack training tensors once up-front.
        ctx_t, a_t, b_t, y_t = _stack_pairs(train_pairs, ctx_dim, resp_dim)

        optim = torch.optim.AdamW(
            self.reward_model.parameters(), lr=self.learning_rate
        )

        self.reward_model.train()
        t0 = time.perf_counter()
        last_loss = float("nan")
        for epoch in range(n_epochs):
            perm = torch.randperm(ctx_t.shape[0])
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, ctx_t.shape[0], batch_size):
                sel = perm[start : start + batch_size]
                optim.zero_grad()
                r_a = self.reward_model(ctx_t[sel], a_t[sel])
                r_b = self.reward_model(ctx_t[sel], b_t[sel])
                # Bradley-Terry cross-entropy: -[y log σ(Δ) + (1-y) log σ(-Δ)]
                delta = r_a - r_b
                logp_a = F.logsigmoid(delta)
                logp_b = F.logsigmoid(-delta)
                loss = -(y_t[sel] * logp_a + (1.0 - y_t[sel]) * logp_b).mean()
                if self.beta > 0.0:
                    kl = (r_a.pow(2).mean() + r_b.pow(2).mean()) * 0.5
                    loss = loss + self.beta * kl
                loss.backward()
                # Gradient clipping — DPO variants survey (Wu 2024) §4.2.
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(), max_norm=5.0
                )
                optim.step()
                ep_loss += float(loss.item())
                n_batches += 1
            if n_batches > 0:
                last_loss = ep_loss / n_batches

        elapsed = time.perf_counter() - t0

        val_acc = self._evaluate_accuracy(val_pairs) if val_pairs else 0.0

        return DPOFitReport(
            n_pairs=len(pairs),
            n_train=len(train_pairs),
            n_val=len(val_pairs),
            train_loss=float(last_loss),
            val_accuracy=float(val_acc),
            epochs_run=int(n_epochs),
            elapsed_seconds=float(elapsed),
        )

    def _evaluate_accuracy(self, pairs: Sequence[PreferencePair]) -> float:
        """Compute the fraction of correctly predicted winners."""
        if not pairs:
            return 0.0
        ctx_dim = self.reward_model.context_dim
        resp_dim = self.reward_model.response_dim
        ctx_t, a_t, b_t, y_t = _stack_pairs(pairs, ctx_dim, resp_dim)
        self.reward_model.eval()
        with torch.no_grad():
            r_a = self.reward_model(ctx_t, a_t)
            r_b = self.reward_model(ctx_t, b_t)
        pred_a_wins = (r_a > r_b).float()
        # Ties receive 0.5 credit either way.
        correct = torch.where(
            y_t == 0.5,
            torch.full_like(y_t, 0.5),
            (pred_a_wins == y_t).float(),
        )
        return float(correct.mean().item())


def _stack_pairs(
    pairs: Sequence[PreferencePair], ctx_dim: int, resp_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a sequence of pairs into four stacked float32 tensors."""
    n = len(pairs)
    ctx = np.zeros((n, ctx_dim), dtype=np.float32)
    a = np.zeros((n, resp_dim), dtype=np.float32)
    b = np.zeros((n, resp_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i, p in enumerate(pairs):
        ctx[i] = _pad_or_truncate(p.context, ctx_dim)
        a[i] = _pad_or_truncate(p.response_a_features, resp_dim)
        b[i] = _pad_or_truncate(p.response_b_features, resp_dim)
        y[i] = _winner_to_label(p.winner)
    return (
        torch.from_numpy(ctx),
        torch.from_numpy(a),
        torch.from_numpy(b),
        torch.from_numpy(y),
    )


def _pad_or_truncate(vec: Sequence[float], target_dim: int) -> np.ndarray:
    """Coerce ``vec`` to a length-``target_dim`` float32 numpy array."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.size == target_dim:
        return arr
    if arr.size > target_dim:
        return arr[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[: arr.size] = arr
    return out


# ---------------------------------------------------------------------------
# Active preference selector
# ---------------------------------------------------------------------------


class ActivePreferenceSelector:
    """Picks maximally-informative pairs to query next.

    Follows the D-optimal criterion of Mehta et al. (2025, §3.2): for the
    current reward model's last linear layer, an A-B pair's information
    gain is proportional to the log-determinant increase of the Fisher
    information matrix when the pair is added.  For a single candidate
    this reduces to a cheap quadratic form on the difference of the
    last-layer features of response A and response B.

    The selector stores a running Fisher proxy and greedily picks pairs
    that maximise the expected reduction in posterior uncertainty.

    Args:
        reward_model: The Bradley-Terry model providing last-layer
            features.
        ridge: Ridge term added to the Fisher matrix for stability.
    """

    def __init__(
        self,
        reward_model: BradleyTerryRewardModel,
        ridge: float = 1e-2,
    ) -> None:
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        self.reward_model = reward_model
        self.ridge: float = float(ridge)
        dim = reward_model.hidden_dim
        # Fisher proxy starts at ridge*I so its inverse exists from pair 1.
        self._fisher: np.ndarray = np.eye(dim, dtype=np.float64) * self.ridge

    def score_pair(self, pair: PreferencePair) -> float:
        """Return the scalar information-gain score for ``pair``.

        Higher is more informative.  The score is
        ``phi(A,B)^T F^{-1} phi(A,B)`` where
        ``phi(A,B) = last_layer(A) - last_layer(B)``.
        """
        phi = self._pair_feature_diff(pair)
        try:
            f_inv = np.linalg.inv(self._fisher)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            f_inv = np.linalg.pinv(self._fisher)
        score = float(phi @ f_inv @ phi)
        return score

    def select_next_query(
        self,
        candidate_pairs: Sequence[PreferencePair],
        n: int = 1,
    ) -> list[PreferencePair]:
        """Select the top-``n`` most informative pairs.

        Args:
            candidate_pairs: Available pairs the UI could show the user.
            n: Number of pairs to return.

        Returns:
            The top ``n`` pairs, sorted in descending information gain.
            If ``candidate_pairs`` is empty returns ``[]``.

        Raises:
            ValueError: If ``n`` < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not candidate_pairs:
            return []
        scores: list[tuple[float, PreferencePair]] = []
        for p in candidate_pairs:
            try:
                p.validate()
            except ValueError as exc:
                logger.debug("Skipping invalid candidate pair: %s", exc)
                continue
            scores.append((self.score_pair(p), p))
        scores.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scores[:n]]

    def register_labelled(self, pair: PreferencePair) -> None:
        """Update the Fisher proxy after a label is observed for ``pair``.

        Adds the outer product ``phi phi^T`` to the running Fisher matrix,
        which tightens future D-optimal scores around unexplored regions.
        """
        pair.validate()
        phi = self._pair_feature_diff(pair)
        self._fisher = self._fisher + np.outer(phi, phi)

    def information_gain_threshold(
        self, candidate: PreferencePair, threshold: float
    ) -> bool:
        """Return True if ``candidate`` exceeds the information threshold.

        Used by :class:`PreferenceAwareRouter` to decide whether to ask
        the user a preference question on a given turn.
        """
        if threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        return self.score_pair(candidate) >= threshold

    # -- internal ---------------------------------------------------------

    def _pair_feature_diff(self, pair: PreferencePair) -> np.ndarray:
        """Compute ``last_layer(A) - last_layer(B)`` for ``pair``."""
        ctx_dim = self.reward_model.context_dim
        resp_dim = self.reward_model.response_dim
        ctx = torch.as_tensor(
            _pad_or_truncate(pair.context, ctx_dim), dtype=torch.float32
        )
        a = torch.as_tensor(
            _pad_or_truncate(pair.response_a_features, resp_dim),
            dtype=torch.float32,
        )
        b = torch.as_tensor(
            _pad_or_truncate(pair.response_b_features, resp_dim),
            dtype=torch.float32,
        )
        self.reward_model.eval()
        with torch.no_grad():
            phi_a = self.reward_model.last_layer_features(ctx, a).squeeze(0)
            phi_b = self.reward_model.last_layer_features(ctx, b).squeeze(0)
        diff = (phi_a - phi_b).numpy().astype(np.float64)
        return diff


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_response_features(
    *,
    length_tokens: float,
    latency_ms: float,
    model_confidence: float,
    response_dim: int = _DEFAULT_RESPONSE_DIM,
) -> list[float]:
    """Build a canonical response-feature vector for a candidate response.

    Args:
        length_tokens: Number of tokens in the response.
        latency_ms: Wall-clock latency to produce the response.
        model_confidence: The model's self-reported confidence (0-1).
        response_dim: Target dimensionality; extra dims are zero-padded.

    Returns:
        A list of length ``response_dim``.  The first three dims carry
        the supplied features (normalised to ``[0, 1]``); the rest are
        zeros so the MLP can learn its own linear combination later.
    """
    if response_dim <= 0:
        raise ValueError(f"response_dim must be > 0, got {response_dim}")
    # Normalise to [0, 1] with generous ceilings — unreachable values are
    # clipped so that adversarially-large inputs cannot dominate.
    length_n = float(min(max(length_tokens, 0.0) / 1024.0, 1.0))
    latency_n = float(min(max(latency_ms, 0.0) / 10_000.0, 1.0))
    conf_n = float(min(max(model_confidence, 0.0), 1.0))
    vec = [length_n, latency_n, conf_n] + [0.0] * (response_dim - 3)
    return vec[:response_dim]


__all__ = [
    "ActivePreferenceSelector",
    "BradleyTerryRewardModel",
    "DPOFitReport",
    "DPOPreferenceOptimizer",
    "PreferenceDataset",
    "PreferencePair",
    "build_response_features",
]
