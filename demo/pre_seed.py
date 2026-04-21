"""Pre-seed the demo environment with a realistic ``demo_user`` history.

This module is the source of truth for the interview-day "returning user"
illusion: twenty prior session summaries spanning the last seven days,
ten diary exchanges carrying TF-IDF topic keywords but **zero raw text**,
a mid-intensity behavioural baseline, and a non-flat bandit posterior
biased the way :mod:`docs.DEMO_SCRIPT` expects.

The seed populates three persisted surfaces:

1. :class:`~i3.user_model.store.UserModelStore` -- the long-term user
   profile (baseline embedding, feature means/stds, long-term style).
2. :class:`~i3.diary.store.DiaryStore` -- 20 session rows + 10 exchange
   rows, all metadata-only (no ``message`` column exists in the schema).
3. :class:`~i3.router.bandit.ContextualThompsonBandit` -- 30 synthesised
   (context, arm, reward) updates that favour ``local_slm`` on short
   queries and ``cloud_llm`` on complex ones.

Usage::

    python -m demo.pre_seed --user-id demo_user --config configs/default.yaml

Or, from the admin router::

    from demo.pre_seed import seed_pipeline
    await seed_pipeline(pipeline, user_id="demo_user")
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scripted "last seven days" of session summaries
# ---------------------------------------------------------------------------

# SEC: the emotion cycle is deterministic — neutral → curious → stressed →
# fatigued → recovered → ... — so the diary panel always tells the same
# arc on demo day, regardless of how many runs it takes to get it right.
_EMOTION_CYCLE: tuple[str, ...] = (
    "neutral",
    "curious",
    "stressed",
    "fatigued",
    "recovered",
)

_TOPIC_POOL: tuple[tuple[str, ...], ...] = (
    ("morning", "coffee", "routine"),
    ("tcn", "dilated", "convolution"),
    ("attention", "transformer", "context"),
    ("keystroke", "biometrics", "identity"),
    ("bandit", "thompson", "exploration"),
    ("fatigue", "break", "rest"),
    ("accessibility", "simple", "direct"),
    ("privacy", "fernet", "encryption"),
    ("edge", "kirin", "latency"),
    ("summary", "diary", "recall"),
)


# ---------------------------------------------------------------------------
# Synthetic baseline helpers
# ---------------------------------------------------------------------------


def _build_baseline_embedding(dim: int = 64, seed: int = 42) -> torch.Tensor:
    """Return a deterministic L2-normalised 64-dim baseline embedding.

    A fixed seed guarantees demo reproducibility across runs so the 2D
    projection of the baseline dot always lands in the same spot on the
    dashboard.

    Args:
        dim: Embedding dimensionality (matches ``encoder.embedding_dim``).
        seed: RNG seed; 42 by default.

    Returns:
        A float32 :class:`torch.Tensor` of shape ``(dim,)``, unit-norm.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm > 0:
        raw = raw / norm
    return torch.from_numpy(raw)


def _build_baseline_features() -> tuple[dict[str, float], dict[str, float]]:
    """Return baseline feature means and stds for a mid-intensity typist.

    The numbers mirror the ``demo_user`` persona in
    :data:`demo.profiles.DEMO_PROFILES`: 180ms median inter-key interval,
    6% correction rate, formality 0.55, valence +0.1.

    Returns:
        A ``(means, stds)`` tuple of dicts keyed by feature name.
    """
    means: dict[str, float] = {
        "mean_iki": 180.0,
        "std_iki": 80.0,
        "message_length": 42.0,
        "type_token_ratio": 0.65,
        "mean_word_length": 4.6,
        "flesch_kincaid": 8.2,
        "formality": 0.55,
        "composition_speed": 4.8,
        "backspace_ratio": 0.06,
        "engagement_velocity": 0.55,
        "sentiment_valence": 0.10,
    }
    stds: dict[str, float] = {
        "mean_iki": 40.0,
        "std_iki": 20.0,
        "message_length": 12.0,
        "type_token_ratio": 0.08,
        "mean_word_length": 0.6,
        "flesch_kincaid": 1.2,
        "formality": 0.10,
        "composition_speed": 0.9,
        "backspace_ratio": 0.02,
        "engagement_velocity": 0.15,
        "sentiment_valence": 0.20,
    }
    return means, stds


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _seed_user_profile(
    db_path: str,
    user_id: str,
    baseline_embedding: torch.Tensor,
    baseline_means: dict[str, float],
    baseline_stds: dict[str, float],
) -> None:
    """Upsert the demo user profile into the UserModelStore.

    Args:
        db_path: SQLite path for the user-model store.
        user_id: Seed target identifier.
        baseline_embedding: Pre-built L2-normalised baseline.
        baseline_means: Feature means from :func:`_build_baseline_features`.
        baseline_stds: Feature stds from :func:`_build_baseline_features`.
    """
    from i3.user_model.store import UserModelStore
    from i3.user_model.types import UserProfile

    now = datetime.now(timezone.utc)
    created = now - timedelta(days=7)
    profile = UserProfile(
        user_id=user_id,
        baseline_embedding=baseline_embedding,
        baseline_features_mean=baseline_means,
        baseline_features_std=baseline_stds,
        total_sessions=20,
        total_messages=110,
        relationship_strength=0.42,
        long_term_style={
            "formality": 0.55,
            "verbosity": 0.50,
            "emotionality": 0.55,
            "directness": 0.65,
        },
        created_at=created,
        updated_at=now,
        baseline_established=True,
    )
    async with UserModelStore(db_path) as store:
        await store.save_profile(profile)
    logger.info(
        "pre_seed.profile.saved",
        extra={"event": "pre_seed", "user_id": user_id, "db": db_path},
    )


async def _seed_diary(
    diary_store: Any,
    user_id: str,
    baseline_embedding: torch.Tensor,
    rng: random.Random,
) -> tuple[int, int]:
    """Insert 20 session summaries and 10 diary exchange rows.

    Sessions are spaced across the last seven days; emotions cycle through
    :data:`_EMOTION_CYCLE`. Exchanges store zero-mean jittered embeddings
    derived from the baseline so the 2D projection trails look plausibly
    clustered on the dashboard.

    Args:
        diary_store: Initialised :class:`~i3.diary.store.DiaryStore`.
        user_id: Seed target identifier.
        baseline_embedding: The same tensor used for the profile.
        rng: Deterministic RNG for topic and emotion jitter.

    Returns:
        ``(sessions_created, exchanges_created)``.
    """
    from i3.diary.store import encrypt_embedding_envelope

    sessions_created = 0
    exchanges_created = 0
    now = datetime.now(timezone.utc)

    for i in range(20):
        session_id = f"seed-{user_id}-{i:02d}"
        # SEC: start/end times remain monotonic and within the last 7 days.
        offset_hours = i * (7 * 24 / 20)
        start_dt = now - timedelta(hours=(7 * 24) - offset_hours)
        duration_min = rng.randint(3, 18)
        end_dt = start_dt + timedelta(minutes=duration_min)

        emotion = _EMOTION_CYCLE[i % len(_EMOTION_CYCLE)]
        topics = list(_TOPIC_POOL[i % len(_TOPIC_POOL)])
        mean_engagement = round(rng.uniform(0.45, 0.82), 3)
        mean_cognitive_load = round(rng.uniform(0.25, 0.70), 3)
        mean_accessibility = round(rng.uniform(0.05, 0.30), 3)
        relationship_strength = round(min(0.1 + i * 0.02, 0.9), 3)

        # Session row (requires create → end so summary columns populate)
        try:
            await diary_store.create_session(session_id, user_id)
        except Exception:  # noqa: BLE001 — tolerate idempotent re-seeds
            logger.debug("pre_seed.session.exists", extra={"session": session_id})
            continue

        # Adjust the created_at to the historical start time via raw UPDATE
        # so the diary panel orders entries correctly. We reuse end_session
        # for the aggregated columns.
        summary_text = (
            f"Session #{i + 1} — a {emotion} exchange covering "
            f"{', '.join(topics)}. No raw text persisted."
        )
        await diary_store.end_session(
            session_id=session_id,
            summary=summary_text,
            dominant_emotion=emotion,
            topics=topics,
            mean_engagement=mean_engagement,
            mean_cognitive_load=mean_cognitive_load,
            mean_accessibility=mean_accessibility,
            relationship_strength=relationship_strength,
        )
        sessions_created += 1

        # Backfill start_time so the session ordering matches the narrative.
        try:
            import aiosqlite  # noqa: PLC0415

            async with aiosqlite.connect(diary_store.db_path) as db:
                await db.execute(
                    "UPDATE sessions SET start_time = ?, end_time = ? "
                    "WHERE session_id = ?",
                    (start_dt.isoformat(), end_dt.isoformat(), session_id),
                )
                await db.commit()
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug(
                "pre_seed.session.timestamp_backfill_failed",
                extra={"err": type(exc).__name__},
            )

        # Seed 10 exchanges spread across the earliest 10 sessions.
        if i < 10:
            jitter = torch.randn_like(baseline_embedding) * 0.05
            emb = baseline_embedding + jitter
            envelope = encrypt_embedding_envelope(emb, encryptor=None)
            adaptation_vector = {
                "cognitive_load": mean_cognitive_load,
                "emotional_tone": 0.5 + rng.uniform(-0.15, 0.15),
                "accessibility": mean_accessibility,
                "style": {
                    "formality": round(rng.uniform(0.4, 0.7), 3),
                    "verbosity": round(rng.uniform(0.3, 0.7), 3),
                    "emotionality": round(rng.uniform(0.4, 0.7), 3),
                    "directness": round(rng.uniform(0.5, 0.8), 3),
                },
            }
            await diary_store.log_exchange(
                session_id=session_id,
                user_state_embedding=envelope,
                adaptation_vector=adaptation_vector,
                route_chosen="local_slm" if rng.random() < 0.6 else "cloud_llm",
                response_latency_ms=rng.randint(140, 720),
                engagement_signal=round(rng.uniform(0.45, 0.82), 3),
                topics=topics,
            )
            exchanges_created += 1

    logger.info(
        "pre_seed.diary.written",
        extra={
            "event": "pre_seed",
            "sessions": sessions_created,
            "exchanges": exchanges_created,
        },
    )
    return sessions_created, exchanges_created


def _seed_bandit(bandit: Any, rng: random.Random) -> int:
    """Push 30 scripted updates into the bandit posterior.

    The rule: short queries (low ``query_complexity``) reward
    ``local_slm`` with ~0.78 engagement; complex queries reward
    ``cloud_llm`` with ~0.70; each gets Gaussian jitter so the Laplace
    refit produces a non-degenerate covariance.

    Args:
        bandit: The :class:`~i3.router.bandit.ContextualThompsonBandit`.
        rng: Deterministic RNG for the reward jitter.

    Returns:
        The number of updates applied (always 30 unless the bandit
        rejects a context).
    """
    dim = int(getattr(bandit, "context_dim", 12))
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    updates = 0
    for _ in range(30):
        complexity = float(rng.uniform(0.05, 0.95))
        ctx = np_rng.standard_normal(dim).astype(np.float64) * 0.3
        # Slot 1 is query_complexity in the RoutingContext layout.
        if dim >= 2:
            ctx[1] = complexity
        if complexity < 0.5:
            arm = 0  # local_slm
            reward = float(np.clip(rng.gauss(0.78, 0.08), 0.0, 1.0))
        else:
            arm = 1  # cloud_llm
            reward = float(np.clip(rng.gauss(0.70, 0.10), 0.0, 1.0))
        try:
            bandit.update(arm, ctx, reward)
            updates += 1
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "pre_seed.bandit.update_failed",
                extra={"err": type(exc).__name__},
            )
    logger.info(
        "pre_seed.bandit.done",
        extra={"event": "pre_seed", "updates": updates},
    )
    return updates


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def seed_pipeline(
    pipeline: Any, user_id: str = "demo_user", *, seed: int = 42
) -> dict[str, Any]:
    """Seed an initialised pipeline in-place.

    Intended for use from the admin router and the CLI entry point.
    Reads the user-model DB path from ``pipeline.config``, falls back to
    ``data/user_model.db`` otherwise.

    Args:
        pipeline: An initialised :class:`~i3.pipeline.engine.Pipeline`.
        user_id: Target user id; defaults to ``"demo_user"``.
        seed: Deterministic RNG seed.

    Returns:
        A dict summarising the artefacts created.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    baseline_embedding = _build_baseline_embedding(
        dim=int(getattr(pipeline.config.encoder, "embedding_dim", 64)),
        seed=seed,
    )
    means, stds = _build_baseline_features()

    db_path = str(
        getattr(pipeline.config.user_model, "db_path", "data/user_model.db")
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    await _seed_user_profile(
        db_path=db_path,
        user_id=user_id,
        baseline_embedding=baseline_embedding,
        baseline_means=means,
        baseline_stds=stds,
    )

    sessions, exchanges = await _seed_diary(
        diary_store=pipeline.diary_store,
        user_id=user_id,
        baseline_embedding=baseline_embedding,
        rng=rng,
    )
    bandit_updates = _seed_bandit(pipeline.router, rng=rng)

    return {
        "user_id": user_id,
        "diary_sessions": sessions,
        "diary_entries": exchanges,
        "bandit_updates": bandit_updates,
    }


async def main() -> None:
    """Entry point for ``python -m demo.pre_seed``.

    Bootstraps a :class:`~i3.pipeline.engine.Pipeline` from the supplied
    config, seeds it, then shuts it down cleanly. Command-line arguments::

        --user-id   Identifier to seed (default ``demo_user``).
        --config    Path to the YAML config (default ``configs/default.yaml``).
        --seed      RNG seed (default ``42``).
    """
    parser = argparse.ArgumentParser(
        description="Pre-seed the I3 demo environment."
    )
    parser.add_argument("--user-id", default="demo_user", type=str)
    parser.add_argument(
        "--config", default="configs/default.yaml", type=str
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("I3_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from i3.config import load_config  # noqa: PLC0415
    from i3.pipeline.engine import Pipeline  # noqa: PLC0415

    config = load_config(args.config)
    pipeline = Pipeline(config)
    await pipeline.initialize()
    try:
        result = await seed_pipeline(pipeline, user_id=args.user_id, seed=args.seed)
        logger.info("pre_seed.complete", extra={"event": "pre_seed", **result})
        # SEC: print the result so CI / shell scripts can consume it without
        # depending on the logging config.
        print(result)  # noqa: T201
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
