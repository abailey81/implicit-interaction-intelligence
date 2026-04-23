#!/usr/bin/env python
"""CLI: find top-k users with similar long-term embeddings via LanceDB.

This script demonstrates the :class:`LanceUserEmbeddingStore` flow:

1. Pull the query user's long-term embedding from a companion Lance
   store (previously populated by the pipeline or by the
   ``--seed-from-user-model`` utility below).
2. Run an IVF-PQ nearest-neighbour search with cosine distance.
3. Print a table of the top-k similar users (excluding the query user
   itself).

Usage::

    python scripts/find_similar_users.py \\
        --lance-uri data/lance_user_embeddings \\
        --user-id alice --k 10

When the Lance store is empty, pass ``--seed-from-user-model`` to
populate it from the existing ``UserProfileStore`` SQLite database
(reads ``baseline_embedding`` per user).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from i3.analytics.lance_vector import LanceUserEmbeddingStore  # noqa: E402

logger = logging.getLogger("i3.analytics.similar")


async def _seed_from_user_model(
    store: LanceUserEmbeddingStore, user_db_path: Path
) -> int:
    """Copy baseline embeddings from the user-model SQLite into the Lance store."""
    import numpy as np
    from i3.user_model.store import UserProfileStore

    ums = UserProfileStore(str(user_db_path))
    await ums.initialize()
    profiles = await ums.list_profiles()
    now = datetime.now(timezone.utc)
    n = 0
    for prof in profiles:
        emb = getattr(prof, "baseline_embedding", None)
        if emb is None:
            continue
        # Accept torch tensors, numpy arrays, or lists.
        try:
            import torch  # local import to avoid hard dep

            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
        except ImportError:  # pragma: no cover
            pass
        arr = np.asarray(emb, dtype=np.float32).ravel()
        if arr.shape[0] != store.embedding_dim:
            logger.warning(
                "Skipping %s: embedding dim %d != store dim %d",
                prof.user_id,
                arr.shape[0],
                store.embedding_dim,
            )
            continue
        store.upsert(
            user_id=prof.user_id,
            session_id="__baseline__",
            ts=now,
            embedding=arr,
            adaptation={},
        )
        n += 1
    logger.info("Seeded %d user baseline embeddings into LanceDB", n)
    return n


def _find(args: argparse.Namespace) -> int:
    import numpy as np

    store = LanceUserEmbeddingStore(args.lance_uri, embedding_dim=args.embedding_dim)

    if args.seed_from_user_model:
        asyncio.run(_seed_from_user_model(store, args.user_db))

    # Try IVF-PQ index if enough rows present; fall back to flat otherwise.
    n_rows = store.count()
    if n_rows >= max(256, args.k * 4):
        try:
            store.create_index(
                num_partitions=min(256, max(1, n_rows // 4)),
                num_sub_vectors=args.num_sub_vectors,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Index creation skipped: %s", exc)

    # Fetch the query user's embedding from the store.
    table = store._get_or_create_table()  # noqa: SLF001 - internal demo access
    df = table.to_pandas()
    hit = df[df["user_id"] == args.user_id]
    if hit.empty:
        print(f"[error] user_id '{args.user_id}' not found in Lance store.")
        return 2
    query_vec = np.asarray(hit.iloc[0]["embedding"], dtype=np.float32)

    results = store.search_similar(
        query_vec, k=args.k + 1, exclude_user_id=args.user_id
    )[: args.k]

    print(f"\nTop-{args.k} users similar to '{args.user_id}'")
    print("-" * 60)
    print(f"{'user_id':<20} {'score':>8} {'distance':>10} {'session_id':<20}")
    for r in results:
        print(
            f"{r.user_id:<20} {r.score:>8.4f} {r.distance:>10.4f} {r.session_id:<20}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--lance-uri",
        type=Path,
        default=Path("data/lance_user_embeddings"),
        help="LanceDB directory (default: data/lance_user_embeddings).",
    )
    p.add_argument("--user-id", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--num-sub-vectors", type=int, default=16)
    p.add_argument(
        "--seed-from-user-model",
        action="store_true",
        help="Populate Lance store from the user_model SQLite before searching.",
    )
    p.add_argument(
        "--user-db",
        type=Path,
        default=Path("data/users.db"),
        help="Path to user_model SQLite (only used with --seed-from-user-model).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _find(args)


if __name__ == "__main__":
    raise SystemExit(main())
