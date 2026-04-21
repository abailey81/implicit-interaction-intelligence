"""Tests for :class:`i3.analytics.lance_vector.LanceUserEmbeddingStore`.

The tests exercise upsert, search, adaptation-cluster lookup, index
creation, and a recall sanity check.  They are skipped on install
footprints that lack either ``lancedb`` or ``pyarrow``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

lancedb = pytest.importorskip("lancedb", reason="lancedb not installed")
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")
np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path):
    """Fresh Lance store in a temp directory."""
    from i3.analytics.lance_vector import LanceUserEmbeddingStore

    return LanceUserEmbeddingStore(tmp_path / "lance_db", embedding_dim=16)


def _rand_vec(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        v[0] = 1.0
        n = 1.0
    return v / n


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_upsert_and_count(store) -> None:
    rng = np.random.default_rng(0)
    ts = datetime.now(timezone.utc)
    for i in range(20):
        store.upsert(
            user_id=f"u{i}",
            session_id=f"s{i}",
            ts=ts + timedelta(minutes=i),
            embedding=_rand_vec(rng),
            adaptation={"verbosity": float(i) / 20.0},
        )
    assert store.count() == 20


def test_upsert_replaces(store) -> None:
    rng = np.random.default_rng(1)
    ts = datetime.now(timezone.utc)
    store.upsert("u0", "s0", ts, _rand_vec(rng), {"a": 1.0})
    store.upsert("u0", "s0", ts, _rand_vec(rng), {"a": 2.0})  # upsert
    assert store.count() == 1


def test_upsert_rejects_wrong_dim(store) -> None:
    ts = datetime.now(timezone.utc)
    bad = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError):
        store.upsert("u", "s", ts, bad)


def test_search_returns_nearest(store) -> None:
    rng = np.random.default_rng(2)
    ts = datetime.now(timezone.utc)
    target_vec = _rand_vec(rng)
    store.upsert("target", "s", ts, target_vec, {"k": 1.0})
    for i in range(30):
        store.upsert(f"n{i}", f"s{i}", ts, _rand_vec(rng))
    hits = store.search_similar(target_vec, k=3)
    assert hits
    assert hits[0].user_id == "target"
    # cosine score is 1 - distance; perfect match -> ~1.0.
    assert hits[0].score > 0.999


def test_search_exclude_user(store) -> None:
    rng = np.random.default_rng(3)
    ts = datetime.now(timezone.utc)
    q = _rand_vec(rng)
    store.upsert("me", "s", ts, q)
    for i in range(10):
        store.upsert(f"o{i}", f"s{i}", ts, _rand_vec(rng))
    hits = store.search_similar(q, k=5, exclude_user_id="me")
    assert all(h.user_id != "me" for h in hits)


def test_search_by_adaptation_cluster(store) -> None:
    rng = np.random.default_rng(4)
    ts = datetime.now(timezone.utc)
    for i in range(10):
        store.upsert(
            user_id=f"u{i}",
            session_id=f"s{i}",
            ts=ts + timedelta(seconds=i),
            embedding=_rand_vec(rng),
            adaptation={"verbosity": 0.1 * i, "warmth": 1.0 - 0.1 * i},
        )
    hits = store.search_by_adaptation_cluster(
        {"verbosity": 0.0, "warmth": 1.0}, k=3
    )
    assert hits
    # Closest user to (0, 1) is u0.
    assert hits[0].user_id == "u0"


def test_create_index(store) -> None:
    rng = np.random.default_rng(5)
    ts = datetime.now(timezone.utc)
    # Need enough rows for a non-trivial IVF build.
    for i in range(300):
        store.upsert(f"u{i}", f"s{i}", ts, _rand_vec(rng))
    # embedding_dim=16, num_sub_vectors must divide it (8 -> 2 bytes each).
    store.create_index(num_partitions=4, num_sub_vectors=8)


def test_create_index_rejects_bad_params(store) -> None:
    rng = np.random.default_rng(6)
    ts = datetime.now(timezone.utc)
    store.upsert("u", "s", ts, _rand_vec(rng))
    with pytest.raises(ValueError):
        # 16 not divisible by 5.
        store.create_index(num_partitions=4, num_sub_vectors=5)


def test_compact_idempotent(store) -> None:
    """compact() must not raise when the table is empty."""
    store.compact()
    rng = np.random.default_rng(7)
    ts = datetime.now(timezone.utc)
    store.upsert("u", "s", ts, _rand_vec(rng))
    store.compact()
    assert store.count() == 1


def test_recall_sanity(store) -> None:
    """Approximate recall: exact top-1 must be returned for identical queries."""
    rng = np.random.default_rng(11)
    ts = datetime.now(timezone.utc)
    vecs = [_rand_vec(rng) for _ in range(100)]
    for i, v in enumerate(vecs):
        store.upsert(f"u{i}", f"s{i}", ts, v)
    # Flat search (no index) should always return the exact match first.
    correct = 0
    for i in range(0, 100, 10):
        hits = store.search_similar(vecs[i], k=1)
        if hits and hits[0].user_id == f"u{i}":
            correct += 1
    # Expect near-perfect recall on flat search.
    assert correct >= 9
