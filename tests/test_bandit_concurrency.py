"""Concurrency tests for :class:`i3.router.bandit.ContextualThompsonBandit`.

The bandit is a single shared instance that every in-flight request
touches — ``select_arm`` and ``update`` race on the posterior state
unless serialised.  The 2026-04-23 audit added a re-entrant lock and
converted the per-arm history to a bounded deque; these tests verify
both behaviours under hostile concurrency.
"""

from __future__ import annotations

import sys
import threading
import types
from collections import deque

import pytest


# Stub torch so this test file imports in environments with a broken
# torch install (Windows DLL issue, minimal CI).
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.tensor = lambda *a, **k: _torch_stub.Tensor()
_torch_stub.float32 = "float32"
sys.modules.setdefault("torch", _torch_stub)


import numpy as np  # noqa: E402

from i3.router.bandit import ContextualThompsonBandit  # noqa: E402


# ---------------------------------------------------------------------------
# History bound — deque(maxlen) keeps overflow O(1)
# ---------------------------------------------------------------------------


def test_history_is_bounded_deque():
    bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
    for arm_history in bandit.history:
        assert isinstance(arm_history, deque)
        assert arm_history.maxlen is not None and arm_history.maxlen >= 1_000


def test_history_overflow_stays_capped():
    """Push far more observations than maxlen and confirm the size caps."""
    bandit = ContextualThompsonBandit(
        n_arms=2, context_dim=4, refit_interval=10_000
    )
    cap = bandit.history[0].maxlen
    assert cap is not None
    for _ in range(cap + 500):
        bandit.update(0, np.zeros(4), reward=0.5)
    assert len(bandit.history[0]) == cap


# ---------------------------------------------------------------------------
# Lock-protected mutation — 8 threads x 200 ops = 1600 total, no loss
# ---------------------------------------------------------------------------


def test_concurrent_updates_do_not_lose_pulls():
    bandit = ContextualThompsonBandit(
        n_arms=2, context_dim=4, refit_interval=50
    )
    ops_per_thread = 200
    n_threads = 8

    def hammer() -> None:
        rng = np.random.default_rng()
        for _ in range(ops_per_thread):
            ctx = rng.standard_normal(4)
            arm, _conf = bandit.select_arm(ctx)
            bandit.update(arm, ctx, reward=0.5)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_pulls = sum(bandit.total_pulls)
    assert total_pulls == n_threads * ops_per_thread, (
        f"lost updates: total_pulls={total_pulls} "
        f"expected={n_threads * ops_per_thread}"
    )


def test_concurrent_select_does_not_crash_mid_refit():
    """Interleave select_arm calls while updates are re-fitting the posterior."""
    bandit = ContextualThompsonBandit(
        n_arms=2, context_dim=4, refit_interval=5
    )
    errors: list[BaseException] = []

    def writer() -> None:
        rng = np.random.default_rng(42)
        try:
            for _ in range(100):
                bandit.update(0, rng.standard_normal(4), reward=1.0)
                bandit.update(1, rng.standard_normal(4), reward=0.0)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader() -> None:
        rng = np.random.default_rng(43)
        try:
            for _ in range(500):
                arm, _ = bandit.select_arm(rng.standard_normal(4))
                assert 0 <= arm < 2
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = (
        [threading.Thread(target=writer) for _ in range(2)]
        + [threading.Thread(target=reader) for _ in range(4)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent errors: {errors}"


def test_update_rejects_out_of_range_arm():
    bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
    with pytest.raises(ValueError):
        bandit.update(5, np.zeros(4), reward=0.5)


def test_update_sanitizes_nan_context():
    """Non-finite contexts are zeroed defensively, not crashed on."""
    bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
    bad = np.array([1.0, np.nan, 2.0, np.inf])
    bandit.update(0, bad, reward=0.5)  # must not raise
    # Resulting history entry should have been sanitized.
    ctx_stored, _ = bandit.history[0][-1]
    assert np.all(np.isfinite(ctx_stored))


def test_update_clips_out_of_range_reward():
    bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
    bandit.update(0, np.zeros(4), reward=5.0)  # > 1.0
    bandit.update(1, np.zeros(4), reward=-3.0)  # < 0
    # Both are clipped to [0, 1] before hitting the Beta posterior.
    assert bandit.alpha[0] <= 2.0 + 1e-9
    assert bandit.beta_param[1] <= 2.0 + 1e-9
