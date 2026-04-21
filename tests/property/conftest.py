"""Shared pytest/Hypothesis fixtures for the property-based test suite.

The fixtures here ensure that every property test runs under a fixed seed
so that flaky failures are reproducible from CI logs, and that the global
Hypothesis configuration is tuned for numerical tests (no deadlines,
moderate example counts, no shrink budgets that exceed CI time).
"""

from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np
import pytest

# Hypothesis is declared as a dev dependency; skip cleanly when absent so
# this package doesn't break the entire suite on a minimal install.
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, Phase, seed as hyp_seed, settings


# ─────────────────────────────────────────────────────────────────────────
#  Global Hypothesis profiles
# ─────────────────────────────────────────────────────────────────────────

settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)

settings.register_profile(
    "ci",
    parent=settings.get_profile("default"),
    max_examples=200,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
)

settings.register_profile(
    "dev",
    parent=settings.get_profile("default"),
    max_examples=50,
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


# ─────────────────────────────────────────────────────────────────────────
#  Deterministic seeds
# ─────────────────────────────────────────────────────────────────────────

# Pinning the Hypothesis seed at collection time makes the search path
# deterministic across local and CI runs.  This is explicitly encouraged by
# the Hypothesis docs for numerical / ML property tests.
hyp_seed(42)


@pytest.fixture(autouse=True)
def _seed_all_rngs() -> Iterator[None]:
    """Seed Python ``random``, numpy, and torch before every test.

    ``autouse=True`` so each property test inside this package starts from
    identical RNG state, independent of the strategy shrinker's history.
    """
    random.seed(42)
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:  # pragma: no cover — torch is a hard dep of I3
        pass
    yield


# ─────────────────────────────────────────────────────────────────────────
#  Shared lightweight fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def hypothesis_settings() -> settings:
    """Expose the active Hypothesis ``settings`` to tests that want to
    wrap an individual ``@given`` with tighter limits (e.g. very slow
    tensor tests)."""
    return settings()
