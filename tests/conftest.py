"""Shared pytest fixtures for the I³ test suite.

These fixtures provide reproducible, isolated test resources:
  - CPU-only torch device (for determinism across hardware)
  - Minimal config loaded from configs/default.yaml
  - Temporary SQLite paths (via tmp_path)
  - Sample interaction + adaptation vectors
  - Seeded RNGs across torch / numpy / random
  - Initialized async DiaryStore
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────
#  Device & Config
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Torch device for tests (CPU to ensure reproducibility)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_config():
    """Minimal config loaded from configs/default.yaml with seeds set."""
    from i3.config import load_config

    return load_config("configs/default.yaml", set_seeds=True)


# ─────────────────────────────────────────────────────────────────────────
#  Filesystem
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Temporary SQLite database path (auto-cleaned by pytest)."""
    return str(tmp_path / "test.db")


# ─────────────────────────────────────────────────────────────────────────
#  Sample Data
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_interaction_features():
    """Generate 10 sample InteractionFeatureVectors for testing."""
    from i3.interaction.types import InteractionFeatureVector

    return [InteractionFeatureVector.zeros() for _ in range(10)]


@pytest.fixture
def sample_adaptation_vector():
    """Sample AdaptationVector with non-default values."""
    from i3.adaptation.types import AdaptationVector, StyleVector

    return AdaptationVector(
        cognitive_load=0.6,
        style_mirror=StyleVector(0.3, 0.5, 0.7, 0.6),
        emotional_tone=0.4,
        accessibility=0.2,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_rng():
    """Seed all RNGs (torch, numpy, random) for reproducibility."""
    import random

    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    yield
    # No teardown needed — next fixture invocation will re-seed.


# ─────────────────────────────────────────────────────────────────────────
#  Async Resources
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def temp_diary_store(temp_db_path: str):
    """Initialized DiaryStore for async tests."""
    from i3.diary.store import DiaryStore

    store = DiaryStore(temp_db_path)
    await store.initialize()
    yield store
    # DiaryStore is backed by a tmp_path SQLite file; pytest handles cleanup.
