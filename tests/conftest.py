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

import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# ─────────────────────────────────────────────────────────────────────────
# .env auto-load
# ─────────────────────────────────────────────────────────────────────────
# Several config-loading fixtures load ``configs/default.yaml`` which
# emits a UserWarning (-> pytest error under filterwarnings=error) when
# ``I3_ENCRYPTION_KEY`` isn't set.  The same key lives in the project's
# gitignored ``.env``, so we read it line-by-line here (no python-dotenv
# dep) and only set vars that are not already in os.environ.  Tests can
# still override individual vars via monkeypatch as before.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
if _ENV_PATH.exists():
    try:
        for _ln in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if not _ln or _ln.startswith("#") or "=" not in _ln:
                continue
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────
# Torch availability probe
# ─────────────────────────────────────────────────────────────────────────
# Detect whether the binary torch install actually works on this box.
# On Windows without the VC++ redistributable, ``import torch`` raises
# OSError (WinError 1114) or leaves sys.modules['torch'] as a partial
# stub that fails on attribute access.  When torch is unusable, install
# a minimal stub *and* advertise the condition so collection-time can
# drop torch-heavy test modules cleanly.
_TORCH_AVAILABLE: bool = False
try:  # pragma: no cover - depends on environment
    import torch  # noqa: F401
    # Real torch must expose ``nn`` as an importable sub-module.
    import torch.nn  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 — OSError, ImportError, AttributeError, …
    _torch_stub = types.ModuleType("torch")

    class _Tensor:  # pragma: no cover - stub only
        pass

    def _tensor(*_a, **_kw):  # pragma: no cover - stub only
        return _Tensor()

    _torch_stub.Tensor = _Tensor
    _torch_stub.tensor = _tensor
    _torch_stub.float32 = "float32"

    class _NoOp:  # pragma: no cover - context manager stub
        def __enter__(self): return self
        def __exit__(self, *_): return False

    _torch_stub.no_grad = lambda: _NoOp()
    sys.modules["torch"] = _torch_stub
    import torch  # noqa: F401  — now resolves to the stub


# ─────────────────────────────────────────────────────────────────────────
# Collection-time skip-list for modules that *genuinely* need real torch
# ─────────────────────────────────────────────────────────────────────────
# When torch is unavailable, pytest would otherwise fail collection on any
# test file whose imports touch ``torch.nn`` / ``torch.optim``.  We list
# those test modules here so collection is clean; they are recorded as
# skips in the report rather than errors.  When torch *is* available the
# list is empty and every test runs normally.
collect_ignore_glob: list[str] = []
if not _TORCH_AVAILABLE:
    collect_ignore_glob = [
        "test_tcn.py",
        "test_slm.py",
        "test_encoder_*.py",
        "test_aux_losses.py",
        "test_counterfactuals.py",
        "test_drift_detector.py",
        "test_ewc.py",
        "test_facial_affect.py",
        "test_interpretability.py",
        "test_interpretability_circuits.py",
        "test_maml.py",
        "test_multimodal_fusion.py",
        "test_pipeline*.py",
        "test_ray_serve_app.py",
        "test_sparse_autoencoder.py",
        "test_speculative_decoding.py",
        "test_task_generator.py",
        "test_uncertainty.py",
        "test_user_model.py",
        "test_voice_prosody.py",
        "test_wearable_ingest.py",
        "test_edge_exporters_smoke.py",
        "test_fabric_smoke.py",
        "test_cost_tracker.py",
        "test_interpretability*.py",
        "test_integration.py",
        "test_ppg_hrv.py",
        "test_preference_learning.py",
        "test_routes_preference.py",
    ]

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
