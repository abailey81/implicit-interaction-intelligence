"""Shared fixtures for the I3 benchmark suite.

Key design decisions:

* Every benchmark tolerates missing checkpoints: models are constructed
  with random-init weights so the benchmarks are reproducible from a
  clean checkout.  Setting ``I3_BENCH_REQUIRE_CKPT=1`` flips this into
  a strict mode that skips benchmarks whose checkpoints are absent.
* Every ``benchmark()`` call uses 3 warmup rounds and 20 measured
  rounds, timed with ``time.perf_counter`` and compared with the
  ``min`` strategy (matches the SLO assumptions in ``benchmarks/slos.yaml``).
* Fixtures are ``session``-scoped where possible to amortise model
  construction across many benchmarks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENCODER_CKPT = PROJECT_ROOT / "checkpoints" / "encoder" / "best.pt"
SLM_CKPT = PROJECT_ROOT / "checkpoints" / "slm" / "best.pt"


def _require_ckpt() -> bool:
    """Return ``True`` when benchmarks must skip on missing checkpoints."""
    return os.environ.get("I3_BENCH_REQUIRE_CKPT", "") == "1"


# --------------------------------------------------------------------------- #
# pytest-benchmark defaults
# --------------------------------------------------------------------------- #


@pytest.fixture
def benchmark_opts() -> dict[str, Any]:
    """Default options threaded into every ``benchmark.pedantic`` call."""
    return {
        "iterations": 1,
        "rounds": 20,
        "warmup_rounds": 3,
    }


def pytest_configure(config: pytest.Config) -> None:
    """Lock in sensible pytest-benchmark defaults if they are not overridden.

    Args:
        config: The active pytest configuration object.
    """
    # These options are interpreted by the pytest-benchmark plugin. We
    # set them defensively with setattr so that a missing plugin (in a
    # minimal install) does not break test collection.
    for name, value in (
        ("benchmark_timer", "time.perf_counter"),
        ("benchmark_warmup", "on"),
        ("benchmark_warmup_iterations", 3),
        ("benchmark_min_rounds", 20),
        ("benchmark_columns", ("min", "median", "mean", "stddev", "ops", "rounds")),
        ("benchmark_sort", "min"),
    ):
        try:
            setattr(config.option, name, value)
        except Exception:  # noqa: BLE001 - non-fatal if plugin missing
            pass


# --------------------------------------------------------------------------- #
# Model fixtures
# --------------------------------------------------------------------------- #


def _maybe_load_state(model: Any, path: Path) -> None:
    """Load ``path`` into ``model`` if it exists; honour ``I3_BENCH_REQUIRE_CKPT``.

    Args:
        model: The target ``nn.Module``.
        path: Candidate checkpoint path.
    """
    if not path.exists():
        if _require_ckpt():
            pytest.skip(f"checkpoint missing: {path} (I3_BENCH_REQUIRE_CKPT=1)")
        return
    import torch  # type: ignore[import-not-found]

    try:
        from i3.mlops.checkpoint import load_verified

        state = load_verified(path, weights_only=True)
    except Exception:  # noqa: BLE001
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


@pytest.fixture(scope="session")
def encoder() -> Any:
    """Session-scoped TCN encoder."""
    torch = pytest.importorskip("torch")
    from i3.encoder.tcn import TemporalConvNet

    model = TemporalConvNet().eval()
    _maybe_load_state(model, ENCODER_CKPT)
    with torch.inference_mode():
        model(torch.randn(1, 10, getattr(model, "input_dim", 32)))  # warm caches
    return model


@pytest.fixture(scope="session")
def slm() -> Any:
    """Session-scoped Adaptive SLM."""
    torch = pytest.importorskip("torch")
    from i3.slm.model import AdaptiveSLM

    model = AdaptiveSLM().eval()
    _maybe_load_state(model, SLM_CKPT)
    with torch.inference_mode():
        model(torch.randint(0, max(model.vocab_size, 1), (1, 8), dtype=torch.long))
    return model


@pytest.fixture(scope="session")
def router() -> Any:
    """Session-scoped IntelligentRouter."""
    from i3.config import Config
    from i3.router.router import IntelligentRouter

    cfg = Config()
    return IntelligentRouter(cfg)


@pytest.fixture(scope="session")
def sanitizer() -> Any:
    """Session-scoped PrivacySanitizer."""
    from i3.privacy.sanitizer import PrivacySanitizer

    return PrivacySanitizer(enabled=True)


# --------------------------------------------------------------------------- #
# Input fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def encoder_input() -> Any:
    """Fresh random encoder input tensor."""
    torch = pytest.importorskip("torch")
    return torch.randn(1, 10, 32)


@pytest.fixture
def encoder_batch_input() -> Any:
    """Fresh random encoder batch input tensor ``[16, 10, 32]``."""
    torch = pytest.importorskip("torch")
    return torch.randn(16, 10, 32)


@pytest.fixture
def slm_prefill_input(slm: Any) -> Any:
    """Prefill-stage input ids for the SLM."""
    torch = pytest.importorskip("torch")
    return torch.randint(0, max(slm.vocab_size, 1), (1, 16), dtype=torch.long)


@pytest.fixture
def slm_decode_input(slm: Any) -> Any:
    """Single-token decode input for the SLM."""
    torch = pytest.importorskip("torch")
    return torch.randint(0, max(slm.vocab_size, 1), (1, 1), dtype=torch.long)


@pytest.fixture
def routing_context() -> Any:
    """A default :class:`RoutingContext` for bandit benchmarks."""
    from i3.router.types import RoutingContext

    return RoutingContext(
        user_state_compressed=[0.1, -0.2, 0.3, 0.0],
        query_complexity=0.6,
        topic_sensitivity=0.1,
        user_patience=0.7,
        session_progress=0.3,
        baseline_established=True,
        previous_route=0,
        previous_engagement=0.8,
        time_of_day=0.5,
        message_count=5,
        cloud_latency_est=0.3,
        slm_confidence=0.6,
    )


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    """Canned realistic input texts used across benchmarks."""
    return [
        "Hi there, can you help me plan my week?",
        "Explain the transformer architecture in simple terms.",
        "My email is alice@example.com and phone 555-123-4567.",
        "I'm feeling anxious about the upcoming presentation.",
        "What's 23 * 47? Also, remind me to buy milk tomorrow.",
    ] * 4  # 20 samples total


# --------------------------------------------------------------------------- #
# Timing helpers re-exported for ad-hoc benchmarks
# --------------------------------------------------------------------------- #


@pytest.fixture
def perf_timer() -> Iterator[Any]:
    """Minimal ``time.perf_counter``-based context timer."""

    class _Timer:
        def __init__(self) -> None:
            self.elapsed_ms: float = 0.0

        def __enter__(self) -> "_Timer":
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_exc: Any) -> None:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    yield _Timer
