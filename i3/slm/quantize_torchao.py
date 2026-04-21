"""torchao-based INT4 / INT8 quantization for the Adaptive SLM.

``torchao`` is the 2026-era native PyTorch quantization library
(https://github.com/pytorch/ao) and supersedes the older
``torch.quantization`` APIs for modern transformer workloads. It
provides weight-only quantization at INT4 and INT8 precisions with
competitive latency on CPU, CUDA, and Apple Silicon.

This module is deliberately **parallel** to the existing
``i3.slm.quantize`` (which uses ``torch.ao.quantization.quantize_dynamic``)
— both paths are preserved so consumers can choose the backend that
best fits their deployment target.

Soft-imports ``torchao``. If the package is missing, every function in
this module raises :class:`RuntimeError` with a clear install command.
Importing this module itself is always safe.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.quantize_torchao import (
        quantize_int4_weight_only,
        quantize_int8_dynamic,
        benchmark_quantization,
    )

    model = AdaptiveSLM().eval()
    int8_model = quantize_int8_dynamic(model)
    int4_model = quantize_int4_weight_only(model)

    report = benchmark_quantization(
        model, int8_model, int4_model,
        example_inputs=(torch.randint(0, 8000, (1, 64)),),
        n_iters=50,
    )
    print(report)
"""

from __future__ import annotations

import copy
import logging
import time
from types import ModuleType
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import torchao as _torchao_module  # type: ignore[import-not-found]
    from torchao.quantization import quantize_ as _torchao_quantize  # type: ignore[import-not-found]

    _TORCHAO_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    _torchao_module = None  # type: ignore[assignment]
    _torchao_quantize = None  # type: ignore[assignment]
    _TORCHAO_AVAILABLE = False
    logger.info(
        "torchao not installed; INT4 / INT8 quantization via torchao "
        "will be unavailable. Install with: pip install torchao>=0.11"
    )


_INSTALL_HINT: str = (
    "torchao is required for this operation. "
    "Install it with: pip install 'torchao>=0.11'"
)


def _require_torchao() -> ModuleType:
    """Raise :class:`RuntimeError` if torchao is not available.

    Returns:
        The imported ``torchao`` module.

    Raises:
        RuntimeError: If ``torchao`` could not be imported.
    """
    if not _TORCHAO_AVAILABLE or _torchao_module is None:
        raise RuntimeError(_INSTALL_HINT)
    return _torchao_module


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------


def quantize_int4_weight_only(
    model: nn.Module,
    device: str = "cpu",
    *,
    group_size: int = 32,
) -> nn.Module:
    """Apply INT4 weight-only quantization to a model in-place-on-copy.

    Uses ``torchao.quantization.quantize_`` with an
    ``Int4WeightOnlyConfig``. The returned model is a deep copy of the
    input with every eligible ``nn.Linear`` weight tensor replaced by
    an INT4 representation (roughly 8x smaller than FP32).

    Args:
        model: The FP32 model to quantize. Must be in ``eval`` mode.
        device: Device on which to run the quantization pass. INT4
            kernels are most mature on ``"cpu"`` and ``"cuda"``.
        group_size: Group size for the weight-only quantizer. Smaller
            groups preserve more accuracy at a modest size cost.

    Returns:
        A new quantized ``nn.Module``. The input model is not mutated.

    Raises:
        RuntimeError: If ``torchao`` is not installed.
    """
    _require_torchao()
    # Import the config class lazily so that module import is cheap.
    try:
        from torchao.quantization import Int4WeightOnlyConfig  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only on older torchao
        raise RuntimeError(
            "torchao is installed but 'Int4WeightOnlyConfig' is missing. "
            "Please upgrade: pip install -U 'torchao>=0.11'"
        ) from exc

    target = copy.deepcopy(model).eval().to(device)
    config = Int4WeightOnlyConfig(group_size=group_size)
    assert _torchao_quantize is not None  # narrowing for type-checkers
    _torchao_quantize(target, config)
    logger.info(
        "Applied torchao INT4 weight-only quantization (group_size=%d) on %s",
        group_size,
        device,
    )
    return target


def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply INT8 dynamic quantization via torchao.

    This is the torchao parallel to the existing ``torch.quantization``
    flow in :mod:`i3.slm.quantize`. Use the torchao variant when you
    want a single quantization stack across the whole deployment.

    Args:
        model: The FP32 model to quantize. Must be in ``eval`` mode.

    Returns:
        A new quantized ``nn.Module``. The input model is not mutated.

    Raises:
        RuntimeError: If ``torchao`` is not installed.
    """
    _require_torchao()
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torchao is installed but the INT8 dynamic config is missing. "
            "Please upgrade: pip install -U 'torchao>=0.11'"
        ) from exc

    target = copy.deepcopy(model).eval()
    config = Int8DynamicActivationInt8WeightConfig()
    assert _torchao_quantize is not None
    _torchao_quantize(target, config)
    logger.info("Applied torchao INT8 dynamic quantization.")
    return target


# ---------------------------------------------------------------------------
# Benchmarking harness
# ---------------------------------------------------------------------------


def _model_size_mb(model: nn.Module) -> float:
    """Return the total parameter + buffer size in megabytes."""
    total_bytes = 0
    seen: set[int] = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total_bytes += t.nelement() * t.element_size()
    return total_bytes / (1024.0 * 1024.0)


def _measure_latency(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    *,
    n_warmup: int = 3,
    n_iters: int = 20,
) -> float:
    """Return mean forward-pass latency in milliseconds."""
    model.eval()
    with torch.inference_mode():
        for _ in range(n_warmup):
            model(*example_inputs)
        start = time.perf_counter()
        for _ in range(n_iters):
            model(*example_inputs)
        elapsed = time.perf_counter() - start
    return (elapsed / n_iters) * 1000.0


def benchmark_quantization(
    fp32_model: nn.Module,
    int8_model: Optional[nn.Module],
    int4_model: Optional[nn.Module],
    example_inputs: tuple[Any, ...],
    *,
    n_warmup: int = 3,
    n_iters: int = 20,
) -> dict[str, dict[str, float]]:
    """Benchmark FP32 / INT8 / INT4 variants side by side.

    Runs each supplied model ``n_iters`` times and reports mean latency
    (ms) and on-disk-equivalent size (MB). Safe to call even if
    torchao is missing — the function simply skips any variant that
    was passed ``None``.

    Args:
        fp32_model: Baseline FP32 model.
        int8_model: Optional INT8 quantized model.
        int4_model: Optional INT4 quantized model.
        example_inputs: Positional example inputs for ``model(*inputs)``.
        n_warmup: Warmup iterations excluded from timing.
        n_iters: Timed iterations.

    Returns:
        A nested dict keyed by variant with ``latency_ms`` and
        ``size_mb`` entries plus ``speedup`` relative to FP32.
    """
    results: dict[str, dict[str, float]] = {}
    base_latency = _measure_latency(
        fp32_model, example_inputs, n_warmup=n_warmup, n_iters=n_iters
    )
    results["fp32"] = {
        "latency_ms": base_latency,
        "size_mb": _model_size_mb(fp32_model),
        "speedup": 1.0,
    }
    if int8_model is not None:
        lat = _measure_latency(
            int8_model, example_inputs, n_warmup=n_warmup, n_iters=n_iters
        )
        results["int8"] = {
            "latency_ms": lat,
            "size_mb": _model_size_mb(int8_model),
            "speedup": base_latency / lat if lat > 0 else 0.0,
        }
    if int4_model is not None:
        lat = _measure_latency(
            int4_model, example_inputs, n_warmup=n_warmup, n_iters=n_iters
        )
        results["int4"] = {
            "latency_ms": lat,
            "size_mb": _model_size_mb(int4_model),
            "speedup": base_latency / lat if lat > 0 else 0.0,
        }
    return results


__all__ = [
    "benchmark_quantization",
    "quantize_int4_weight_only",
    "quantize_int8_dynamic",
]
