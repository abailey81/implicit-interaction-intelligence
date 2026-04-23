"""torchao INT8 dynamic quantization for the TCN user-state encoder.

Companion to :mod:`i3.slm.quantize_torchao` — provides a single-call
entry point to apply INT8 dynamic quantization to a
:class:`i3.encoder.tcn.TemporalConvNet` instance using the modern
``torchao`` stack.

Soft-imports ``torchao``. The module is always importable; calling the
quantization function without the dependency raises
:class:`RuntimeError` with an install hint.

Usage::

    from i3.encoder.tcn import TemporalConvNet
    from i3.encoder.quantize import quantize_tcn_int8_dynamic

    encoder = TemporalConvNet().eval()
    q_encoder = quantize_tcn_int8_dynamic(encoder)
"""

from __future__ import annotations

import copy
import logging
from types import ModuleType

import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import torchao as _torchao_module  # type: ignore[import-not-found]
    from torchao.quantization import (
        quantize_ as _torchao_quantize,  # type: ignore[import-not-found]
    )

    _TORCHAO_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _torchao_module = None  # type: ignore[assignment]
    _torchao_quantize = None  # type: ignore[assignment]
    _TORCHAO_AVAILABLE = False
    logger.info(
        "torchao not installed; TCN INT8 quantization via torchao will be "
        "unavailable. Install with: pip install torchao>=0.11"
    )


_INSTALL_HINT: str = (
    "torchao is required to quantize the TCN encoder. "
    "Install it with: pip install 'torchao>=0.11'"
)


def _require_torchao() -> ModuleType:
    """Return the torchao module or raise a clear RuntimeError."""
    if not _TORCHAO_AVAILABLE or _torchao_module is None:
        raise RuntimeError(_INSTALL_HINT)
    return _torchao_module


def quantize_tcn_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply torchao INT8 dynamic quantization to a TCN encoder.

    Only the ``nn.Linear`` layers inside the TCN (input and output
    projections) are quantized — the dilated causal convolutions and
    :class:`nn.LayerNorm` / :class:`nn.ReLU` blocks are left in FP32
    because quantizing temporal convolution kernels typically degrades
    the learned contrastive geometry on the unit hypersphere.

    Args:
        model: The FP32 TCN encoder to quantize. Must be in ``eval``
            mode (the function will call ``.eval()`` defensively).

    Returns:
        A deep copy of ``model`` with eligible Linear weights
        quantized to INT8 via torchao.

    Raises:
        RuntimeError: If ``torchao`` is not installed.
    """
    _require_torchao()
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,  # type: ignore[import-not-found]
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torchao is installed but the INT8 dynamic config is missing. "
            "Please upgrade: pip install -U 'torchao>=0.11'"
        ) from exc

    target = copy.deepcopy(model).eval()
    config = Int8DynamicActivationInt8WeightConfig()
    assert _torchao_quantize is not None
    _torchao_quantize(target, config)
    logger.info(
        "Applied torchao INT8 dynamic quantization to TCN encoder (%s).",
        type(target).__name__,
    )
    return target


__all__ = ["quantize_tcn_int8_dynamic"]
