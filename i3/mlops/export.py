"""ONNX export entry point for I3 models.

This module is a thin dispatcher that imports the per-model exporters
lazily so that missing optional dependencies (onnx, onnxruntime) do not
break module import.  The actual graph-export logic lives in
:mod:`i3.encoder.onnx_export` and :mod:`i3.slm.onnx_export`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def export_encoder(
    model: Any,
    output_path: str | Path,
    *,
    opset: int = 17,
    dynamic_axes: bool = True,
    verify: bool = True,
) -> Path:
    """Export a TCN encoder to ONNX.

    Args:
        model: A :class:`i3.encoder.tcn.TemporalConvNet` instance.
        output_path: Destination ``.onnx`` path.
        opset: Target ONNX opset (default 17).
        dynamic_axes: If ``True``, batch and time dims are exported as
            dynamic axes.
        verify: If ``True``, run a PyTorch-vs-ONNXRuntime parity check.

    Returns:
        Path to the exported ONNX file.
    """
    from i3.encoder.onnx_export import export_tcn

    return export_tcn(
        model,
        Path(output_path),
        opset=opset,
        dynamic_axes=dynamic_axes,
        verify=verify,
    )


def export_slm(
    model: Any,
    output_path: str | Path,
    *,
    opset: int = 17,
    dynamic_axes: bool = True,
    verify: bool = True,
) -> Path:
    """Export an Adaptive SLM (prefill variant) to ONNX.

    Args:
        model: A :class:`i3.slm.model.AdaptiveSLM` instance.
        output_path: Destination ``.onnx`` path.
        opset: Target ONNX opset (default 17).
        dynamic_axes: If ``True``, batch and time dims are exported as
            dynamic axes.
        verify: If ``True``, run a PyTorch-vs-ONNXRuntime parity check.

    Returns:
        Path to the exported ONNX file.
    """
    from i3.slm.onnx_export import export_slm as _export_slm

    return _export_slm(
        model,
        Path(output_path),
        opset=opset,
        dynamic_axes=dynamic_axes,
        verify=verify,
    )


def export_all(
    encoder: Any,
    slm: Any,
    out_dir: str | Path,
    *,
    opset: int = 17,
    verify: bool = True,
) -> dict[str, Path]:
    """Export both the encoder and SLM to ``out_dir``.

    Args:
        encoder: TCN encoder instance (or ``None`` to skip).
        slm: Adaptive SLM instance (or ``None`` to skip).
        out_dir: Output directory for the ``.onnx`` files.
        opset: Target ONNX opset.
        verify: If ``True``, run parity checks after each export.

    Returns:
        Mapping of model label to exported file path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Path] = {}
    if encoder is not None:
        exports["encoder"] = export_encoder(
            encoder, out / "encoder.onnx", opset=opset, verify=verify
        )
    if slm is not None:
        exports["slm"] = export_slm(
            slm, out / "slm.onnx", opset=opset, verify=verify
        )
    return exports


__all__ = ["export_all", "export_encoder", "export_slm"]
