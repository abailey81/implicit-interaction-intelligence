"""ExecuTorch export pipeline for the TCN user-state encoder.

Companion to :mod:`i3.edge.executorch_export`. Same four-stage
pipeline (``torch.export`` → optional torchao quant → ``to_edge`` →
``to_executorch`` → ``.pte``) but targets
:class:`i3.encoder.tcn.TemporalConvNet`.

Soft-imports ``executorch``. If the package is missing, calling the
exporter raises :class:`RuntimeError` with the install hint.

Usage::

    import torch
    from i3.encoder.tcn import TemporalConvNet
    from i3.edge.tcn_executorch_export import export_tcn_to_executorch

    encoder = TemporalConvNet().eval()
    example = (torch.zeros(1, 100, 32),)  # [B, seq_len, input_dim]
    pte = export_tcn_to_executorch(
        model=encoder,
        example_inputs=example,
        out_path=Path("tcn.pte"),
        quantization="int8",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import executorch as _executorch_module  # type: ignore[import-not-found]

    _EXECUTORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _executorch_module = None  # type: ignore[assignment]
    _EXECUTORCH_AVAILABLE = False
    logger.info(
        "executorch not installed; TCN .pte export will be unavailable. "
        "Install with: pip install executorch  (CPU-only wheel)"
    )


Quantization = Literal["none", "int8"]

_INSTALL_HINT: str = (
    "ExecuTorch is required to export the TCN encoder to a .pte file. "
    "Install the CPU-only wheel with:\n\n    pip install executorch\n\n"
    "See https://pytorch.org/executorch/ for hardware-specific builds."
)


def _require_executorch() -> ModuleType:
    """Return the ``executorch`` module or raise a RuntimeError."""
    if not _EXECUTORCH_AVAILABLE or _executorch_module is None:
        raise RuntimeError(_INSTALL_HINT)
    return _executorch_module


def _apply_tcn_quantization(
    model: nn.Module, quantization: Quantization
) -> nn.Module:
    """Apply torchao INT8 dynamic quantization to the TCN if requested."""
    if quantization == "none":
        return model
    if quantization == "int8":
        from i3.encoder.quantize import quantize_tcn_int8_dynamic

        return quantize_tcn_int8_dynamic(model)
    raise ValueError(
        f"Unknown TCN quantization level: {quantization!r}. "
        f"Expected one of: none, int8."
    )


def export_tcn_to_executorch(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    out_path: Path,
    quantization: Quantization = "int8",
) -> Path:
    """Export a TCN encoder to an ExecuTorch ``.pte`` file.

    Args:
        model: The :class:`TemporalConvNet` (or compatible) instance.
        example_inputs: Positional example tensors — typically
            ``(torch.zeros(1, seq_len, input_dim),)``.
        out_path: Destination path for the ``.pte`` file.
        quantization: ``"none"`` or ``"int8"``. INT4 is not recommended
            for the TCN because weight-only quantization of 1D
            convolutions typically degrades the contrastive geometry.

    Returns:
        The absolute path to the written ``.pte`` file.

    Raises:
        RuntimeError: If ExecuTorch is not installed or any stage fails.
        ValueError: If ``quantization`` is not recognised.
    """
    _require_executorch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()

    logger.info("Step 1/4: torch.export.export (TCN)")
    try:
        exported = torch.export.export(model, example_inputs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"torch.export.export failed ({type(exc).__name__}): {exc}"
        ) from exc

    if quantization != "none":
        logger.info(
            "Step 2/4: applying torchao %s quantization to TCN", quantization
        )
        try:
            q_model = _apply_tcn_quantization(model, quantization).eval()
            exported = torch.export.export(q_model, example_inputs)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"TCN quantization/re-export failed "
                f"({type(exc).__name__}): {exc}"
            ) from exc
    else:
        logger.info("Step 2/4: skipping quantization (FP32 TCN)")

    logger.info("Step 3/4: to_edge (TCN)")
    try:
        from executorch.exir import to_edge  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "executorch is installed but 'executorch.exir.to_edge' is "
            "missing. Please upgrade: pip install -U 'executorch>=0.4'"
        ) from exc
    try:
        edge_program = to_edge(exported)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"to_edge failed ({type(exc).__name__}): {exc}"
        ) from exc

    logger.info("Step 4/4: to_executorch + write_to_file (TCN)")
    try:
        et_program = edge_program.to_executorch()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"to_executorch failed ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        write_fn = getattr(et_program, "write_to_file", None)
        if callable(write_fn):
            write_fn(str(out_path))
        else:
            buffer = getattr(et_program, "buffer", None)
            if buffer is None:
                raise RuntimeError(
                    "ExecuTorchProgramManager has neither write_to_file() "
                    "nor .buffer; cannot persist TCN .pte."
                )
            out_path.write_bytes(bytes(buffer))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Writing TCN .pte failed ({type(exc).__name__}): {exc}"
        ) from exc

    logger.info("Wrote TCN ExecuTorch program to %s", out_path)
    return out_path.resolve()


__all__ = ["export_tcn_to_executorch"]
