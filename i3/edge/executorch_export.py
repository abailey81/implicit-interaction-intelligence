"""ExecuTorch export pipeline for the Adaptive SLM.

ExecuTorch (https://pytorch.org/executorch/) is the PyTorch 2026
on-device runtime. The full export pipeline for an I3 SLM model is::

    +-------------------+   torch.export.export(...)
    |   AdaptiveSLM     | ------------------------+
    |  (nn.Module)      |                         v
    +-------------------+         +--------------------------+
                                  |    ExportedProgram       |
                                  |  (torch.fx graph + IR)   |
                                  +--------------------------+
                                           |
                       (optional) torchao quantization
                                           |
                                           v
                                  +--------------------------+
                                  |   EdgeProgramManager     |
                                  |   via to_edge(exported)  |
                                  +--------------------------+
                                           |
                                           v
                                  +---------------------------+
                                  | ExecuTorchProgramManager  |
                                  |  via .to_executorch()     |
                                  +---------------------------+
                                           |
                                           v
                                  +---------------------------+
                                  |   slm.pte  (on-disk)      |
                                  | ExecuTorch flat buffer    |
                                  +---------------------------+

The ``.pte`` (PyTorch ExecuTorch) container is a FlatBuffer-based
binary that pairs the lowered graph with the weight tensors in a
format that the ExecuTorch C++ runtime can load without Python. The
resulting file is ideal for deployment on Huawei Kirin SoCs, Apple
Silicon, and Android NNAPI targets.

Soft-imports ``executorch``. If the package is missing every function
in this module raises :class:`RuntimeError` with a clear install hint:
``pip install executorch`` (CPU-only wheel).

Usage::

    import torch
    from i3.slm.model import AdaptiveSLM
    from i3.edge.executorch_export import export_slm_to_executorch

    model = AdaptiveSLM().eval()
    example = (
        torch.zeros(1, 16, dtype=torch.long),  # input_ids
        torch.zeros(1, 8),                     # adaptation_vector
        torch.zeros(1, 64),                    # user_state
    )
    pte = export_slm_to_executorch(
        model=model,
        example_inputs=example,
        out_path=Path("slm.pte"),
        quantization="int8",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional

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
        "executorch not installed; .pte export will be unavailable. "
        "Install with: pip install executorch  (CPU-only wheel)"
    )


Quantization = Literal["none", "int8", "int4"]

_INSTALL_HINT: str = (
    "ExecuTorch is required to export a .pte file. "
    "Install the CPU-only wheel with:\n\n    pip install executorch\n\n"
    "See https://pytorch.org/executorch/ for hardware-specific builds."
)


def _require_executorch() -> ModuleType:
    """Return the ``executorch`` module or raise a RuntimeError."""
    if not _EXECUTORCH_AVAILABLE or _executorch_module is None:
        raise RuntimeError(_INSTALL_HINT)
    return _executorch_module


# ---------------------------------------------------------------------------
# Internal wrapper
# ---------------------------------------------------------------------------


class _SLMExecuTorchWrapper(nn.Module):
    """Wrap :class:`AdaptiveSLM` so ExecuTorch sees a tensor-only graph.

    ``AdaptiveSLM.forward`` returns a ``(logits, layer_info)`` tuple
    where ``layer_info`` is a nested ``dict`` — ExecuTorch's
    ``torch.export`` backend only handles pure tensor I/O, so the
    wrapper returns ``logits`` alone.
    """

    def __init__(self, slm: nn.Module) -> None:
        super().__init__()
        self.slm = slm

    def forward(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor,
        user_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the logits tensor for ExecuTorch export.

        Args:
            input_ids: ``[B, T]`` token ids.
            adaptation_vector: ``[B, 8]`` adaptation signal.
            user_state: ``[B, 64]`` user-state embedding.

        Returns:
            Logits tensor of shape ``[B, T, vocab_size]``.
        """
        logits, _ = self.slm(
            input_ids=input_ids,
            adaptation_vector=adaptation_vector,
            user_state=user_state,
        )
        return logits


def _apply_quantization(
    model: nn.Module, quantization: Quantization
) -> nn.Module:
    """Dispatch to the torchao quantization helpers."""
    if quantization == "none":
        return model
    # Local import so that a missing torchao never breaks FP32 export.
    from i3.slm.quantize_torchao import (
        quantize_int4_weight_only,
        quantize_int8_dynamic,
    )

    if quantization == "int8":
        return quantize_int8_dynamic(model)
    if quantization == "int4":
        return quantize_int4_weight_only(model)
    raise ValueError(
        f"Unknown quantization level: {quantization!r}. "
        f"Expected one of: none, int8, int4."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_slm_to_executorch(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    out_path: Path,
    quantization: Quantization = "int8",
) -> Path:
    """Export an :class:`AdaptiveSLM` to an ExecuTorch ``.pte`` file.

    Pipeline:

    1. ``torch.export.export(wrapped, example_inputs)`` → ``ExportedProgram``
    2. Apply torchao quantization (INT8 or INT4) if requested.
    3. ``to_edge(exported)`` → :class:`EdgeProgramManager`.
    4. ``.to_executorch()`` → :class:`ExecuTorchProgramManager`.
    5. ``.write_to_file(out_path)`` writes the ``.pte`` binary.

    Args:
        model: The SLM instance to export. Must be in ``eval`` mode.
        example_inputs: Positional example tensors for tracing. For
            :class:`AdaptiveSLM` this is typically
            ``(input_ids, adaptation_vector, user_state)``.
        out_path: Destination path for the ``.pte`` file.
        quantization: ``"none"``, ``"int8"``, or ``"int4"``.

    Returns:
        The absolute path to the written ``.pte`` file.

    Raises:
        RuntimeError: If ExecuTorch is not installed or any stage of
            the pipeline fails.
        ValueError: If ``quantization`` is not recognised.
    """
    _require_executorch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    wrapped = _SLMExecuTorchWrapper(model).eval()

    logger.info("Step 1/4: torch.export.export")
    try:
        exported = torch.export.export(wrapped, example_inputs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"torch.export.export failed ({type(exc).__name__}): {exc}"
        ) from exc

    if quantization != "none":
        logger.info("Step 2/4: applying torchao %s quantization", quantization)
        try:
            # Quantize the underlying module, then re-export the
            # quantized version so the ExportedProgram sees the
            # low-bit weights.
            q_wrapped = _apply_quantization(wrapped, quantization).eval()
            exported = torch.export.export(q_wrapped, example_inputs)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Quantization/re-export failed "
                f"({type(exc).__name__}): {exc}"
            ) from exc
    else:
        logger.info("Step 2/4: skipping quantization (FP32)")

    logger.info("Step 3/4: to_edge (EdgeProgramManager)")
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

    logger.info("Step 4/4: to_executorch + write_to_file")
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
                    "nor .buffer; cannot persist .pte."
                )
            out_path.write_bytes(bytes(buffer))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Writing .pte failed ({type(exc).__name__}): {exc}"
        ) from exc

    logger.info("Wrote ExecuTorch program to %s", out_path)
    return out_path.resolve()


__all__ = ["export_slm_to_executorch"]
