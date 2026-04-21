"""TensorRT-LLM exporter for the Adaptive SLM.

`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ is NVIDIA's
LLM-specialised inference stack. It builds on TensorRT's low-level
compiler with paged attention, KV-cache management, in-flight batching
and SmoothQuant INT8 / INT4 paths. It is the fastest-known serving
runtime for NVIDIA GPUs at 2026 and is included here **for
completeness only** — the I3 product surface targets on-device
deployment, not data-centre GPUs.

This module therefore:

* Soft-imports ``tensorrt_llm`` so the file is importable everywhere.
* Raises a clear :class:`RuntimeError` at call-time on any non-NVIDIA
  host (no CUDA runtime available), explaining that TensorRT-LLM
  requires an NVIDIA GPU with CUDA 12 or newer.
* Provides a ``convert_slm_to_trtllm`` stub that constructs a
  TRT-LLM builder config when the environment is valid.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from types import ModuleType
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import tensorrt_llm as _trtllm  # type: ignore[import-not-found]

    _TRTLLM_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _trtllm = None  # type: ignore[assignment]
    _TRTLLM_AVAILABLE = False
    logger.info(
        "tensorrt_llm not installed; TRT-LLM export will be unavailable. "
        "Install with: pip install tensorrt_llm  (NVIDIA GPU + CUDA 12+ required)"
    )


Dtype = Literal["fp16", "bf16", "int8", "int4"]

_INSTALL_HINT: str = (
    "tensorrt_llm is required to build a TRT-LLM engine. Install with:\n\n"
    "    pip install tensorrt_llm\n\n"
    "NOTE: This package only works on Linux x86_64 or aarch64 hosts "
    "with an NVIDIA GPU, driver >= 550, and CUDA 12.4 or newer. "
    "See https://nvidia.github.io/TensorRT-LLM/."
)

_NO_GPU_HINT: str = (
    "TensorRT-LLM requires an NVIDIA GPU with CUDA 12+. This host has "
    "no CUDA runtime visible (no nvidia-smi on PATH and "
    "CUDA_VISIBLE_DEVICES is empty). The I3 product does not target "
    "data-centre GPUs — this module is shipped for completeness only."
)


def _require_trtllm() -> ModuleType:
    """Return the ``tensorrt_llm`` module or raise.

    Raises:
        RuntimeError: If ``tensorrt_llm`` is not installed.
    """
    if not _TRTLLM_AVAILABLE or _trtllm is None:
        raise RuntimeError(_INSTALL_HINT)
    return _trtllm


def _nvidia_gpu_available() -> bool:
    """Return True when at least one NVIDIA GPU is visible.

    Detection uses two heuristics:

    * ``nvidia-smi`` binary is on ``PATH``.
    * Either ``CUDA_VISIBLE_DEVICES`` is unset (all GPUs visible) or
      it is set to a non-empty value.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and visible.strip() == "":
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_slm_to_trtllm(
    onnx_path: Path,
    out_dir: Path,
    dtype: Dtype = "int8",
) -> Path:
    """Build a TensorRT-LLM engine from an ONNX-serialised SLM.

    The real conversion is delegated to the TRT-LLM ``Builder`` +
    ``Network`` APIs. This function refuses to run on a non-NVIDIA
    host; that is deliberate — the I3 on-device story does not route
    through a data-centre GPU.

    Args:
        onnx_path: Path to the ONNX SLM file.
        out_dir: Directory that will contain ``rank0.engine`` and
            ``config.json``.
        dtype: One of ``"fp16"``, ``"bf16"``, ``"int8"``, ``"int4"``.

    Returns:
        The resolved path of ``out_dir``.

    Raises:
        RuntimeError: If ``tensorrt_llm`` is not installed, no NVIDIA
            GPU is visible, or the engine build fails.
        FileNotFoundError: If ``onnx_path`` does not exist.
    """
    _require_trtllm()
    if not _nvidia_gpu_available():
        raise RuntimeError(_NO_GPU_HINT)

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Happy-path integration stub. Real TRT-LLM build uses the
    # tensorrt_llm.builder.Builder API; its signature has changed
    # between releases, so we import lazily.
    try:
        from tensorrt_llm.builder import Builder  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt_llm is installed but tensorrt_llm.builder.Builder "
            "is missing. Please upgrade: pip install -U tensorrt_llm"
        ) from exc

    try:
        builder = Builder()
        builder_config = builder.create_builder_config(
            name="i3_slm",
            precision=dtype,
        )
        # NOTE: a full implementation wires an ONNX parser through a
        # TRT-LLM Network here. The shape below is a placeholder that
        # makes the build config self-consistent; replace with the
        # real graph when integrating.
        network = builder.create_network()
        engine = builder.build_engine(network, builder_config)
        if engine is None:
            raise RuntimeError(
                "tensorrt_llm.Builder.build_engine returned None — "
                "check TRT logs for the failed layer."
            )
        (out_dir / "rank0.engine").write_bytes(bytes(engine))
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "TensorRT-LLM engine build failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    logger.info("Wrote TRT-LLM engine to %s", out_dir)
    return out_dir.resolve()


__all__ = ["convert_slm_to_trtllm"]
