"""IREE (MLIR) exporter for I3 models.

`IREE <https://iree.dev/>`_ — Intermediate Representation Execution
Environment — is an MLIR-based compiler and runtime originally spun
out of Google. Unlike TVM, which has its own IR stack, IREE lowers
models through stable MLIR dialects (``linalg``, ``flow``,
``stream``, ``hal``) down to one of several back-ends: ``vmvx`` (a
portable interpreter VM), ``llvm-cpu`` (AOT-compiled native CPU
kernels) or ``vulkan-spirv`` (cross-vendor GPU). IREE is a natural
fit for workloads that already sit on an ONNX / StableHLO front-end
and need deterministic, AOT-compiled, cross-platform binaries.

Pipeline::

    +-----------+  onnx         +---------------+  iree.compiler.onnx
    | PyTorch   | ------------> |   ONNX file   | ----------------+
    +-----------+               +---------------+                 |
                                                                  v
                                                      +--------------------+
                                                      |    MLIR module     |
                                                      +--------------------+
                                                                  |
                                               iree-compile -iree-hal-target-backends=<backend>
                                                                  |
                                                                  v
                                                      +--------------------+
                                                      |   module.vmfb      |
                                                      | (FlatBuffer bin)   |
                                                      +--------------------+

Soft-imports ``iree.compiler`` and ``iree.runtime``. If either is
missing, functions raise :class:`RuntimeError` with the install hint
``pip install iree-compiler iree-runtime``.

References:
    IREE project — https://iree.dev/
    StableHLO spec — https://openxla.org/stablehlo
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import iree.compiler as _iree_compiler  # type: ignore[import-not-found]

    _IREE_COMPILER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _iree_compiler = None  # type: ignore[assignment]
    _IREE_COMPILER_AVAILABLE = False
    logger.info(
        "iree-compiler not installed; IREE compilation unavailable. "
        "Install with: pip install iree-compiler iree-runtime"
    )


IREEBackend = Literal["vmvx", "llvm-cpu", "vulkan-spirv"]
SUPPORTED_BACKENDS: tuple[IREEBackend, ...] = (
    "vmvx",
    "llvm-cpu",
    "vulkan-spirv",
)


_INSTALL_HINT: str = (
    "IREE is required for MLIR compilation. Install with:\n\n"
    "    pip install iree-compiler iree-runtime\n\n"
    "See https://iree.dev/guides/ for platform-specific notes."
)


def _require_iree_compiler() -> ModuleType:
    """Return the ``iree.compiler`` module or raise.

    Raises:
        RuntimeError: If IREE is not installed.
    """
    if not _IREE_COMPILER_AVAILABLE or _iree_compiler is None:
        raise RuntimeError(_INSTALL_HINT)
    return _iree_compiler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_to_iree(
    onnx_path: Path,
    backend: IREEBackend = "vmvx",
    out_path: Path | None = None,
) -> Path:
    """Compile an ONNX graph to an IREE ``.vmfb`` FlatBuffer.

    Uses ``iree.compiler.tools.compile_file`` (or the newer
    ``iree.compiler.onnx.compile_file``, whichever is available in the
    installed package) with ``-iree-hal-target-backends=<backend>``.

    Args:
        onnx_path: Path to the ONNX file to compile.
        backend: One of ``"vmvx"`` (portable VM interpreter),
            ``"llvm-cpu"`` (native AOT CPU), or ``"vulkan-spirv"``
            (cross-vendor GPU). Defaults to ``"vmvx"`` which is the
            most portable option.
        out_path: Destination ``.vmfb`` path. Defaults to
            ``<onnx_stem>.vmfb`` next to the input.

    Returns:
        The resolved path of the ``.vmfb`` file.

    Raises:
        RuntimeError: If IREE is not installed, the compiler cannot be
            invoked, or compilation fails.
        FileNotFoundError: If ``onnx_path`` does not exist.
        ValueError: If ``backend`` is not in :data:`SUPPORTED_BACKENDS`.
    """
    iree_compiler = _require_iree_compiler()

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported IREE backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}."
        )

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_path = (
        Path(out_path)
        if out_path is not None
        else onnx_path.with_suffix(".vmfb")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The compile_file API has moved around between iree-compiler
    # minor versions. Try the modern onnx-focused entry point first,
    # then the generic one.
    compile_file = None
    onnx_module = getattr(iree_compiler, "onnx", None)
    if onnx_module is not None:
        compile_file = getattr(onnx_module, "compile_file", None)
    if compile_file is None:
        tools = getattr(iree_compiler, "tools", None)
        if tools is not None:
            compile_file = getattr(tools, "compile_file", None)
    if compile_file is None:
        raise RuntimeError(
            "iree-compiler is installed but neither "
            "iree.compiler.onnx.compile_file nor "
            "iree.compiler.tools.compile_file was found. Upgrade with: "
            "pip install -U iree-compiler"
        )

    try:
        compile_file(
            str(onnx_path),
            input_type="onnx",
            target_backends=[backend],
            output_file=str(out_path),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "iree compilation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    logger.info(
        "Wrote IREE .vmfb (backend=%s) to %s", backend, out_path
    )
    return out_path.resolve()


__all__ = ["SUPPORTED_BACKENDS", "compile_to_iree"]
