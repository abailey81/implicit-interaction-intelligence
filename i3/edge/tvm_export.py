"""Apache TVM exporter for the I3 TCN encoder.

`Apache TVM <https://tvm.apache.org/>`_ is an open-source ML compiler
with an aggressive operator fusion and tensor scheduling stack (AutoTVM
+ Ansor/MetaSchedule) that consistently produces the fastest
CPU/GPU/mobile-NPU kernels at a given bit-width.

    "TVM: An Automated End-to-End Optimizing Compiler for Deep
    Learning" — Chen et al., OSDI 2018.
    https://arxiv.org/abs/1802.04799

TVM's compilation flow for I3 is::

    +-----------+   onnx_exporter    +------+   tvm.relay.frontend.from_onnx
    | PyTorch   | -----------------> | ONNX | -------------------------+
    +-----------+                    +------+                          |
                                                                       v
                                                           +---------------------+
                                                           |    Relay IR         |
                                                           +---------------------+
                                                                       |
                                                       tvm.relay.build(target=...)
                                                                       |
                                                                       v
                                                           +---------------------+
                                                           |  tvm.runtime.Module |
                                                           |  + params.bin       |
                                                           +---------------------+

Targets covered:

* ``llvm`` — x86-64 CPU (server / developer laptop).
* ``llvm -mcpu=cortex-a76`` — Kirin 9-series class ARM CPU (Cortex-A76
  big cores) — the target for Smart Hanhan and related Huawei edge
  devices.
* ``opencl`` — Kirin Mali GPU via the OpenCL back-end.
* ``vulkan`` — portable GPU path (desktop + recent Android).

Soft-imports ``tvm``. If absent every function raises
:class:`RuntimeError` with the hint ``pip install apache-tvm``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import tvm as _tvm  # type: ignore[import-not-found]
    from tvm import relay as _tvm_relay  # type: ignore[import-not-found]

    _TVM_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _tvm = None  # type: ignore[assignment]
    _tvm_relay = None  # type: ignore[assignment]
    _TVM_AVAILABLE = False
    logger.info(
        "tvm not installed; TVM compilation will be unavailable. "
        "Install with: pip install apache-tvm"
    )


SUPPORTED_TARGETS: tuple[str, ...] = (
    "llvm",
    "llvm -mcpu=cortex-a76",
    "opencl",
    "vulkan",
)


_INSTALL_HINT: str = (
    "Apache TVM is required to compile via Relay. Install with:\n\n"
    "    pip install apache-tvm\n\n"
    "For arm64 / mobile targets you typically need a source build with "
    "the right LLVM cross-compiler — see "
    "https://tvm.apache.org/docs/install/from_source.html."
)


def _require_tvm() -> tuple[ModuleType, ModuleType]:
    """Return the ``(tvm, tvm.relay)`` modules or raise.

    Raises:
        RuntimeError: If TVM is not installed.
    """
    if not _TVM_AVAILABLE or _tvm is None or _tvm_relay is None:
        raise RuntimeError(_INSTALL_HINT)
    return _tvm, _tvm_relay


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_tcn_to_tvm(
    onnx_path: Path,
    target: str = "llvm -mcpu=cortex-a76",
    out_dir: Path | None = None,
) -> Path:
    """Compile an ONNX-serialised TCN to a TVM runtime module.

    Pipeline:

    1. ``onnx.load`` loads the graph.
    2. ``tvm.relay.frontend.from_onnx`` lifts it to Relay IR.
    3. ``tvm.relay.build(mod, target=..., params=params)`` produces a
       ``GraphExecutorFactoryModule``.
    4. The ``.so``, graph JSON and params are written to ``out_dir``.

    Args:
        onnx_path: Path to the ONNX file to compile.
        target: A TVM target string. Defaults to
            ``"llvm -mcpu=cortex-a76"``. Must be one of
            :data:`SUPPORTED_TARGETS` (or a TVM-parseable variant).
        out_dir: Output directory for ``lib.so`` +
            ``graph.json`` + ``params.bin``. Defaults to a sibling
            directory of ``onnx_path`` named ``<stem>_tvm``.

    Returns:
        The resolved path of the output directory.

    Raises:
        RuntimeError: If TVM is not installed, the ONNX file cannot be
            loaded, or compilation fails.
        FileNotFoundError: If ``onnx_path`` does not exist.
    """
    tvm, relay = _require_tvm()

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_dir = (
        Path(out_dir)
        if out_dir is not None
        else onnx_path.with_name(onnx_path.stem + "_tvm")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required to load the graph for TVM compilation. "
            "Install with: pip install onnx"
        ) from exc

    try:
        onnx_model = onnx.load(str(onnx_path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to load ONNX file {onnx_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        mod, params = relay.frontend.from_onnx(onnx_model)
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            "tvm.relay.frontend.from_onnx failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    logger.info("Compiling with TVM target=%s", target)
    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "tvm.relay.build failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    lib_path = out_dir / "lib.tar"
    try:
        lib.export_library(str(lib_path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"lib.export_library failed: {type(exc).__name__}: {exc}"
        ) from exc

    logger.info("Wrote TVM module to %s", lib_path)
    return out_dir.resolve()


def benchmark_tvm(
    module: Any, input_shape: tuple[int, ...]
) -> dict[str, float]:
    """Benchmark a compiled TVM module with warmup + 100 iterations.

    Args:
        module: A ``tvm.runtime.Module`` or ``GraphModule``. Must
            expose ``run`` / ``set_input`` / ``get_output`` semantics
            (the standard TVM Graph Executor interface).
        input_shape: Shape of the single model input (batch
            included).

    Returns:
        A dict with keys ``p50_ms``, ``p95_ms``, ``mean_ms``,
        ``n_iter`` for the measured latency.

    Raises:
        RuntimeError: If TVM is not installed or the module does not
            expose the expected runtime API.
    """
    tvm, _ = _require_tvm()

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "numpy is required for benchmark_tvm."
        ) from exc

    set_input = getattr(module, "set_input", None)
    run = getattr(module, "run", None)
    if not callable(set_input) or not callable(run):
        raise RuntimeError(
            "benchmark_tvm expects a TVM GraphModule with set_input() "
            "and run() methods — got " + type(module).__name__
        )

    rng = np.random.default_rng(0)
    dummy = rng.standard_normal(input_shape).astype("float32")
    tvm_arr = tvm.nd.array(dummy)

    # Discover the first input name via the runtime introspection API
    # if available; otherwise fall back to positional indexing.
    try:
        input_name = module.get_input_info()  # type: ignore[attr-defined]
        first = next(iter(input_name)) if input_name else 0
        set_input(first, tvm_arr)
    except (AttributeError, StopIteration):
        set_input(0, tvm_arr)

    # Warm-up
    for _ in range(10):
        run()

    latencies_ms: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        run()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    return {
        "p50_ms": latencies_ms[len(latencies_ms) // 2],
        "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)],
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "n_iter": float(len(latencies_ms)),
    }


__all__ = [
    "SUPPORTED_TARGETS",
    "benchmark_tvm",
    "compile_tcn_to_tvm",
]
