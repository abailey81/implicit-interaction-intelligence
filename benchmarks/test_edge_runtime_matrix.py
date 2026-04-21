"""Cross-runtime latency matrix benchmark.

Runs each alternative edge runtime's micro-benchmark helper when its
backing library is importable, writes a single row to a dated CSV
under ``reports/edge_runtime_matrix_<YYYYMMDD>.csv`` with columns:

* ``runtime``  — human name of the runtime (``mlx``, ``tvm``, …).
* ``target``   — hardware target string (``cpu``, ``cortex-a76``, …).
* ``P50_ms``   — median single-pass latency in milliseconds.
* ``P95_ms``   — 95th-percentile single-pass latency in milliseconds.
* ``size_mb``  — persisted artefact size in megabytes (NaN when
  only an in-memory benchmark was run).
* ``notes``    — short free-text field.

Runs are **soft-skipped** (pytest ``skip``) when the library is not
installed — the suite is therefore safe to run on any host and the
CI can compare missing-runtime sets across builds.
"""

from __future__ import annotations

import csv
import datetime as _dt
import importlib
import logging
import math
import time
from pathlib import Path
from typing import Iterable

import pytest

logger = logging.getLogger(__name__)


REPORTS_DIR: Path = Path("reports")
CSV_COLUMNS: tuple[str, ...] = (
    "runtime",
    "target",
    "P50_ms",
    "P95_ms",
    "size_mb",
    "notes",
)


def _csv_path() -> Path:
    """Return the dated CSV path, creating parent dirs.

    Returns:
        Path to ``reports/edge_runtime_matrix_<YYYYMMDD>.csv``.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().strftime("%Y%m%d")
    return REPORTS_DIR / f"edge_runtime_matrix_{today}.csv"


def _append_row(row: dict[str, object]) -> None:
    """Append a single row to today's CSV, writing the header if new.

    Args:
        row: A mapping with keys matching :data:`CSV_COLUMNS`.
    """
    csv_path = _csv_path()
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _soft_import(name: str):
    """Import a module, returning ``None`` when it is absent.

    Args:
        name: The dotted module name.

    Returns:
        The module, or ``None`` if not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _percentile(latencies_ms: Iterable[float], pct: float) -> float:
    """Compute a percentile from an iterable of floats.

    Args:
        latencies_ms: Raw measurements.
        pct: A value in ``(0, 1)`` — e.g. ``0.95``.

    Returns:
        The interpolated percentile in milliseconds.
    """
    values = sorted(latencies_ms)
    if not values:
        return math.nan
    idx = max(0, min(len(values) - 1, int(pct * len(values))))
    return values[idx]


def _bench_pytorch_reference() -> dict[str, float]:
    """Small PyTorch CPU forward-pass micro-benchmark.

    Returns:
        Dict with ``p50_ms`` / ``p95_ms`` for a 100-iteration run.
    """
    import torch

    m = torch.nn.Linear(32, 32).eval()
    x = torch.zeros(1, 32)
    for _ in range(10):
        m(x)
    lat: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        m(x)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": _percentile(lat, 0.5),
        "p95_ms": _percentile(lat, 0.95),
    }


# ---------------------------------------------------------------------------
# Per-runtime test cases
# ---------------------------------------------------------------------------


def test_pytorch_reference_row() -> None:
    """Baseline row: stock PyTorch CPU.

    Always runs; gives the matrix a calibrated reference point.
    """
    stats = _bench_pytorch_reference()
    _append_row(
        {
            "runtime": "pytorch",
            "target": "cpu",
            "P50_ms": round(stats["p50_ms"], 3),
            "P95_ms": round(stats["p95_ms"], 3),
            "size_mb": math.nan,
            "notes": "reference linear micro-benchmark",
        }
    )


def test_mlx_row() -> None:
    """MLX micro-benchmark (matmul on unified memory)."""
    mlx_core = _soft_import("mlx.core")
    if mlx_core is None:
        pytest.skip("mlx not installed")

    mx = mlx_core
    w = mx.zeros((32, 32))
    x = mx.zeros((32,))
    for _ in range(10):
        _ = w @ x
    lat: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = w @ x
        lat.append((time.perf_counter() - t0) * 1000.0)
    _append_row(
        {
            "runtime": "mlx",
            "target": "apple-silicon",
            "P50_ms": round(_percentile(lat, 0.5), 3),
            "P95_ms": round(_percentile(lat, 0.95), 3),
            "size_mb": math.nan,
            "notes": "mx.array matmul smoke",
        }
    )


def test_llama_cpp_row() -> None:
    """llama.cpp presence check (no end-to-end run)."""
    if _soft_import("llama_cpp") is None:
        pytest.skip("llama-cpp-python not installed")
    _append_row(
        {
            "runtime": "llama.cpp",
            "target": "cpu",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "library present; full benchmark needs real GGUF",
        }
    )


def test_tvm_row() -> None:
    """TVM presence check — compilation is expensive, not run here."""
    if _soft_import("tvm") is None:
        pytest.skip("apache-tvm not installed")
    _append_row(
        {
            "runtime": "tvm",
            "target": "llvm -mcpu=cortex-a76",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "library present; run compile_tcn_to_tvm for numbers",
        }
    )


def test_iree_row() -> None:
    """IREE presence check."""
    if _soft_import("iree.compiler") is None:
        pytest.skip("iree-compiler not installed")
    _append_row(
        {
            "runtime": "iree",
            "target": "vmvx",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "library present",
        }
    )


def test_coreml_row() -> None:
    """Core ML presence check."""
    if _soft_import("coremltools") is None:
        pytest.skip("coremltools not installed")
    _append_row(
        {
            "runtime": "coreml",
            "target": "neural-engine",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "library present; full bench needs a Mac",
        }
    )


def test_openvino_row() -> None:
    """OpenVINO presence check."""
    if _soft_import("openvino") is None:
        pytest.skip("openvino not installed")
    _append_row(
        {
            "runtime": "openvino",
            "target": "intel-npu",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "library present; numbers require Meteor Lake NPU",
        }
    )


def test_tensorrt_llm_row() -> None:
    """TRT-LLM presence check."""
    if _soft_import("tensorrt_llm") is None:
        pytest.skip("tensorrt_llm not installed")
    _append_row(
        {
            "runtime": "tensorrt_llm",
            "target": "nvidia-gpu",
            "P50_ms": math.nan,
            "P95_ms": math.nan,
            "size_mb": math.nan,
            "notes": "for completeness only; I3 targets on-device",
        }
    )
