"""Inference benchmarking CLI for I3.

Benchmarks four variants of the TCN encoder and Adaptive SLM:

    * PyTorch FP32
    * PyTorch INT8 (``quantize_dynamic`` on ``nn.Linear``)
    * ONNX FP32   (requires a ``.onnx`` export)
    * ONNX INT8   (requires ``onnxruntime.quantization.quantize_dynamic``
                    to have been run on the FP32 ``.onnx``)

Produces a Markdown table and optionally writes a JSON report.

Usage::

    python scripts/benchmark_inference.py \\
        --encoder-onnx exports/encoder.onnx \\
        --slm-onnx     exports/slm.onnx     \\
        --iterations   100                  \\
        --output       reports/bench.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("i3.scripts.benchmark_inference")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-percentile of ``values`` using linear interpolation.

    Args:
        values: Non-empty list of floats.
        pct: Percentile in ``[0, 100]``.

    Returns:
        Interpolated percentile value.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _time_fn(fn: Callable[[], Any], iterations: int, warmup: int = 5) -> dict[str, float]:
    """Time ``fn`` ``iterations`` times after ``warmup`` warmup calls.

    Args:
        fn: Zero-arg callable to time.
        iterations: Number of measured calls.
        warmup: Number of warmup calls whose timings are discarded.

    Returns:
        Dict with ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
        ``min_ms``, ``max_ms``.
    """
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iterations": len(samples),
    }


# --------------------------------------------------------------------------- #
# Benchmark variants
# --------------------------------------------------------------------------- #


def _bench_pytorch_tcn(iterations: int, int8: bool = False) -> dict[str, float]:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn

    from i3.encoder.tcn import TemporalConvNet

    model = TemporalConvNet().eval()
    if int8:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TCN INT8 quantisation failed (%s); using FP32.", exc)
    x = torch.randn(1, 10, 32)

    def _step() -> None:
        with torch.no_grad():
            model(x)

    return _time_fn(_step, iterations)


def _bench_pytorch_slm(iterations: int, int8: bool = False) -> dict[str, float]:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn

    from i3.slm.model import AdaptiveSLM

    model = AdaptiveSLM().eval()
    if int8:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SLM INT8 quantisation failed (%s); using FP32.", exc)
    input_ids = torch.randint(0, max(model.vocab_size, 1), (1, 16), dtype=torch.long)

    def _step() -> None:
        with torch.no_grad():
            model(input_ids)

    return _time_fn(_step, iterations)


def _bench_onnx(onnx_path: Path, kind: str, iterations: int) -> dict[str, float]:
    import numpy as np

    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        logger.warning("onnxruntime unavailable (%s); skipping %s", exc, onnx_path)
        return {}

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )

    if kind == "encoder":
        x = np.random.randn(1, 10, 32).astype(np.float32)

        def _step() -> None:
            session.run(["embedding"], {"input": x})
    elif kind == "slm":
        input_ids = np.random.randint(0, 8000, size=(1, 16), dtype=np.int64)
        cond = np.random.randn(1, 4, 256).astype(np.float32)
        mask = np.ones((1, 16), dtype=np.int64)

        def _step() -> None:
            session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "conditioning_tokens": cond,
                    "attention_mask": mask,
                },
            )
    else:
        raise ValueError(f"unknown kind: {kind}")

    return _time_fn(_step, iterations)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _render_markdown(results: dict[str, dict[str, float]]) -> str:
    """Render a benchmark dict as a concise Markdown table.

    Args:
        results: Mapping of variant name to timing dict.

    Returns:
        Markdown string.
    """
    lines = [
        "| Variant | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | N |",
        "|---------|-----------|----------|----------|----------|---|",
    ]
    for name, stats in results.items():
        if not stats:
            lines.append(f"| {name} | _skipped_ | | | | 0 |")
            continue
        lines.append(
            f"| {name} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['p99_ms']:.3f} | {stats['iterations']} |"
        )
    return "\n".join(lines)


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on unexpected error.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark I3 models across PyTorch and ONNX variants."
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--encoder-onnx", type=Path, default=None)
    parser.add_argument("--slm-onnx", type=Path, default=None)
    parser.add_argument("--encoder-onnx-int8", type=Path, default=None)
    parser.add_argument("--slm-onnx-int8", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-torch-int8",
        action="store_true",
        help="Skip PyTorch INT8 variants (sometimes slower than FP32 on CPU).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    results: dict[str, dict[str, float]] = {}

    # PyTorch variants
    results["tcn.pytorch.fp32"] = _bench_pytorch_tcn(args.iterations, int8=False)
    results["slm.pytorch.fp32"] = _bench_pytorch_slm(args.iterations, int8=False)
    if not args.skip_torch_int8:
        results["tcn.pytorch.int8"] = _bench_pytorch_tcn(args.iterations, int8=True)
        results["slm.pytorch.int8"] = _bench_pytorch_slm(args.iterations, int8=True)

    # ONNX FP32
    if args.encoder_onnx and args.encoder_onnx.exists():
        results["tcn.onnx.fp32"] = _bench_onnx(args.encoder_onnx, "encoder", args.iterations)
    if args.slm_onnx and args.slm_onnx.exists():
        results["slm.onnx.fp32"] = _bench_onnx(args.slm_onnx, "slm", args.iterations)

    # ONNX INT8 (assumed to be produced externally via quantize_dynamic)
    if args.encoder_onnx_int8 and args.encoder_onnx_int8.exists():
        results["tcn.onnx.int8"] = _bench_onnx(
            args.encoder_onnx_int8, "encoder", args.iterations
        )
    if args.slm_onnx_int8 and args.slm_onnx_int8.exists():
        results["slm.onnx.int8"] = _bench_onnx(args.slm_onnx_int8, "slm", args.iterations)

    print(_render_markdown(results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
