"""Inference latency benchmarking for PyTorch models.

Provides precise wall-clock measurements with warmup, percentile
statistics, throughput estimation, and FP32-vs-INT8 comparison.
"""

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LatencyReport:
    """Latency benchmark results.

    All timing values are in **milliseconds**.

    Attributes:
        mean_ms: Arithmetic mean latency.
        std_ms: Sample standard deviation.
        p50_ms: Median (50th percentile).
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        min_ms: Fastest observed run.
        max_ms: Slowest observed run.
        n_iterations: Number of timed iterations (excludes warmup).
        throughput_hz: Estimated inferences per second (1000 / mean_ms).
    """

    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n_iterations: int
    throughput_hz: float

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"mean {self.mean_ms:.2f} ms | "
            f"p50 {self.p50_ms:.2f} | p95 {self.p95_ms:.2f} | p99 {self.p99_ms:.2f} | "
            f"{self.throughput_hz:.0f} Hz  (n={self.n_iterations})"
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the value at the given percentile from a pre-sorted list.

    Uses nearest-rank method, clamped to valid indices.

    Args:
        sorted_values: Ascending-sorted list of floats.
        pct: Percentile in [0, 1] (e.g. 0.95 for P95).

    Returns:
        The value at the requested percentile.
    """
    idx = int(len(sorted_values) * pct)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


class LatencyBenchmark:
    """Benchmarks inference latency for PyTorch models.

    Example::

        bench = LatencyBenchmark()
        report = bench.benchmark(model, torch.randn(1, 6, 128), n_iterations=200)
        print(report.summary())
    """

    @staticmethod
    def benchmark(
        model: nn.Module,
        input_sample: torch.Tensor,
        n_iterations: int = 100,
        warmup: int = 10,
        **model_kwargs: Any,
    ) -> LatencyReport:
        """Run a latency benchmark.

        The model is set to ``eval()`` mode. A configurable number of
        warmup iterations run first (un-timed) to stabilize caches and
        JIT compilation.  Then ``n_iterations`` timed forward passes are
        recorded using :func:`time.perf_counter`.

        Args:
            model: The ``nn.Module`` to benchmark.
            input_sample: A representative input tensor (with batch dim).
            n_iterations: Number of timed forward passes.
            warmup: Number of warmup passes (not timed).
            **model_kwargs: Additional keyword arguments forwarded to
                ``model.forward()``.

        Returns:
            A :class:`LatencyReport` with full percentile statistics.
        """
        model.eval()

        # --- warmup ---
        with torch.no_grad():
            for _ in range(warmup):
                model(input_sample, **model_kwargs)

        # --- timed runs ---
        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.perf_counter()
                model(input_sample, **model_kwargs)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)  # seconds -> ms

        latencies.sort()
        n = len(latencies)
        mean = statistics.mean(latencies)

        report = LatencyReport(
            mean_ms=mean,
            std_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            p50_ms=_percentile(latencies, 0.50),
            p95_ms=_percentile(latencies, 0.95),
            p99_ms=_percentile(latencies, 0.99),
            min_ms=latencies[0],
            max_ms=latencies[-1],
            n_iterations=n,
            throughput_hz=1000.0 / mean if mean > 0 else 0.0,
        )
        logger.info("Latency benchmark: %s", report.summary())
        return report

    @staticmethod
    def compare_fp32_vs_int8(
        model: nn.Module,
        input_sample: torch.Tensor,
        n_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Compare latency between FP32 and dynamically-quantized INT8.

        Args:
            model: The original FP32 ``nn.Module``.
            input_sample: A representative input tensor (with batch dim).
            n_iterations: Number of timed iterations per variant.

        Returns:
            Dictionary with keys ``"fp32"`` (:class:`LatencyReport`),
            ``"int8"`` (:class:`LatencyReport`), and ``"speedup"`` (float).
        """
        fp32_report = LatencyBenchmark.benchmark(
            model, input_sample, n_iterations
        )

        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        int8_report = LatencyBenchmark.benchmark(
            quantized, input_sample, n_iterations
        )

        speedup = (
            fp32_report.mean_ms / int8_report.mean_ms
            if int8_report.mean_ms > 0
            else 0.0
        )
        logger.info(
            "FP32 vs INT8: %.2f ms vs %.2f ms  (%.2fx speedup)",
            fp32_report.mean_ms,
            int8_report.mean_ms,
            speedup,
        )

        return {
            "fp32": fp32_report,
            "int8": int8_report,
            "speedup": speedup,
        }
