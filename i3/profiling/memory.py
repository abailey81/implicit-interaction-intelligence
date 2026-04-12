"""Memory profiling for PyTorch models targeting edge deployment.

Measures parameter counts, on-disk sizes (FP32 and INT8-quantized),
peak inference memory via tracemalloc, and buffer overhead. These
metrics feed into the EdgeProfiler's device-feasibility assessment.
"""

import logging
import os
import tempfile
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MemoryReport:
    """Memory usage report for a model.

    Attributes:
        param_count: Total number of parameters (trainable + frozen).
        param_count_trainable: Number of parameters with requires_grad=True.
        fp32_size_mb: On-disk state_dict size in MB (float32 weights).
        int8_size_mb: On-disk size after dynamic INT8 quantization in MB.
        peak_inference_mb: Peak Python-heap memory during inference (MB).
        buffer_size_mb: Estimated size of model buffers in MB (FP32).
    """

    param_count: int
    param_count_trainable: int
    fp32_size_mb: float
    int8_size_mb: float
    peak_inference_mb: float
    buffer_size_mb: float

    @property
    def compression_ratio(self) -> float:
        """Ratio of FP32 size to INT8 size (higher is better)."""
        if self.int8_size_mb > 0:
            return self.fp32_size_mb / self.int8_size_mb
        return 1.0

    @property
    def param_count_millions(self) -> float:
        """Parameter count expressed in millions for readability."""
        return self.param_count / 1_000_000

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.param_count_millions:.2f}M params | "
            f"FP32 {self.fp32_size_mb:.1f} MB | "
            f"INT8 {self.int8_size_mb:.1f} MB | "
            f"peak infer {self.peak_inference_mb:.1f} MB | "
            f"compression {self.compression_ratio:.1f}x"
        )


class MemoryProfiler:
    """Profiles memory usage of PyTorch models.

    Provides static helpers for individual measurements and a ``profile``
    method that returns a complete :class:`MemoryReport`.

    Example::

        profiler = MemoryProfiler()
        report = profiler.profile(model, torch.randn(1, 6, 128))
        print(report.summary())
    """

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count total, trainable, and buffer elements.

        Args:
            model: Any ``nn.Module``.

        Returns:
            Dictionary with keys ``"total"``, ``"trainable"``, ``"buffers"``.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in model.buffers())
        return {"total": total, "trainable": trainable, "buffers": buffers}

    @staticmethod
    def measure_model_size(model: nn.Module) -> float:
        """Measure actual model size on disk in MB.

        Saves the ``state_dict`` to a temporary file, reads its byte size,
        then deletes the file.

        Args:
            model: Any ``nn.Module``.

        Returns:
            Size of the serialized state dict in megabytes.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(model.state_dict(), tmp_path)
            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        finally:
            os.unlink(tmp_path)
        return size_mb

    @staticmethod
    def measure_quantized_size(model: nn.Module) -> float:
        """Dynamically quantize ``nn.Linear`` layers to INT8 and measure size.

        Uses ``torch.quantization.quantize_dynamic`` which requires no
        calibration data -- suitable for quick feasibility checks.

        Args:
            model: Any ``nn.Module`` containing ``nn.Linear`` layers.

        Returns:
            On-disk size of the INT8-quantized model in megabytes.
        """
        try:
            quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return MemoryProfiler.measure_model_size(quantized)
        except Exception as exc:
            logger.warning("INT8 quantization failed (%s); returning FP32 size.", exc)
            return MemoryProfiler.measure_model_size(model)

    @staticmethod
    def measure_inference_memory(
        model: nn.Module,
        input_sample: torch.Tensor,
        n_runs: int = 10,
    ) -> float:
        """Measure peak Python-heap memory during inference.

        Uses :mod:`tracemalloc` to capture the high-water mark across
        ``n_runs`` forward passes.  This captures Python-level allocations;
        CUDA memory is *not* tracked here.

        Args:
            model: Any ``nn.Module``.
            input_sample: A representative input tensor (batch dim included).
            n_runs: Number of forward passes to execute.

        Returns:
            Peak traced memory in megabytes.
        """
        model.eval()

        # Ensure tracemalloc is stopped before we start a fresh session.
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        tracemalloc.start()
        try:
            with torch.no_grad():
                for _ in range(n_runs):
                    model(input_sample)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        return peak / (1024 * 1024)  # bytes -> MB

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def profile(self, model: nn.Module, input_sample: torch.Tensor) -> MemoryReport:
        """Generate a complete :class:`MemoryReport` for *model*.

        Args:
            model: The ``nn.Module`` to profile.
            input_sample: A representative input tensor (with batch dim).

        Returns:
            A populated :class:`MemoryReport`.
        """
        params = self.count_parameters(model)
        fp32_size = self.measure_model_size(model)
        int8_size = self.measure_quantized_size(model)
        peak_mem = self.measure_inference_memory(model, input_sample)
        # Each buffer element is stored as FP32 (4 bytes) by default.
        buffer_size = params["buffers"] * 4 / (1024 * 1024)

        report = MemoryReport(
            param_count=params["total"],
            param_count_trainable=params["trainable"],
            fp32_size_mb=fp32_size,
            int8_size_mb=int8_size,
            peak_inference_mb=peak_mem,
            buffer_size_mb=buffer_size,
        )
        logger.info("Memory profile: %s", report.summary())
        return report
