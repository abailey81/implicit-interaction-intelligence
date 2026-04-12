"""Edge feasibility reporting for Huawei target devices.

Combines memory and latency profiling into a unified report that
assesses whether each model (TCN encoder, Adaptive SLM) can feasibly
run on Huawei edge hardware (Kirin 9000 phone, Kirin A2 wearable,
Smart Hanhan IoT hub).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from i3.profiling.latency import LatencyBenchmark, LatencyReport
from i3.profiling.memory import MemoryProfiler, MemoryReport

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #


@dataclass
class TargetDevice:
    """Target edge device specification.

    Attributes:
        name: Human-readable device label (e.g. "Kirin 9000 (Phone)").
        memory_mb: Total on-device RAM available for ML workloads (MB).
        tops: Tera-operations per second (INT8 peak throughput).
    """

    name: str
    memory_mb: float
    tops: float  # Tera operations per second

    @property
    def memory_budget_mb(self) -> float:
        """Model budget is 50 % of total device memory.

        The remaining 50 % is reserved for the OS, application runtime,
        sensor buffers, and other concurrent tasks.
        """
        return self.memory_mb * 0.5


@dataclass
class DeviceFeasibility:
    """Feasibility assessment for a specific device.

    Attributes:
        device: The :class:`TargetDevice` being assessed.
        fits_in_memory: True if INT8 model size < device memory budget.
        memory_utilization: Fraction of the memory budget consumed (0--1+).
        estimated_latency_ms: Latency scaled from the benchmark host to
            the target device using a TOPS ratio.
        feasibility_rating: One of ``"feasible"``, ``"tight"``,
            ``"infeasible"``.
        notes: Human-readable notes explaining the rating.
    """

    device: TargetDevice
    fits_in_memory: bool
    memory_utilization: float
    estimated_latency_ms: float
    feasibility_rating: str  # "feasible" | "tight" | "infeasible"
    notes: str


@dataclass
class ProfileReport:
    """Complete edge feasibility report for a single model.

    Attributes:
        model_name: Descriptive label (e.g. "TCN Encoder").
        memory: Memory profiling results.
        latency: FP32 latency benchmark.
        latency_quantized: INT8 latency benchmark.
        device_assessments: Per-device feasibility verdicts.
    """

    model_name: str
    memory: MemoryReport
    latency: LatencyReport
    latency_quantized: LatencyReport
    device_assessments: List[DeviceFeasibility]

    # ---------------------------------------------------------------- #
    # Serialization helpers
    # ---------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary for API responses."""
        return {
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory": {
                "param_count": self.memory.param_count,
                "param_count_trainable": self.memory.param_count_trainable,
                "param_count_millions": self.memory.param_count_millions,
                "fp32_size_mb": round(self.memory.fp32_size_mb, 3),
                "int8_size_mb": round(self.memory.int8_size_mb, 3),
                "compression_ratio": round(self.memory.compression_ratio, 2),
                "peak_inference_mb": round(self.memory.peak_inference_mb, 3),
                "buffer_size_mb": round(self.memory.buffer_size_mb, 3),
            },
            "latency_fp32": {
                "mean_ms": round(self.latency.mean_ms, 3),
                "std_ms": round(self.latency.std_ms, 3),
                "p50_ms": round(self.latency.p50_ms, 3),
                "p95_ms": round(self.latency.p95_ms, 3),
                "p99_ms": round(self.latency.p99_ms, 3),
                "min_ms": round(self.latency.min_ms, 3),
                "max_ms": round(self.latency.max_ms, 3),
                "throughput_hz": round(self.latency.throughput_hz, 1),
                "n_iterations": self.latency.n_iterations,
            },
            "latency_int8": {
                "mean_ms": round(self.latency_quantized.mean_ms, 3),
                "std_ms": round(self.latency_quantized.std_ms, 3),
                "p50_ms": round(self.latency_quantized.p50_ms, 3),
                "p95_ms": round(self.latency_quantized.p95_ms, 3),
                "p99_ms": round(self.latency_quantized.p99_ms, 3),
                "min_ms": round(self.latency_quantized.min_ms, 3),
                "max_ms": round(self.latency_quantized.max_ms, 3),
                "throughput_hz": round(self.latency_quantized.throughput_hz, 1),
                "n_iterations": self.latency_quantized.n_iterations,
            },
            "device_assessments": [
                {
                    "device": a.device.name,
                    "fits_in_memory": a.fits_in_memory,
                    "memory_utilization_pct": round(a.memory_utilization * 100, 1),
                    "estimated_latency_ms": round(a.estimated_latency_ms, 1),
                    "feasibility_rating": a.feasibility_rating,
                    "notes": a.notes,
                }
                for a in self.device_assessments
            ],
        }

    def to_markdown(self) -> str:
        """Generate a Markdown report suitable for presentation slides."""
        lines: list[str] = []
        _a = lines.append

        _a(f"# Edge Feasibility Report: {self.model_name}")
        _a("")
        _a(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
        _a("")

        # --- Memory ---
        _a("## Memory Footprint")
        _a("")
        _a("| Metric | Value |")
        _a("|--------|-------|")
        _a(f"| Parameters | {self.memory.param_count:,} ({self.memory.param_count_millions:.2f} M) |")
        _a(f"| Trainable | {self.memory.param_count_trainable:,} |")
        _a(f"| FP32 size | {self.memory.fp32_size_mb:.2f} MB |")
        _a(f"| INT8 size | {self.memory.int8_size_mb:.2f} MB |")
        _a(f"| Compression ratio | {self.memory.compression_ratio:.1f}x |")
        _a(f"| Peak inference memory | {self.memory.peak_inference_mb:.2f} MB |")
        _a(f"| Buffer overhead | {self.memory.buffer_size_mb:.4f} MB |")
        _a("")

        # --- Latency ---
        _a("## Inference Latency (host machine)")
        _a("")
        _a("| Metric | FP32 | INT8 |")
        _a("|--------|------|------|")
        _a(f"| Mean | {self.latency.mean_ms:.2f} ms | {self.latency_quantized.mean_ms:.2f} ms |")
        _a(f"| Std | {self.latency.std_ms:.2f} ms | {self.latency_quantized.std_ms:.2f} ms |")
        _a(f"| P50 | {self.latency.p50_ms:.2f} ms | {self.latency_quantized.p50_ms:.2f} ms |")
        _a(f"| P95 | {self.latency.p95_ms:.2f} ms | {self.latency_quantized.p95_ms:.2f} ms |")
        _a(f"| P99 | {self.latency.p99_ms:.2f} ms | {self.latency_quantized.p99_ms:.2f} ms |")
        _a(f"| Throughput | {self.latency.throughput_hz:.0f} Hz | {self.latency_quantized.throughput_hz:.0f} Hz |")
        _a("")

        if self.latency_quantized.mean_ms > 0:
            speedup = self.latency.mean_ms / self.latency_quantized.mean_ms
            _a(f"**INT8 speedup: {speedup:.2f}x**")
            _a("")

        # --- Device feasibility ---
        _a("## Device Feasibility")
        _a("")
        _a("| Device | Fits? | Mem. Used | Est. Latency | Rating |")
        _a("|--------|-------|-----------|--------------|--------|")
        for a in self.device_assessments:
            fits_icon = "YES" if a.fits_in_memory else "NO"
            _a(
                f"| {a.device.name} | {fits_icon} | "
                f"{a.memory_utilization * 100:.1f}% | "
                f"{a.estimated_latency_ms:.1f} ms | "
                f"**{a.feasibility_rating.upper()}** |"
            )
        _a("")

        # --- Notes ---
        _a("### Notes")
        _a("")
        for a in self.device_assessments:
            if a.notes:
                _a(f"- **{a.device.name}**: {a.notes}")
        _a("")

        return "\n".join(lines)


# ------------------------------------------------------------------ #
# EdgeProfiler
# ------------------------------------------------------------------ #


class EdgeProfiler:
    """Comprehensive edge deployment profiler.

    Profiles both the TCN encoder and the Adaptive SLM, comparing
    against Huawei target devices to provide evidence for the
    "edge-feasible" claim.

    Default target devices mirror the Huawei HMI project brief:

    * **Kirin 9000 (Phone)** -- 512 MB model budget, 2.0 INT8 TOPS
    * **Kirin A2 (Wearable)** -- 128 MB budget, 0.5 TOPS
    * **Smart Hanhan (IoT)** -- 64 MB budget, 0.1 TOPS

    Example::

        profiler = EdgeProfiler()
        report = profiler.profile_model(
            model=my_tcn, model_name="TCN Encoder",
            input_sample=torch.randn(1, 6, 128),
        )
        print(report.to_markdown())
    """

    DEFAULT_DEVICES: List[TargetDevice] = [
        TargetDevice("Kirin 9000 (Phone)", memory_mb=512, tops=2.0),
        TargetDevice("Kirin A2 (Wearable)", memory_mb=128, tops=0.5),
        TargetDevice("Smart Hanhan (IoT)", memory_mb=64, tops=0.1),
    ]

    # Assumed benchmark-host throughput (INT8 TOPS).  Adjusted during
    # device scaling to estimate on-device latency.
    _HOST_TOPS: float = 1.0

    def __init__(self, config: Any = None) -> None:
        """Initialise the profiler.

        Args:
            config: Optional configuration object.  If it exposes
                ``config.profiling.target_devices`` (list of objects with
                ``name``, ``memory_mb``, ``tops`` attributes) and
                ``config.profiling.benchmark_iterations``, those values
                override the defaults.
        """
        self.memory_profiler = MemoryProfiler()
        self.latency_benchmark = LatencyBenchmark()

        if config and hasattr(config, "profiling"):
            prof_cfg = config.profiling
            self.devices: List[TargetDevice] = [
                TargetDevice(d.name, d.memory_mb, d.tops)
                for d in getattr(prof_cfg, "target_devices", self.DEFAULT_DEVICES)
            ]
            self.n_iterations: int = getattr(prof_cfg, "benchmark_iterations", 100)
        else:
            self.devices = list(self.DEFAULT_DEVICES)
            self.n_iterations = 100

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _generate_notes(
        memory: MemoryReport,
        latency_q: LatencyReport,
        device: TargetDevice,
    ) -> str:
        """Produce human-readable feasibility notes for one device.

        Args:
            memory: The model's memory report.
            latency_q: INT8 latency report from the host machine.
            device: Target device specification.

        Returns:
            A concise note string.
        """
        parts: list[str] = []

        utilization = memory.int8_size_mb / device.memory_budget_mb
        if utilization > 1.0:
            over = memory.int8_size_mb - device.memory_budget_mb
            parts.append(
                f"Model exceeds memory budget by {over:.1f} MB "
                f"({utilization:.0%} utilization). "
                "Consider aggressive pruning or sub-1-bit quantization."
            )
        elif utilization > 0.7:
            parts.append(
                f"Memory utilization is {utilization:.0%} of budget -- "
                "limited headroom for runtime buffers."
            )
        else:
            parts.append(f"Comfortable memory fit ({utilization:.0%} of budget).")

        scale = EdgeProfiler._HOST_TOPS / device.tops
        est_latency = latency_q.mean_ms * scale

        if est_latency > 2000:
            parts.append(
                f"Estimated latency {est_latency:.0f} ms exceeds 2 s -- "
                "real-time interaction infeasible without further optimisation."
            )
        elif est_latency > 500:
            parts.append(
                f"Estimated latency {est_latency:.0f} ms is marginal; "
                "consider TFLite/ONNX runtime or model distillation."
            )
        else:
            parts.append(
                f"Estimated latency {est_latency:.0f} ms supports real-time use."
            )

        return " ".join(parts)

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def profile_model(
        self,
        model: nn.Module,
        model_name: str,
        input_sample: torch.Tensor,
        **model_kwargs: Any,
    ) -> ProfileReport:
        """Generate a complete profiling report for a single model.

        Steps:
            1. Memory profiling (FP32 size, INT8 size, peak inference).
            2. FP32 latency benchmark.
            3. INT8 latency benchmark.
            4. Per-device feasibility assessment.

        Args:
            model: The ``nn.Module`` to profile.
            model_name: Human-readable label (e.g. "TCN Encoder").
            input_sample: Representative input tensor (with batch dim).
            **model_kwargs: Extra kwargs forwarded to ``model.forward()``.

        Returns:
            A :class:`ProfileReport` with memory, latency, and device
            feasibility data.
        """
        logger.info("Profiling model '%s' ...", model_name)

        # 1. Memory
        memory = self.memory_profiler.profile(model, input_sample)

        # 2. FP32 latency
        latency = self.latency_benchmark.benchmark(
            model, input_sample, self.n_iterations, **model_kwargs
        )

        # 3. INT8 latency
        try:
            quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as exc:
            logger.warning(
                "INT8 quantization failed (%s); using FP32 model for "
                "quantized latency estimate.",
                exc,
            )
            quantized = model
        latency_q = self.latency_benchmark.benchmark(
            quantized, input_sample, self.n_iterations, **model_kwargs
        )

        # 4. Device feasibility
        assessments: List[DeviceFeasibility] = []
        for device in self.devices:
            fits = memory.int8_size_mb < device.memory_budget_mb
            utilization = memory.int8_size_mb / device.memory_budget_mb

            scale_factor = self._HOST_TOPS / device.tops
            est_latency = latency_q.mean_ms * scale_factor

            if fits and est_latency < 500:
                rating = "feasible"
            elif fits and est_latency < 2000:
                rating = "tight"
            else:
                rating = "infeasible"

            assessments.append(
                DeviceFeasibility(
                    device=device,
                    fits_in_memory=fits,
                    memory_utilization=utilization,
                    estimated_latency_ms=est_latency,
                    feasibility_rating=rating,
                    notes=self._generate_notes(memory, latency_q, device),
                )
            )

        report = ProfileReport(
            model_name=model_name,
            memory=memory,
            latency=latency,
            latency_quantized=latency_q,
            device_assessments=assessments,
        )
        logger.info(
            "Profile complete for '%s': FP32 %.1f MB, INT8 %.1f MB, "
            "FP32 latency %.2f ms, INT8 latency %.2f ms",
            model_name,
            memory.fp32_size_mb,
            memory.int8_size_mb,
            latency.mean_ms,
            latency_q.mean_ms,
        )
        return report

    def profile_full_system(
        self,
        encoder: nn.Module,
        slm: nn.Module,
        encoder_input: torch.Tensor,
        slm_input: torch.Tensor,
    ) -> Dict[str, Any]:
        """Profile the complete I3 system (encoder + SLM).

        Generates individual :class:`ProfileReport` instances for each
        sub-model and computes combined metrics (total INT8 size and
        end-to-end latency).

        Args:
            encoder: The TCN encoder module.
            slm: The Adaptive SLM module.
            encoder_input: Representative input for the encoder.
            slm_input: Representative input for the SLM.

        Returns:
            Dictionary with keys ``"encoder"`` and ``"slm"``
            (:class:`ProfileReport`), plus ``"combined_int8_mb"`` and
            ``"combined_latency_ms"`` aggregates.
        """
        encoder_report = self.profile_model(encoder, "TCN Encoder", encoder_input)
        slm_report = self.profile_model(slm, "Adaptive SLM", slm_input)

        combined_size = (
            encoder_report.memory.int8_size_mb + slm_report.memory.int8_size_mb
        )
        combined_latency = (
            encoder_report.latency_quantized.mean_ms
            + slm_report.latency_quantized.mean_ms
        )

        logger.info(
            "Full system: combined INT8 %.1f MB, combined latency %.2f ms",
            combined_size,
            combined_latency,
        )

        return {
            "encoder": encoder_report,
            "slm": slm_report,
            "combined_int8_mb": combined_size,
            "combined_latency_ms": combined_latency,
        }
