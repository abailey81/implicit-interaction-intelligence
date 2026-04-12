"""Edge Feasibility Profiler for Implicit Interaction Intelligence (I3).

Measures model sizes, inference latencies, and memory footprints,
then compares against target Huawei devices to demonstrate edge
deployment feasibility.
"""

from i3.profiling.memory import MemoryProfiler, MemoryReport
from i3.profiling.latency import LatencyBenchmark, LatencyReport
from i3.profiling.report import EdgeProfiler, ProfileReport, TargetDevice, DeviceFeasibility

__all__ = [
    "EdgeProfiler",
    "MemoryProfiler",
    "LatencyBenchmark",
    "ProfileReport",
    "MemoryReport",
    "LatencyReport",
    "TargetDevice",
    "DeviceFeasibility",
]
