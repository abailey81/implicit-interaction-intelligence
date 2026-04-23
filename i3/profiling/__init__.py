"""Edge Feasibility Profiler for Implicit Interaction Intelligence (I3).

Measures model sizes, inference latencies, and memory footprints,
then compares against target Huawei devices to demonstrate edge
deployment feasibility.
"""

from i3.profiling.latency import LatencyBenchmark, LatencyReport
from i3.profiling.memory import MemoryProfiler, MemoryReport
from i3.profiling.report import DeviceFeasibility, EdgeProfiler, ProfileReport, TargetDevice

__all__ = [
    "DeviceFeasibility",
    "EdgeProfiler",
    "LatencyBenchmark",
    "LatencyReport",
    "MemoryProfiler",
    "MemoryReport",
    "ProfileReport",
    "TargetDevice",
]
