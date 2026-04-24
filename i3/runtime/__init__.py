"""Runtime helpers for I3.

Device selection, mixed-precision, and CUDA fast-path toggles live here so
that training scripts, inference hot paths, and the FastAPI bootstrap all
pick them up from a single source of truth.
"""

from i3.runtime.device import (
    autocast_context,
    enable_cuda_optimizations,
    pick_device,
)

__all__ = [
    "autocast_context",
    "enable_cuda_optimizations",
    "pick_device",
]
