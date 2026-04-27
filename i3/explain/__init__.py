"""Plain-English reasoning-trace composer for the I3 demo.

This sub-package contains pure-Python (no torch, no LLM call) helpers
that compose a per-turn explanation of *why* the system produced its
visible response.  See :mod:`i3.explain.reasoning_trace` for the public
entry point.
"""

from i3.explain.reasoning_trace import build_reasoning_trace

__all__ = ["build_reasoning_trace"]
