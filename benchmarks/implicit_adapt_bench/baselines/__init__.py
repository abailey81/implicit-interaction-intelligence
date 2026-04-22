"""Runnable baselines for ImplicitAdaptBench.

Three baselines are provided, mirroring the three conditions of the Batch A
ablation:

* :mod:`baseline_none` — no conditioning; the SLM runs in a neutral state.
* :mod:`baseline_prompt` — the archetype is verbalised as a system prompt
  prefix (the "ChatGPT way"); the architectural conditioning path stays
  neutral.
* :mod:`baseline_cross_attention` — the AdaptationVector flows through the
  :class:`i3.slm.cross_attention.ConditioningProjector` into every cross-
  attention layer.

All baselines accept an already-built :class:`i3.slm.model.AdaptiveSLM`
(random-init weights are fine — the benchmark measures
responsiveness, not quality). See each module for the exact signature.
"""

from __future__ import annotations

__all__ = [
    "run_baseline_none",
    "run_baseline_prompt",
    "run_baseline_cross_attention",
]

from benchmarks.implicit_adapt_bench.baselines.baseline_cross_attention import (
    run_baseline_cross_attention,
)
from benchmarks.implicit_adapt_bench.baselines.baseline_none import (
    run_baseline_none,
)
from benchmarks.implicit_adapt_bench.baselines.baseline_prompt import (
    run_baseline_prompt,
)
