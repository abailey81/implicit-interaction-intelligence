"""LLM evaluation harness for the I3 Adaptive SLM.

Public evaluation entry points:

* :func:`i3.eval.perplexity.compute_perplexity` — held-out
  perplexity over a plain-text corpus.
* :func:`i3.eval.conditioning_sensitivity.measure_conditioning_sensitivity`
  — the flagship test for cross-attention conditioning: measures how
  much the next-token distribution shifts when the AdaptationVector
  changes while the prompt is held constant.
* :func:`i3.eval.responsiveness_golden.evaluate_golden_set` — runs a
  golden set of ``(prompt, expected_tone_class)`` pairs and returns
  classifier-agreement statistics.

Each submodule is import-safe without runtime state; CLI wrappers
live under ``scripts/``.
"""

from __future__ import annotations

__all__: list[str] = []
