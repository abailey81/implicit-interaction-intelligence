"""Baseline: full I³ cross-attention conditioning.

The AdaptationVector flows through
:class:`i3.slm.cross_attention.ConditioningProjector` into every cross-
attention layer of :class:`i3.slm.model.AdaptiveSLM`, and a projected
behavioural window is supplied as a ``[1, 64]`` user-state tensor.

This is the reference baseline for the benchmark's target mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.implicit_adapt_bench.baselines._common import (
    adaptation_tensor,
    build_harness,
    extract_user_state,
)
from benchmarks.implicit_adapt_bench.data_schema import (
    BenchmarkRecord,
    BenchmarkSubmission,
)


def run_baseline_cross_attention(
    records: Sequence[BenchmarkRecord],
    *,
    slm: object | None = None,
    device: str = "cpu",
    max_new_tokens: int = 32,
    seed: int = 42,
) -> list[BenchmarkSubmission]:
    """Generate one submission per record using the full conditioning path.

    Args:
        records: Gold benchmark records.
        slm: Ignored; kept for API symmetry with the other baselines.
        device: Torch device string.
        max_new_tokens: Generation budget per record.
        seed: Torch seed for the random-init model weights.

    Returns:
        A list of :class:`BenchmarkSubmission` rows, one per record.
    """
    _ = slm
    prompts = [r.prompt for r in records]
    harness = build_harness(
        prompts, device=device, max_new_tokens=max_new_tokens, seed=seed
    )

    submissions: list[BenchmarkSubmission] = []
    for rec in records:
        av = adaptation_tensor(rec.target_adaptation_vector, device=device)
        us = extract_user_state(rec, device=device)
        text, p50, p95 = harness.generate(
            prompt=rec.prompt, adaptation_vector=av, user_state=us
        )
        submissions.append(
            BenchmarkSubmission(
                record_id=rec.record_id,
                generated_text=text,
                method_name="baseline_cross_attention",
                runtime_ms_p50=p50,
                runtime_ms_p95=p95,
            )
        )
    return submissions
