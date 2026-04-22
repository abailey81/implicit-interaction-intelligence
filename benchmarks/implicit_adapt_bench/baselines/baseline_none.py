"""Baseline: no conditioning.

The SLM runs with a zero AdaptationVector and a zero user-state embedding.
This is the architectural lower bound — the output is expected to be
identical regardless of the behavioural signal.

Intended as the leaderboard floor: any submission that claims to use
implicit behavioural signals should beat this baseline on ``style_match``
and ``cognitive_load_fidelity``.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.implicit_adapt_bench.baselines._common import (
    build_harness,
    neutral_adaptation,
    neutral_user_state,
)
from benchmarks.implicit_adapt_bench.data_schema import (
    BenchmarkRecord,
    BenchmarkSubmission,
)


def run_baseline_none(
    records: Sequence[BenchmarkRecord],
    *,
    slm: object | None = None,
    device: str = "cpu",
    max_new_tokens: int = 32,
    seed: int = 42,
) -> list[BenchmarkSubmission]:
    """Generate one submission per record with zero conditioning.

    Args:
        records: Gold benchmark records.
        slm: Ignored; kept for API symmetry with the other baselines. If
            ``None`` a fresh random-init :class:`AdaptiveSLM` is built.
        device: Torch device string.
        max_new_tokens: Generation budget per record.
        seed: Torch seed for the random-init model weights.

    Returns:
        A list of :class:`BenchmarkSubmission` rows, one per record.
    """
    # ``slm`` is accepted for signature parity with the other baselines but is
    # ignored — the baseline is defined by the absence of conditioning.
    _ = slm
    prompts = [r.prompt for r in records]
    harness = build_harness(
        prompts, device=device, max_new_tokens=max_new_tokens, seed=seed
    )

    submissions: list[BenchmarkSubmission] = []
    neutral_ad = neutral_adaptation(device=device)
    neutral_us = neutral_user_state(device=device)
    for rec in records:
        text, p50, p95 = harness.generate(
            prompt=rec.prompt,
            adaptation_vector=neutral_ad,
            user_state=neutral_us,
        )
        submissions.append(
            BenchmarkSubmission(
                record_id=rec.record_id,
                generated_text=text,
                method_name="baseline_none",
                runtime_ms_p50=p50,
                runtime_ms_p95=p95,
            )
        )
    return submissions
