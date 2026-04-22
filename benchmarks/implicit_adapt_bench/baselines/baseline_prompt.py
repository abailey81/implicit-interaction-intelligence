"""Baseline: prompt-based conditioning (the "ChatGPT way").

The archetype is verbalised into a system-prompt-style prefix via
:func:`benchmarks.implicit_adapt_bench.baselines._common.verbalise` and
prepended to the prompt. The architectural conditioning path is held
neutral (zero AdaptationVector, zero user state), so any output difference
vs. :func:`baseline_none` is purely a function of the prompt prefix.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.implicit_adapt_bench.baselines._common import (
    build_harness,
    neutral_adaptation,
    neutral_user_state,
    verbalise,
)
from benchmarks.implicit_adapt_bench.data_schema import (
    BenchmarkRecord,
    BenchmarkSubmission,
)


def run_baseline_prompt(
    records: Sequence[BenchmarkRecord],
    *,
    slm: object | None = None,
    device: str = "cpu",
    max_new_tokens: int = 32,
    seed: int = 42,
) -> list[BenchmarkSubmission]:
    """Generate one submission per record with a verbalised system prefix.

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
    # We add the verbalised prefixes into the tokenizer's training corpus so
    # the vocabulary covers the ``[System: ...]`` tokens that will appear in
    # the prompt at generation time.
    prompts = [r.prompt for r in records]
    prefixes = [
        verbalise(r.target_archetype, r.target_adaptation_vector) for r in records
    ]
    harness = build_harness(
        prompts + prefixes,
        device=device,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )

    submissions: list[BenchmarkSubmission] = []
    neutral_ad = neutral_adaptation(device=device)
    neutral_us = neutral_user_state(device=device)
    for rec, prefix in zip(records, prefixes):
        conditioned_prompt = prefix + rec.prompt
        text, p50, p95 = harness.generate(
            prompt=conditioned_prompt,
            adaptation_vector=neutral_ad,
            user_state=neutral_us,
        )
        submissions.append(
            BenchmarkSubmission(
                record_id=rec.record_id,
                generated_text=text,
                method_name="baseline_prompt",
                runtime_ms_p50=p50,
                runtime_ms_p95=p95,
            )
        )
    return submissions
