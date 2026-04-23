"""Compile the I3 DSPy adaptive-prompt program with a teleprompter.

This script:

1. Builds a small labelled train / dev set from the fixtures in
   :mod:`i3.eval.responsiveness_golden`.
2. Configures a DSPy LM backed by Anthropic Claude (requires
   ``ANTHROPIC_API_KEY``).  If no key is set, the script exits cleanly
   with an informative message so CI does not hard-fail.
3. Runs :func:`i3.cloud.dspy_adapter.optimize_program` with either
   ``BootstrapFewShot`` (default) or ``MIPROv2``.
4. Saves the compiled artefact to
   ``checkpoints/dspy/i3_program.json``.

Example::

    python scripts/optimize_dspy_program.py \\
        --teleprompter BootstrapFewShot \\
        --out checkpoints/dspy/i3_program.json

References:
    Khattab et al. 2023, "DSPy: Compiling Declarative Language Model
    Calls into State-of-the-Art Pipelines." arXiv:2310.03714.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("i3.dspy.compile")


def _build_dataset() -> tuple[list[Any], list[Any]]:
    """Return ``(train_set, dev_set)`` of :class:`dspy.Example` objects.

    Examples are derived from :func:`default_golden_set` and paired
    with the canonical adaptation vectors.
    """
    import dspy  # type: ignore[import-not-found]

    from i3.eval.responsiveness_golden import default_golden_set

    golden = default_golden_set()

    # Map the golden-set labels to short natural-language user_state
    # summaries.  This keeps the DSPy prompt structured but compact.
    label_to_state: dict[str, str] = {
        "low_cognitive_warm": "stressed, low cognitive bandwidth, seeks warmth",
        "high_cognitive_technical": "engaged, high cognitive bandwidth, technical",
        "urgent_formal": "task-focused, urgent, formal register",
    }
    # Adaptation vectors summarised as dicts; sent verbatim as strings.
    label_to_adaptation: dict[str, dict[str, float]] = {
        "low_cognitive_warm": {
            "formality": 0.15,
            "verbosity": 0.55,
            "emotionality": 0.85,
            "directness": 0.3,
            "cognitive_load": 0.2,
            "simplification": 0.85,
            "urgency": 0.1,
            "empathy": 0.95,
        },
        "high_cognitive_technical": {
            "formality": 0.55,
            "verbosity": 0.7,
            "emotionality": 0.2,
            "directness": 0.75,
            "cognitive_load": 0.9,
            "simplification": 0.1,
            "urgency": 0.2,
            "empathy": 0.2,
        },
        "urgent_formal": {
            "formality": 0.9,
            "verbosity": 0.3,
            "emotionality": 0.1,
            "directness": 0.95,
            "cognitive_load": 0.6,
            "simplification": 0.3,
            "urgency": 0.95,
            "empathy": 0.2,
        },
    }

    examples: list[Any] = []
    for ex in golden:
        user_state = label_to_state.get(ex.adaptation_label, "")
        av = label_to_adaptation.get(ex.adaptation_label, {})
        dspy_example = dspy.Example(
            user_state=user_state,
            adaptation_vector=av,
            message=ex.prompt,
            expected_tone=ex.expected_tone,
        ).with_inputs("user_state", "adaptation_vector", "message")
        examples.append(dspy_example)

    # Deterministic split: first 8 train, last 4 dev.
    split = max(1, int(len(examples) * 2 / 3))
    return examples[:split], examples[split:]


def _metric(gold: Any, pred: Any, trace: Any = None) -> float:
    """Conditioning-fidelity metric.

    Returns 1.0 if the predicted response's classified tone matches
    ``gold.expected_tone``; 0.0 otherwise.  Built on the lexical
    classifier from :mod:`i3.eval.responsiveness_golden` so it is
    deterministic and cheap.
    """
    from i3.eval.responsiveness_golden import classify_tone

    predicted_text = getattr(pred, "response", None) or str(pred)
    return 1.0 if classify_tone(predicted_text) == gold.expected_tone else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teleprompter",
        choices=["BootstrapFewShot", "MIPROv2"],
        default="BootstrapFewShot",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/dspy/i3_program.json",
        help="Where to save the compiled program JSON.",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    # ---------------- DSPy availability ----------------
    try:
        import dspy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        logger.error(
            "DSPy is not installed (%s).  Install with "
            "`pip install \"dspy-ai>=2.5\"` and re-run.",
            exc,
        )
        return 2

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set.  The teleprompter needs to "
            "call the model during compilation.  Set the key and re-run."
        )
        return 3

    # ---------------- Configure LM ----------------
    try:
        lm = dspy.LM(
            f"anthropic/{args.model}",
            api_key=api_key,
            max_tokens=512,
        )
        dspy.settings.configure(lm=lm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to configure dspy.LM: %s", exc)
        return 4

    # ---------------- Build data ----------------
    train_set, dev_set = _build_dataset()
    logger.info("Dataset: train=%d dev=%d", len(train_set), len(dev_set))

    # ---------------- Compile ----------------
    from i3.cloud.dspy_adapter import I3AdaptivePromptProgram, optimize_program

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    program = I3AdaptivePromptProgram(lm=lm)
    _, info = optimize_program(
        train_set=train_set,
        dev_set=dev_set,
        metric=_metric,
        program=program,
        teleprompter=args.teleprompter,
        save_path=str(out_path),
    )
    logger.info(
        "Saved compiled DSPy program to %s (teleprompter=%s)",
        info.path,
        info.teleprompter,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
