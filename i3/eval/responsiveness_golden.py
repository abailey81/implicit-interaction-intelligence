"""Golden-set responsiveness evaluation for the Adaptive SLM.

Runs a fixed collection of ``(prompt, expected_tone_class)`` pairs
through the SLM under each of several canonical AdaptationVectors
(see :func:`i3.eval.conditioning_sensitivity.standard_adaptation_vectors`)
and measures whether the generated tone matches the label that was
intended to be induced.

Tone classification is intentionally simple and deterministic —
lexical keyword matching against three classes:

* ``warm``: empathic, informal, supportive.
* ``technical``: jargon, dense, domain-specific.
* ``formal``: polite, measured, structured.

The classifier is not a learned model; the goal is rigour and
reproducibility for interview-grade reporting, not state-of-the-art
tone classification.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer
    from i3.slm.generate import SLMGenerator
    from i3.eval.responsiveness_golden import (
        default_golden_set,
        evaluate_golden_set,
    )

    generator = SLMGenerator(model, tokenizer)
    stats = evaluate_golden_set(
        generator=generator,
        golden_set=default_golden_set(),
    )
    print(stats)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)


ToneClass = Literal["warm", "technical", "formal", "neutral"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GoldenExample:
    """A single golden-set example."""

    prompt: str
    expected_tone: ToneClass
    adaptation_label: str


@dataclass
class EvaluationResult:
    """Aggregate result of running the golden set."""

    total: int
    correct: int
    per_class_accuracy: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall agreement rate."""
        return self.correct / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Lexical tone classifier
# ---------------------------------------------------------------------------


_WARM_KEYWORDS: tuple[str, ...] = (
    "i understand",
    "that's okay",
    "it's okay",
    "take your time",
    "feel",
    "sorry",
    "together",
    "here for you",
    "happy",
    "love",
    "gentle",
    "kind",
    "thank you for sharing",
)

_TECHNICAL_KEYWORDS: tuple[str, ...] = (
    "algorithm",
    "parameter",
    "gradient",
    "optimisation",
    "optimization",
    "matrix",
    "tensor",
    "complexity",
    "quantisation",
    "quantization",
    "architecture",
    "convolution",
    "transformer",
    "kernel",
    "throughput",
    "latency",
)

_FORMAL_KEYWORDS: tuple[str, ...] = (
    "dear",
    "sincerely",
    "regards",
    "please find",
    "as per",
    "furthermore",
    "moreover",
    "in addition",
    "kindly",
    "shall",
    "hereby",
    "with respect to",
)


def _count_matches(text: str, keywords: tuple[str, ...]) -> int:
    """Return the number of keyword matches in *text* (case-insensitive)."""
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def classify_tone(text: str) -> ToneClass:
    """Return the dominant tone class for *text*.

    Args:
        text: The generated response.

    Returns:
        One of ``"warm"``, ``"technical"``, ``"formal"``, or
        ``"neutral"`` when no class dominates.
    """
    if not text:
        return "neutral"
    scores = {
        "warm": _count_matches(text, _WARM_KEYWORDS),
        "technical": _count_matches(text, _TECHNICAL_KEYWORDS),
        "formal": _count_matches(text, _FORMAL_KEYWORDS),
    }
    best = max(scores.values())
    if best == 0:
        return "neutral"
    # Break ties in favour of the lexicographically earliest class to
    # keep the evaluation deterministic.
    winners = sorted(k for k, v in scores.items() if v == best)
    return winners[0]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


def default_golden_set() -> list[GoldenExample]:
    """Return a small built-in golden set.

    The set is intentionally compact (12 examples) so evaluation runs
    in seconds even on CPU.
    """
    return [
        GoldenExample(
            "I feel really overwhelmed today.",
            "warm",
            "low_cognitive_warm",
        ),
        GoldenExample(
            "Can you explain attention heads?",
            "technical",
            "high_cognitive_technical",
        ),
        GoldenExample(
            "Please send the report summary.",
            "formal",
            "urgent_formal",
        ),
        GoldenExample(
            "I'm nervous about my interview tomorrow.",
            "warm",
            "low_cognitive_warm",
        ),
        GoldenExample(
            "What is the time complexity of self-attention?",
            "technical",
            "high_cognitive_technical",
        ),
        GoldenExample(
            "Kindly confirm receipt.",
            "formal",
            "urgent_formal",
        ),
        GoldenExample(
            "My cat just passed away, I'm really sad.",
            "warm",
            "low_cognitive_warm",
        ),
        GoldenExample(
            "Describe the dilation schedule of a TCN.",
            "technical",
            "high_cognitive_technical",
        ),
        GoldenExample(
            "Please submit the deliverables by Friday.",
            "formal",
            "urgent_formal",
        ),
        GoldenExample(
            "I want to thank you for listening.",
            "warm",
            "low_cognitive_warm",
        ),
        GoldenExample(
            "How does INT4 weight-only quantization work?",
            "technical",
            "high_cognitive_technical",
        ),
        GoldenExample(
            "Furthermore, the matter requires immediate attention.",
            "formal",
            "urgent_formal",
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _generate_with(
    generator: Any,
    prompt: str,
    adaptation_vector: torch.Tensor,
    user_state: torch.Tensor | None,
) -> str:
    """Call the generator with a conditioning vector and return text.

    Accepts any object whose ``generate`` method follows the shape of
    :class:`i3.slm.generate.SLMGenerator.generate`.
    """
    gen_fn: Callable[..., Any] | None = getattr(generator, "generate", None)
    if not callable(gen_fn):
        raise TypeError("generator must expose a generate() method")
    try:
        out = gen_fn(
            prompt,
            adaptation_vector=adaptation_vector,
            user_state=user_state,
            max_new_tokens=64,
        )
    except TypeError:
        # Older signatures without user_state.
        out = gen_fn(prompt, adaptation_vector=adaptation_vector, max_new_tokens=64)
    if isinstance(out, tuple):
        out = out[0]
    return str(out) if out is not None else ""


def evaluate_golden_set(
    generator: Any,
    golden_set: list[GoldenExample] | None = None,
    *,
    adaptation_vectors: dict[str, torch.Tensor] | None = None,
    user_state: torch.Tensor | None = None,
) -> EvaluationResult:
    """Run the golden set and return agreement statistics.

    Args:
        generator: Any object with a ``generate(prompt, ...)`` method
            returning generated text.
        golden_set: Custom examples; defaults to :func:`default_golden_set`.
        adaptation_vectors: Map of adaptation label → ``[1, 8]``
            tensor. Defaults to
            :func:`i3.eval.conditioning_sensitivity.standard_adaptation_vectors`.
        user_state: Optional fixed user-state embedding.

    Returns:
        An :class:`EvaluationResult` with overall and per-class
        accuracy and one entry per evaluated example.
    """
    examples = golden_set or default_golden_set()
    if adaptation_vectors is None:
        from i3.eval.conditioning_sensitivity import standard_adaptation_vectors

        adaptation_vectors = standard_adaptation_vectors()

    per_example: list[dict[str, Any]] = []
    class_totals: dict[str, int] = {}
    class_correct: dict[str, int] = {}
    correct = 0

    for ex in examples:
        vec = adaptation_vectors.get(ex.adaptation_label)
        if vec is None:
            logger.warning(
                "Adaptation label %r not in adaptation_vectors; skipping.",
                ex.adaptation_label,
            )
            continue
        text = _generate_with(generator, ex.prompt, vec, user_state)
        predicted = classify_tone(text)
        hit = predicted == ex.expected_tone
        correct += int(hit)
        class_totals[ex.expected_tone] = class_totals.get(ex.expected_tone, 0) + 1
        class_correct[ex.expected_tone] = class_correct.get(
            ex.expected_tone, 0
        ) + int(hit)
        per_example.append(
            {
                "prompt": ex.prompt,
                "adaptation_label": ex.adaptation_label,
                "expected_tone": ex.expected_tone,
                "predicted_tone": predicted,
                "correct": hit,
                "generated": text,
            }
        )

    per_class_accuracy: dict[str, float] = {}
    for cls, total in class_totals.items():
        per_class_accuracy[cls] = (
            class_correct.get(cls, 0) / total if total else 0.0
        )

    return EvaluationResult(
        total=len(per_example),
        correct=correct,
        per_class_accuracy=per_class_accuracy,
        per_example=per_example,
    )


__all__ = [
    "EvaluationResult",
    "GoldenExample",
    "ToneClass",
    "classify_tone",
    "default_golden_set",
    "evaluate_golden_set",
]


# Silence an unused-import warning when consumers narrow __all__.
_ = re
