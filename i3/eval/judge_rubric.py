"""Canonical rubrics for the Batch G4 LLM-as-judge evaluation harness.

This module exposes four rubric constants plus a factory that formats a
rubric as a short prompt block grounded in a concrete
:class:`~i3.adaptation.types.AdaptationVector`. Each rubric is a list of
*dimension names* that the judge must score in ``[0, 5]``. The rubric
vocabulary is deliberately small so that the judge's chain-of-thought is
bounded, reducing rubric-prompt sensitivity (Liu et al., 2023, G-Eval).

Design rationale
~~~~~~~~~~~~~~~~
Batch A measures *numerical* responsiveness (KL divergence between next-
token distributions) and Batch C measures rule-based style metrics. Neither
captures a human's "which is better-adapted?" judgement. These rubrics
project the 8-dim :class:`AdaptationVector` onto the four axes a human
rater would use when scoring a generated response:

- formality / verbosity / emotionality / directness (style mirror);
- pace / sentence length / vocabulary simplicity / structure clarity
  (cognitive load);
- short-sentence ratio / jargon-free / yes-no suitability / explicit
  structure (accessibility);
- all of the above, for ``FULL_ADAPTATION_RUBRIC``.

Citations
~~~~~~~~~
* Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS Datasets & Benchmarks.
* Liu, Y. et al. (2023). *G-Eval: NLG Evaluation Using GPT-4 with
  Better Human Alignment.* EMNLP 2023.
* Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI
  Feedback.* arXiv:2212.08073 (rubric-style principle grading).
"""

from __future__ import annotations

from typing import Final

from i3.adaptation.types import AdaptationVector

# ---------------------------------------------------------------------------
# Rubric constants
# ---------------------------------------------------------------------------


STYLE_MATCH_RUBRIC: Final[list[str]] = [
    "formality match",
    "verbosity match",
    "emotionality match",
    "directness match",
]
"""Style-mirroring rubric — four axes from :class:`StyleVector`."""


COGNITIVE_LOAD_RUBRIC: Final[list[str]] = [
    "pace appropriateness",
    "sentence length fit",
    "vocabulary simplicity",
    "structure clarity",
]
"""Cognitive-load rubric — response complexity tracks ``cognitive_load``."""


ACCESSIBILITY_RUBRIC: Final[list[str]] = [
    "short-sentence ratio",
    "jargon-free",
    "yes-no suitability",
    "explicit structure",
]
"""Accessibility rubric — active only when ``accessibility`` is high."""


FULL_ADAPTATION_RUBRIC: Final[list[str]] = (
    STYLE_MATCH_RUBRIC + COGNITIVE_LOAD_RUBRIC + ACCESSIBILITY_RUBRIC
)
"""Union of all three rubrics (12 dimensions)."""


_ALL_RUBRICS: Final[dict[str, list[str]]] = {
    "style": STYLE_MATCH_RUBRIC,
    "cognitive_load": COGNITIVE_LOAD_RUBRIC,
    "accessibility": ACCESSIBILITY_RUBRIC,
    "full": FULL_ADAPTATION_RUBRIC,
}


def get_rubric(name: str) -> list[str]:
    """Return the rubric list for ``name``.

    Args:
        name: One of ``"style"``, ``"cognitive_load"``, ``"accessibility"``,
            ``"full"``.

    Returns:
        The list of dimension names.

    Raises:
        KeyError: If ``name`` is not a known rubric.
    """
    key = name.lower().strip()
    if key not in _ALL_RUBRICS:
        raise KeyError(
            f"Unknown rubric {name!r}; known: {sorted(_ALL_RUBRICS)}"
        )
    return list(_ALL_RUBRICS[key])


# ---------------------------------------------------------------------------
# Verbalisation of the target adaptation
# ---------------------------------------------------------------------------


def _bucket(value: float, low: str, mid: str, high: str) -> str:
    """Map ``value`` in ``[0, 1]`` onto one of three adjectives.

    Args:
        value: Scalar in ``[0, 1]``.
        low: Label for the low bucket (``value < 0.4``).
        mid: Label for the mid bucket (``0.4 <= value <= 0.6``).
        high: Label for the high bucket (``value > 0.6``).

    Returns:
        One of ``low`` / ``mid`` / ``high``.
    """
    if value < 0.4:
        return low
    if value > 0.6:
        return high
    return mid


def describe_target(target: AdaptationVector) -> str:
    """Render a target :class:`AdaptationVector` as a short natural-language spec.

    Args:
        target: The adaptation vector the system was asked to match.

    Returns:
        A one-paragraph description of the target style.
    """
    style = target.style_mirror
    formality = _bucket(style.formality, "casual", "neutral", "formal")
    verbosity = _bucket(style.verbosity, "concise", "balanced", "elaborate")
    emotionality = _bucket(style.emotionality, "reserved", "neutral", "expressive")
    directness = _bucket(style.directness, "indirect", "balanced", "direct")
    load = _bucket(
        target.cognitive_load, "simple", "everyday", "technical"
    )
    tone = _bucket(
        target.emotional_tone, "warm and supportive", "neutral", "objective"
    )
    access = (
        "accessibility mode ON — short sentences, no idioms, binary confirm"
        if target.accessibility > 0.7
        else "standard accessibility"
    )
    return (
        f"Target style: {formality}, {verbosity}, {emotionality}, {directness}. "
        f"Cognitive-load register: {load}. "
        f"Emotional tone: {tone}. "
        f"{access}."
    )


# ---------------------------------------------------------------------------
# Rubric prompt factory
# ---------------------------------------------------------------------------


def make_rubric_prompt(
    rubric_name: str, target_adaptation: AdaptationVector
) -> str:
    """Build the rubric-description block inserted into the judge prompt.

    The block is a short, deterministic paragraph that (a) names the
    rubric dimensions and (b) restates the target adaptation in natural
    language so the judge can score dimension-by-dimension.

    Args:
        rubric_name: One of ``"style"``, ``"cognitive_load"``,
            ``"accessibility"``, or ``"full"``.
        target_adaptation: The target :class:`AdaptationVector`.

    Returns:
        A non-empty string suitable for embedding in the judge prompt.

    Raises:
        KeyError: If ``rubric_name`` is unknown.
    """
    rubric = get_rubric(rubric_name)
    target_desc = describe_target(target_adaptation)
    bullets = "\n".join(f"- {dim}" for dim in rubric)
    return (
        f"RUBRIC ({rubric_name}):\n"
        f"{bullets}\n\n"
        f"TARGET ADAPTATION:\n"
        f"{target_desc}\n\n"
        "Score each rubric dimension 0-5 (0 = no match, 5 = perfect match). "
        "Higher scores mean the response better matches the target adaptation "
        "on that dimension."
    )


__all__ = [
    "ACCESSIBILITY_RUBRIC",
    "COGNITIVE_LOAD_RUBRIC",
    "FULL_ADAPTATION_RUBRIC",
    "STYLE_MATCH_RUBRIC",
    "describe_target",
    "get_rubric",
    "make_rubric_prompt",
]
