"""Panel-of-judges ensemble for LLM-as-judge evaluation (Batch G4).

Verga et al. (2024) *"Replacing Judges with Juries"* showed that a small
panel of diverse judges — e.g. Anthropic + OpenAI + Google + Mistral
via the G7 universal provider layer — outperforms any single judge on
most human-preference benchmarks, while also smoothing out judge-
specific biases (self-preference, stylistic affinity, provider outage).

This module wraps a list of :class:`~i3.eval.llm_judge.LLMJudge`
instances and runs them in parallel via :func:`asyncio.gather`, then
aggregates their per-judge :class:`~i3.eval.llm_judge.JudgementResult`s
into a single :class:`EnsembleJudgement`.

Three aggregation modes are supported:

- ``"majority"`` — winner is the category with the most votes; ties go
  to ``"tie"``. Per-rubric scores are the element-wise *median* across
  judges (robust to outliers).
- ``"mean"`` — winner is the one with the higher total mean rubric
  score across judges; per-rubric scores are the element-wise mean.
- ``"median"`` — winner and scores both via element-wise median.

Citations
~~~~~~~~~
* Verga, P. et al. (2024). *Replacing Judges with Juries: Evaluating
  LLM Generations with a Panel of Diverse Models.* arXiv:2404.18796.
* Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS Datasets & Benchmarks.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.eval.judge_calibration import cohens_kappa, fleiss_kappa
from i3.eval.llm_judge import JudgementResult, LLMJudge, Winner

logger = logging.getLogger(__name__)


Aggregation = Literal["majority", "mean", "median"]


_ALLOWED_WINNERS: Final[tuple[Winner, Winner, Winner]] = ("A", "B", "tie")


class EnsembleJudgement(BaseModel):
    """Aggregated result of a multi-judge panel.

    Attributes:
        per_judge: Ordered list of per-judge :class:`JudgementResult`s.
        aggregated_winner: Final winner after aggregation.
        aggregated_scores_a: Aggregated per-rubric scores for A.
        aggregated_scores_b: Aggregated per-rubric scores for B.
        aggregated_confidence: Mean of per-judge confidences.
        inter_judge_kappa: Inter-judge Cohen's (or Fleiss's) kappa on
            the per-judge winners.
        aggregation: The aggregation mode used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_judge: list[JudgementResult] = Field(default_factory=list)
    aggregated_winner: Winner = "tie"
    aggregated_scores_a: dict[str, float] = Field(default_factory=dict)
    aggregated_scores_b: dict[str, float] = Field(default_factory=dict)
    aggregated_confidence: float = 0.0
    inter_judge_kappa: float = 0.0
    aggregation: Aggregation = "majority"


class MultiJudgeEnsemble:
    """Run multiple judges in parallel and aggregate their verdicts.

    Args:
        judges: List of :class:`LLMJudge` instances. Must be non-empty;
            a panel of diverse providers (Anthropic + OpenAI + Google +
            Mistral) is the recommended configuration.
        aggregation: Aggregation mode — ``"majority"`` (default),
            ``"mean"``, or ``"median"``.

    Raises:
        ValueError: If ``judges`` is empty or ``aggregation`` is invalid.
    """

    def __init__(
        self,
        judges: list[LLMJudge],
        aggregation: Aggregation = "majority",
    ) -> None:
        if not judges:
            raise ValueError("MultiJudgeEnsemble requires >= 1 judge")
        if aggregation not in ("majority", "mean", "median"):
            raise ValueError(
                f"aggregation must be one of "
                f"'majority', 'mean', 'median', got {aggregation!r}"
            )
        self._judges: list[LLMJudge] = list(judges)
        self._aggregation: Aggregation = aggregation

    @property
    def n_judges(self) -> int:
        """Return the number of judges in the panel."""
        return len(self._judges)

    @property
    def aggregation(self) -> Aggregation:
        """Return the aggregation mode."""
        return self._aggregation

    async def judge_pair_ensemble(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        target_adaptation: AdaptationVector,
        rubric: list[str],
    ) -> EnsembleJudgement:
        """Run every judge on the same pair and aggregate.

        Args:
            prompt: The user prompt.
            response_a: First response.
            response_b: Second response.
            target_adaptation: Target adaptation vector.
            rubric: Rubric dimensions.

        Returns:
            An :class:`EnsembleJudgement` with per-judge and aggregated
            results.

        Raises:
            ValueError: If every judge raises; the error is propagated.
        """
        if not rubric:
            raise ValueError("rubric must be non-empty")
        tasks = [
            judge.judge_pair(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                target_adaptation=target_adaptation,
                rubric=rubric,
            )
            for judge in self._judges
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        per_judge: list[JudgementResult] = []
        for idx, outcome in enumerate(raw_results):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "Judge %d (%s) raised %s; dropping from panel.",
                    idx,
                    self._judges[idx].provider_name,
                    type(outcome).__name__,
                )
                continue
            per_judge.append(outcome)
        if not per_judge:
            raise ValueError("every judge in the panel failed")

        return self._aggregate(per_judge, rubric)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, per_judge: list[JudgementResult], rubric: list[str]
    ) -> EnsembleJudgement:
        """Aggregate per-judge results according to the configured mode.

        Args:
            per_judge: List of per-judge results (non-empty).
            rubric: Rubric dimensions.

        Returns:
            An :class:`EnsembleJudgement`.
        """
        if self._aggregation == "majority":
            winner = self._majority_winner(per_judge)
            scores_a = self._per_rubric_aggregate(per_judge, "A", "median", rubric)
            scores_b = self._per_rubric_aggregate(per_judge, "B", "median", rubric)
        elif self._aggregation == "mean":
            scores_a = self._per_rubric_aggregate(per_judge, "A", "mean", rubric)
            scores_b = self._per_rubric_aggregate(per_judge, "B", "mean", rubric)
            winner = _winner_from_totals(scores_a, scores_b)
        else:  # median
            scores_a = self._per_rubric_aggregate(per_judge, "A", "median", rubric)
            scores_b = self._per_rubric_aggregate(per_judge, "B", "median", rubric)
            winner = _winner_from_totals(scores_a, scores_b)

        conf = (
            sum(r.confidence for r in per_judge) / len(per_judge)
        )
        kappa = _panel_kappa([r.winner for r in per_judge])
        return EnsembleJudgement(
            per_judge=per_judge,
            aggregated_winner=winner,
            aggregated_scores_a=scores_a,
            aggregated_scores_b=scores_b,
            aggregated_confidence=conf,
            inter_judge_kappa=kappa,
            aggregation=self._aggregation,
        )

    @staticmethod
    def _majority_winner(per_judge: list[JudgementResult]) -> Winner:
        """Return the modal winner; ties (within 1 vote) resolve to ``'tie'``."""
        counts: dict[Winner, int] = dict.fromkeys(_ALLOWED_WINNERS, 0)
        for r in per_judge:
            counts[r.winner] = counts.get(r.winner, 0) + 1
        sorted_counts = sorted(
            counts.items(), key=lambda kv: kv[1], reverse=True
        )
        top_label, top_count = sorted_counts[0]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        if top_count == second_count:
            return "tie"
        return top_label

    @staticmethod
    def _per_rubric_aggregate(
        per_judge: list[JudgementResult],
        slot: Literal["A", "B"],
        mode: Literal["mean", "median"],
        rubric: list[str],
    ) -> dict[str, float]:
        """Aggregate per-judge rubric scores for the A or B slot.

        Args:
            per_judge: Per-judge results.
            slot: ``"A"`` or ``"B"``.
            mode: ``"mean"`` or ``"median"``.
            rubric: Dimension names (defines the output keys).

        Returns:
            Mapping ``dimension -> aggregated float score``.
        """
        out: dict[str, float] = {}
        for dim in rubric:
            values: list[float] = []
            for r in per_judge:
                scores = (
                    r.per_rubric_scores_a if slot == "A"
                    else r.per_rubric_scores_b
                )
                if dim in scores:
                    values.append(float(scores[dim]))
            if not values:
                out[dim] = 0.0
                continue
            out[dim] = (
                float(statistics.fmean(values))
                if mode == "mean"
                else float(statistics.median(values))
            )
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _winner_from_totals(
    scores_a: dict[str, float], scores_b: dict[str, float]
) -> Winner:
    """Decide winner from aggregated total rubric scores.

    Args:
        scores_a: Aggregated per-rubric scores for A.
        scores_b: Aggregated per-rubric scores for B.

    Returns:
        ``"A"``, ``"B"``, or ``"tie"``. Uses a tolerance of 0.5 rubric
        points for tie-break.
    """
    total_a = sum(scores_a.values())
    total_b = sum(scores_b.values())
    if abs(total_a - total_b) < 0.5:
        return "tie"
    return "A" if total_a > total_b else "B"


def _panel_kappa(winners: list[Winner]) -> float:
    """Compute kappa across the panel when the same *item* is shown.

    Since every judge saw the same single item, the straightforward
    pairwise / Fleiss kappa on categorical agreement across judges is
    degenerate (only one item). We instead return the fraction of
    agreeing pairs as a bounded ``[0, 1]`` surrogate for panel
    coherence.

    Args:
        winners: Per-judge winners on the single item.

    Returns:
        Fraction of judge pairs that agreed, in ``[0, 1]``.
    """
    n = len(winners)
    if n < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if winners[i] == winners[j]:
                agree += 1
    return float(agree / total) if total else 1.0


__all__ = [
    "Aggregation",
    "EnsembleJudgement",
    "MultiJudgeEnsemble",
    "cohens_kappa",
    "fleiss_kappa",
]
