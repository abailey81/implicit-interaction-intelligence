"""Calibration and reliability checks for LLM-as-judge (Batch G4).

Zheng et al. (2023) identified four bias / reliability axes that any
LLM-as-judge pipeline must audit:

- **Position bias** — the judge prefers the response in the A slot (or
  B slot) regardless of content.
- **Length bias** — the judge prefers the longer response.
- **Self-consistency** — repeated queries to the *same* judge on the
  *same* item give the same answer.
- **Inter-judge agreement** — different judges (different providers or
  different models) agree on winners above chance.

This module implements all four as small, pure helpers, with no side
effects and no network calls beyond those issued by the supplied
:class:`~i3.eval.llm_judge.LLMJudge` instances.

Citations
~~~~~~~~~
* Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS Datasets & Benchmarks. — canonical bias
  taxonomy (position, verbosity, self-consistency, majority).
* Cohen, J. (1960). *A coefficient of agreement for nominal scales.*
  Educational and Psychological Measurement, 20(1), 37-46.
* Fleiss, J. L. (1971). *Measuring nominal scale agreement among many
  raters.* Psychological Bulletin, 76(5), 378-382.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from i3.adaptation.types import AdaptationVector
from i3.eval.llm_judge import LLMJudge, Winner

logger = logging.getLogger(__name__)


_ALLOWED_WINNERS: Final[tuple[Winner, Winner, Winner]] = ("A", "B", "tie")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairItem:
    """One pair-judgement item for calibration audits.

    Attributes:
        prompt: User prompt.
        response_a: First response.
        response_b: Second response.
        target_adaptation: Target adaptation vector both responses should
            match.
        context: Free-form context dict for :meth:`LLMJudge.judge_preference`.
    """

    prompt: str
    response_a: str
    response_b: str
    target_adaptation: AdaptationVector
    context: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Cohen's kappa (pair-wise; Fleiss fallback for >=3 judges)
# ---------------------------------------------------------------------------


def cohens_kappa(
    labels_a: Sequence[Winner], labels_b: Sequence[Winner]
) -> float:
    """Compute Cohen's kappa between two sequences of categorical winners.

    Args:
        labels_a: Winners produced by judge A.
        labels_b: Winners produced by judge B. Must be the same length
            as ``labels_a``.

    Returns:
        Cohen's kappa in ``[-1, 1]``. Returns ``1.0`` when both judges
        fully agree, ``0.0`` when agreement equals chance.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"label sequences differ in length: {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        raise ValueError("label sequences must be non-empty")

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    # Expected agreement by chance.
    categories = _ALLOWED_WINNERS
    count_a: dict[Winner, int] = {c: 0 for c in categories}
    count_b: dict[Winner, int] = {c: 0 for c in categories}
    for a in labels_a:
        count_a[a] = count_a.get(a, 0) + 1
    for b in labels_b:
        count_b[b] = count_b.get(b, 0) + 1
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    if p_e >= 1.0:
        # Both judges deterministic on the same category — perfect agreement.
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def fleiss_kappa(matrix: Sequence[Sequence[Winner]]) -> float:
    """Fleiss's kappa for ``>=3`` judges.

    Args:
        matrix: ``[n_judges, n_items]`` sequence of winner labels.

    Returns:
        Fleiss's kappa in ``[-1, 1]``.

    Raises:
        ValueError: If the matrix is not rectangular, has fewer than 2
            judges, or has no items.
    """
    if len(matrix) < 2:
        raise ValueError("need >= 2 judges for kappa")
    n_items = len(matrix[0])
    if n_items == 0:
        raise ValueError("need >= 1 item for kappa")
    for row in matrix:
        if len(row) != n_items:
            raise ValueError("matrix is not rectangular")

    n_judges = len(matrix)
    categories = _ALLOWED_WINNERS
    # n_ij : count of judges that assigned category j to item i.
    per_item_counts: list[dict[Winner, int]] = []
    for i in range(n_items):
        counts: dict[Winner, int] = {c: 0 for c in categories}
        for j in range(n_judges):
            counts[matrix[j][i]] = counts.get(matrix[j][i], 0) + 1
        per_item_counts.append(counts)

    # P_i: extent of agreement on item i.
    p_items: list[float] = []
    for counts in per_item_counts:
        s = sum(c * (c - 1) for c in counts.values())
        p_items.append(s / (n_judges * (n_judges - 1)))
    p_bar = sum(p_items) / n_items

    # p_j: marginal probability of category j.
    totals: dict[Winner, int] = {c: 0 for c in categories}
    for counts in per_item_counts:
        for c in categories:
            totals[c] += counts[c]
    p_marginals = {
        c: totals[c] / (n_items * n_judges) for c in categories
    }
    p_e_bar = sum(p**2 for p in p_marginals.values())
    if p_e_bar >= 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return float((p_bar - p_e_bar) / (1.0 - p_e_bar))


# ---------------------------------------------------------------------------
# JudgeCalibrator
# ---------------------------------------------------------------------------


class JudgeCalibrator:
    """Runs the four Zheng-2023 bias audits against an
    :class:`LLMJudge` (or list of them).

    The calibrator issues concurrent preference queries via
    :meth:`LLMJudge.judge_preference` so that it works with any
    :class:`~i3.cloud.providers.base.CloudProvider`-backed judge without
    requiring a separate rubric.
    """

    async def inter_judge_agreement(
        self,
        judges: list[LLMJudge],
        items: Sequence[PairItem],
    ) -> float:
        """Compute Cohen's (or Fleiss's) kappa across multiple judges.

        Args:
            judges: Two or more :class:`LLMJudge` instances.
            items: Pair-judgement items.

        Returns:
            Kappa in ``[-1, 1]``. For exactly 2 judges returns Cohen's
            kappa; for 3+ returns Fleiss's kappa.

        Raises:
            ValueError: If fewer than 2 judges or no items are supplied.
        """
        if len(judges) < 2:
            raise ValueError("inter_judge_agreement needs >= 2 judges")
        if not items:
            raise ValueError("items must be non-empty")
        per_judge_labels: list[list[Winner]] = []
        for judge in judges:
            labels = await _collect_preferences(judge, items)
            per_judge_labels.append(labels)
        if len(judges) == 2:
            return cohens_kappa(per_judge_labels[0], per_judge_labels[1])
        return fleiss_kappa(per_judge_labels)

    async def self_consistency(
        self,
        judge: LLMJudge,
        item: PairItem,
        n_samples: int = 5,
    ) -> float:
        """Fraction of ``n_samples`` repeats that match the modal answer.

        Use a judge with ``temperature > 0`` — this measure is trivially
        ``1.0`` at temperature 0.

        Args:
            judge: The judge to audit.
            item: A single pair item.
            n_samples: Number of repeated preference queries.

        Returns:
            A float in ``[1 / n_samples, 1.0]``.

        Raises:
            ValueError: If ``n_samples < 2``.
        """
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        tasks = [
            judge.judge_preference(
                prompt=item.prompt,
                response_a=item.response_a,
                response_b=item.response_b,
                context=item.context or {},
            )
            for _ in range(n_samples)
        ]
        answers: list[Winner] = await asyncio.gather(*tasks)
        counts: dict[Winner, int] = {c: 0 for c in _ALLOWED_WINNERS}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1
        modal = max(counts.values())
        return float(modal / n_samples)

    async def position_bias_test(
        self,
        judge: LLMJudge,
        items: Sequence[PairItem],
    ) -> float:
        """Flip rate when A and B are swapped in the prompt.

        A perfectly position-unbiased judge flips its winner *every* time
        the order is swapped (A becomes B and vice versa), yielding a
        flip rate of 1.0. A position-biased judge reports the same slot
        regardless, giving 0.0. Ties are treated as stable.

        Args:
            judge: The judge to audit.
            items: Pair-judgement items.

        Returns:
            Flip rate in ``[0, 1]``.

        Raises:
            ValueError: If ``items`` is empty.
        """
        if not items:
            raise ValueError("items must be non-empty")
        forward: list[Winner] = await _collect_preferences(judge, items)
        swapped_items = [
            PairItem(
                prompt=it.prompt,
                response_a=it.response_b,
                response_b=it.response_a,
                target_adaptation=it.target_adaptation,
                context=it.context,
            )
            for it in items
        ]
        backward: list[Winner] = await _collect_preferences(judge, swapped_items)
        flips = 0
        counted = 0
        for fwd, bwd in zip(forward, backward):
            if fwd == "tie" or bwd == "tie":
                continue
            counted += 1
            # Expected flip: A in forward -> B in backward, and vice versa.
            expected = "B" if fwd == "A" else "A"
            if bwd == expected:
                flips += 1
        if counted == 0:
            return 0.0
        return float(flips / counted)

    async def length_bias_test(
        self,
        judge: LLMJudge,
        items: Sequence[PairItem],
    ) -> float:
        """Pearson correlation between winning-response length and preference.

        For each non-tie item, computes ``len(winner) - len(loser)``. The
        function then Pearson-correlates that signed length delta against
        a constant ``+1`` — effectively the *mean* normalised length
        advantage of winners. A value near zero indicates no bias; a
        positive value means the judge prefers the longer response; a
        negative value, the shorter.

        Args:
            judge: The judge to audit.
            items: Pair-judgement items.

        Returns:
            A float in ``[-1, 1]``.

        Raises:
            ValueError: If ``items`` is empty.
        """
        if not items:
            raise ValueError("items must be non-empty")
        winners: list[Winner] = await _collect_preferences(judge, items)
        deltas: list[float] = []
        for it, w in zip(items, winners):
            if w == "tie":
                continue
            len_a = float(len(it.response_a))
            len_b = float(len(it.response_b))
            if w == "A":
                deltas.append(len_a - len_b)
            else:
                deltas.append(len_b - len_a)
        if not deltas:
            return 0.0
        mean = sum(deltas) / len(deltas)
        total_abs = sum(abs(d) for d in deltas)
        if total_abs <= 0.0:
            return 0.0
        return float(max(-1.0, min(1.0, mean / (total_abs / len(deltas)))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_preferences(
    judge: LLMJudge, items: Iterable[PairItem]
) -> list[Winner]:
    """Run ``judge_preference`` over ``items`` concurrently.

    Args:
        judge: The judge.
        items: Pair-judgement items.

    Returns:
        List of winners aligned with ``items``.
    """
    items_list = list(items)
    tasks = [
        judge.judge_preference(
            prompt=it.prompt,
            response_a=it.response_a,
            response_b=it.response_b,
            context=it.context or {},
        )
        for it in items_list
    ]
    return await asyncio.gather(*tasks)


__all__ = [
    "JudgeCalibrator",
    "PairItem",
    "cohens_kappa",
    "fleiss_kappa",
]
