"""Tests for :mod:`i3.eval.judge_calibration`.

Covers:
    * Cohen's kappa = 1.0 on identical outputs.
    * Cohen's kappa on completely disagreeing outputs (-ish).
    * Fleiss's kappa on 3+ judges with identical outputs is 1.0.
    * Position-bias test returns a high flip rate for an unbiased judge.
    * Position-bias test returns a low flip rate for a pinned judge.
    * Length-bias correlation is positive when the judge favours the
      longer response.
    * Length-bias correlation is negative when the judge favours the
      shorter response.
    * All calibrator metrics produce finite numbers on non-trivial
      inputs.
    * Inter-judge agreement raises on mismatched / empty input.
    * kappa helpers raise on empty / non-rectangular input.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    TokenUsage,
)
from i3.eval.judge_calibration import (
    JudgeCalibrator,
    PairItem,
    cohens_kappa,
    fleiss_kappa,
)
from i3.eval.llm_judge import LLMJudge


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class _MockProvider:
    def __init__(
        self,
        *,
        reply: str | None = None,
        reply_fn: Any = None,
        name: str = "mock",
    ) -> None:
        self.provider_name = name
        self._reply = reply
        self._reply_fn = reply_fn

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        text = self._reply_fn(request) if self._reply_fn else self._reply
        assert text is not None
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model="mock-model",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1,
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass


def _reply(winner: str, conf: float = 0.8) -> str:
    return json.dumps(
        {
            "winner": winner,
            "rationale": "stub",
            "confidence": conf,
        }
    )


def _target() -> AdaptationVector:
    return AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector(
            formality=0.5, verbosity=0.5, emotionality=0.5, directness=0.5
        ),
        emotional_tone=0.5,
        accessibility=0.0,
    )


# ---------------------------------------------------------------------------
# 1. kappa identities
# ---------------------------------------------------------------------------


def test_cohens_kappa_perfect_agreement() -> None:
    labels = ["A", "B", "tie", "A", "A"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_disagreement() -> None:
    a = ["A"] * 5
    b = ["B"] * 5
    # Both judges deterministic on different categories: agreement = 0,
    # expected = 0 -> kappa = 0.
    assert cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cohens_kappa_mismatched_length_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa(["A", "B"], ["A"])


def test_cohens_kappa_empty_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_fleiss_kappa_perfect_agreement() -> None:
    matrix = [["A", "B", "tie", "A"]] * 3
    assert fleiss_kappa(matrix) == pytest.approx(1.0)


def test_fleiss_kappa_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        fleiss_kappa([["A", "B"], ["A"]])
    with pytest.raises(ValueError):
        fleiss_kappa([["A"]])  # only one judge
    with pytest.raises(ValueError):
        fleiss_kappa([[], []])  # no items


# ---------------------------------------------------------------------------
# 2. Position-bias test
# ---------------------------------------------------------------------------


def _content_based_reply_fn(marker: str) -> Any:
    def _fn(request: CompletionRequest) -> str:
        user_text = request.messages[0].content
        a_block = user_text.split("BEGIN_RESPONSE_A")[1].split("END_RESPONSE_A")[0]
        b_block = user_text.split("BEGIN_RESPONSE_B")[1].split("END_RESPONSE_B")[0]
        if marker in a_block and marker not in b_block:
            return _reply("A")
        if marker in b_block and marker not in a_block:
            return _reply("B")
        return _reply("tie")

    return _fn


def test_position_bias_unbiased_judge() -> None:
    provider = _MockProvider(reply_fn=_content_based_reply_fn("WINNING"))
    judge = LLMJudge(provider=provider)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a="A text with WINNING marker" if i % 2 == 0 else "plain A text",
            response_b="plain B text" if i % 2 == 0 else "B text with WINNING marker",
            target_adaptation=_target(),
        )
        for i in range(6)
    ]
    rate = asyncio.run(calibrator.position_bias_test(judge, items))
    assert rate == pytest.approx(1.0)


def test_position_bias_pinned_judge() -> None:
    provider = _MockProvider(reply=_reply("A"))
    judge = LLMJudge(provider=provider)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a=f"a{i}",
            response_b=f"b{i}",
            target_adaptation=_target(),
        )
        for i in range(4)
    ]
    rate = asyncio.run(calibrator.position_bias_test(judge, items))
    assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Length bias
# ---------------------------------------------------------------------------


def _length_biased_reply_fn(prefer_longer: bool) -> Any:
    def _fn(request: CompletionRequest) -> str:
        user_text = request.messages[0].content
        a_block = user_text.split("BEGIN_RESPONSE_A")[1].split("END_RESPONSE_A")[0]
        b_block = user_text.split("BEGIN_RESPONSE_B")[1].split("END_RESPONSE_B")[0]
        if len(a_block) == len(b_block):
            return _reply("tie")
        if prefer_longer:
            return _reply("A" if len(a_block) > len(b_block) else "B")
        return _reply("A" if len(a_block) < len(b_block) else "B")

    return _fn


def test_length_bias_positive_for_longer_preferer() -> None:
    provider = _MockProvider(reply_fn=_length_biased_reply_fn(prefer_longer=True))
    judge = LLMJudge(provider=provider)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a="short." if i % 2 == 0 else "a much much much longer response " * 4,
            response_b="a much much much longer response " * 4 if i % 2 == 0 else "short.",
            target_adaptation=_target(),
        )
        for i in range(6)
    ]
    corr = asyncio.run(calibrator.length_bias_test(judge, items))
    assert corr > 0.5
    assert math.isfinite(corr)


def test_length_bias_negative_for_shorter_preferer() -> None:
    provider = _MockProvider(reply_fn=_length_biased_reply_fn(prefer_longer=False))
    judge = LLMJudge(provider=provider)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a="short." if i % 2 == 0 else "a much much much longer response " * 4,
            response_b="a much much much longer response " * 4 if i % 2 == 0 else "short.",
            target_adaptation=_target(),
        )
        for i in range(6)
    ]
    corr = asyncio.run(calibrator.length_bias_test(judge, items))
    assert corr < -0.5
    assert math.isfinite(corr)


# ---------------------------------------------------------------------------
# 4. Inter-judge agreement
# ---------------------------------------------------------------------------


def test_inter_judge_agreement_identical_judges() -> None:
    provider_a = _MockProvider(reply=_reply("A"), name="j1")
    provider_b = _MockProvider(reply=_reply("A"), name="j2")
    judges = [LLMJudge(provider=provider_a), LLMJudge(provider=provider_b)]
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a=f"a{i}",
            response_b=f"b{i}",
            target_adaptation=_target(),
        )
        for i in range(3)
    ]
    kappa = asyncio.run(calibrator.inter_judge_agreement(judges, items))
    assert kappa == pytest.approx(1.0)


def test_inter_judge_agreement_three_judges_fleiss() -> None:
    judges = [
        LLMJudge(provider=_MockProvider(reply=_reply("A"), name=f"j{i}"))
        for i in range(3)
    ]
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a=f"a{i}",
            response_b=f"b{i}",
            target_adaptation=_target(),
        )
        for i in range(3)
    ]
    kappa = asyncio.run(calibrator.inter_judge_agreement(judges, items))
    assert kappa == pytest.approx(1.0)


def test_inter_judge_agreement_rejects_single_judge() -> None:
    judge = LLMJudge(provider=_MockProvider(reply=_reply("A")))
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt="P",
            response_a="a",
            response_b="b",
            target_adaptation=_target(),
        )
    ]
    with pytest.raises(ValueError):
        asyncio.run(calibrator.inter_judge_agreement([judge], items))


def test_inter_judge_agreement_rejects_empty_items() -> None:
    judges = [
        LLMJudge(provider=_MockProvider(reply=_reply("A"), name=f"j{i}"))
        for i in range(2)
    ]
    calibrator = JudgeCalibrator()
    with pytest.raises(ValueError):
        asyncio.run(calibrator.inter_judge_agreement(judges, []))


# ---------------------------------------------------------------------------
# 5. Finite numerical outputs
# ---------------------------------------------------------------------------


def test_all_calibrator_outputs_are_finite() -> None:
    provider = _MockProvider(reply_fn=_content_based_reply_fn("W"))
    judge = LLMJudge(provider=provider, temperature=0.3)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a=f"A text {i} with W marker" if i % 2 == 0 else f"A plain {i}",
            response_b=f"B plain {i}" if i % 2 == 0 else f"B text {i} with W marker",
            target_adaptation=_target(),
        )
        for i in range(4)
    ]
    pos = asyncio.run(calibrator.position_bias_test(judge, items))
    lng = asyncio.run(calibrator.length_bias_test(judge, items))
    consistent = asyncio.run(
        calibrator.self_consistency(judge, items[0], n_samples=3)
    )
    assert math.isfinite(pos)
    assert math.isfinite(lng)
    assert math.isfinite(consistent)
    assert 0.0 <= pos <= 1.0
    assert -1.0 <= lng <= 1.0
    assert 0.0 <= consistent <= 1.0


# ---------------------------------------------------------------------------
# 6. Position bias with ties does not crash
# ---------------------------------------------------------------------------


def test_position_bias_all_ties_returns_zero() -> None:
    provider = _MockProvider(reply=_reply("tie"))
    judge = LLMJudge(provider=provider)
    calibrator = JudgeCalibrator()
    items = [
        PairItem(
            prompt=f"P{i}",
            response_a=f"a{i}",
            response_b=f"b{i}",
            target_adaptation=_target(),
        )
        for i in range(3)
    ]
    rate = asyncio.run(calibrator.position_bias_test(judge, items))
    assert rate == pytest.approx(0.0)
