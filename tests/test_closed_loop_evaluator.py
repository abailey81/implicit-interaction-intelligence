"""Unit tests for :class:`ClosedLoopEvaluator`.

These tests replace the real I3 :class:`Pipeline` with a minimal
mock that implements the duck-typed contract used by the evaluator
(``async process_message(PipelineInput) -> PipelineOutput`` and
``async start_session(user_id) -> str``). This avoids loading the SLM,
the TCN encoder, or the cloud client, and keeps the test suite fast and
hermetic.

Coverage:

* Constructor argument validation.
* A 2-persona, 1-session, 3-message run produces a well-formed
  :class:`ClosedLoopResult`.
* Persona confusion matrix dimensions match ``persona_order``.
* Convergence speed is ``None`` for a mock that never converges.
* Router bias correctly identifies an elevated local-routing rate under
  the motor-impaired persona when the mock router uses the
  accessibility threshold.
* Aggregate metrics aggregate correctly with known per-persona values.
* Bootstrap CIs are finite and bracket the sample mean.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.eval.simulation import (
    ClosedLoopEvaluator,
    ClosedLoopResult,
    FRESH_USER,
    HCIPersona,
    MOTOR_IMPAIRED_USER,
)
from i3.eval.simulation.closed_loop import (
    MessageRecord,
    _adaptation_vector_l2,
)
from i3.pipeline.types import PipelineInput, PipelineOutput


# ---------------------------------------------------------------------------
# Mock pipeline helpers
# ---------------------------------------------------------------------------


class _MockPipeline:
    """Minimal pipeline stub implementing the evaluator's contract.

    Args:
        mode: One of ``"ground_truth"``, ``"constant"``, or
            ``"access_router"``.

            * ``ground_truth``: Echo the persona's true adaptation
              vector back as the inferred vector, decorated with tiny
              per-persona noise. Routing is always ``"local_slm"``.
            * ``constant``: Return a fixed default vector regardless of
              input (never converges for most personas).
            * ``access_router``: Return the ground-truth vector and
              choose ``"local_slm"`` iff ``accessibility > 0.6``, else
              ``"cloud_llm"``. Used for the router-bias test.
        persona_lookup: Map ``persona_name -> HCIPersona`` so the mock
            can read the ground-truth vector from the input's
            ``user_id``.
    """

    def __init__(
        self,
        mode: str,
        persona_lookup: dict[str, HCIPersona],
    ) -> None:
        self.mode = mode
        self.persona_lookup = persona_lookup
        self.calls: list[PipelineInput] = []

    async def start_session(self, user_id: str) -> str:
        return f"mock:{user_id}"

    async def process_message(self, inp: PipelineInput) -> PipelineOutput:
        self.calls.append(inp)
        # user_id format: "persona:<name>:session:<idx>"
        parts = inp.user_id.split(":")
        persona_name = parts[1] if len(parts) >= 2 else ""
        persona = self.persona_lookup.get(persona_name)

        if self.mode == "constant":
            adapt = AdaptationVector.default()
        elif self.mode in ("ground_truth", "access_router") and persona is not None:
            adapt = persona.expected_adaptation
        else:
            adapt = AdaptationVector.default()

        if self.mode == "access_router":
            route = (
                "local_slm" if adapt.accessibility > 0.6 else "cloud_llm"
            )
        else:
            route = "local_slm"

        return PipelineOutput(
            response_text="mock response",
            route_chosen=route,
            latency_ms=1.0,
            user_state_embedding_2d=(0.1, 0.2),
            adaptation=adapt.to_dict(),
            engagement_score=0.5,
            deviation_from_baseline=0.1,
            routing_confidence={"local_slm": 0.6, "cloud_llm": 0.4},
            messages_in_session=len(self.calls),
            baseline_established=True,
        )


def _persona_lookup(*personas: HCIPersona) -> dict[str, HCIPersona]:
    return {p.name: p for p in personas}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_evaluator_rejects_zero_sessions() -> None:
    with pytest.raises(ValueError):
        ClosedLoopEvaluator(
            pipeline=_MockPipeline("ground_truth", {}),
            personas=[FRESH_USER],
            n_sessions_per_persona=0,
        )


def test_evaluator_rejects_zero_messages() -> None:
    with pytest.raises(ValueError):
        ClosedLoopEvaluator(
            pipeline=_MockPipeline("ground_truth", {}),
            personas=[FRESH_USER],
            n_messages_per_session=0,
        )


def test_evaluator_rejects_empty_persona_list() -> None:
    with pytest.raises(ValueError):
        ClosedLoopEvaluator(
            pipeline=_MockPipeline("ground_truth", {}),
            personas=[],
        )


def test_evaluator_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError):
        ClosedLoopEvaluator(
            pipeline=_MockPipeline("ground_truth", {}),
            personas=[FRESH_USER],
            adapt_converged_threshold=0.0,
        )


# ---------------------------------------------------------------------------
# End-to-end minimal run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_runs_small_configuration() -> None:
    """Evaluator completes a 2-persona, 1-session, 3-message run."""
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("ground_truth", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=1,
        n_messages_per_session=3,
        seed=42,
    )
    result = await evaluator.run()
    assert isinstance(result, ClosedLoopResult)
    assert result.persona_order == [p.name for p in personas]
    assert result.n_sessions_per_persona == 1
    assert result.n_messages_per_session == 3
    # One session per persona × 3 messages × 2 personas == 6 records.
    assert len(result.per_message_records) == 6
    for rec in result.per_message_records:
        assert isinstance(rec, MessageRecord)
        assert len(rec.inferred) == 7
        assert len(rec.ground_truth) == 7


@pytest.mark.asyncio
async def test_persona_confusion_matrix_is_square_and_sized() -> None:
    """Confusion matrix has the expected shape."""
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("ground_truth", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=2,
        seed=7,
    )
    result = await evaluator.run()
    n = len(personas)
    assert len(result.persona_confusion_matrix) == n
    for row in result.persona_confusion_matrix:
        assert len(row) == n
    # With a ground-truth pipeline, every final message should map to
    # the true persona -> all mass on the diagonal.
    for i in range(n):
        for j in range(n):
            expected = 2 if i == j else 0
            assert result.persona_confusion_matrix[i][j] == expected


# ---------------------------------------------------------------------------
# Convergence semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convergence_speed_is_none_when_pipeline_never_converges() -> None:
    """A constant-output pipeline should not converge for non-default personas."""
    personas = [MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("constant", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=4,
        adapt_converged_threshold=0.01,  # very strict
        seed=1,
    )
    result = await evaluator.run()
    assert result.convergence_speeds[MOTOR_IMPAIRED_USER.name] is None


@pytest.mark.asyncio
async def test_convergence_speed_is_zero_when_pipeline_returns_ground_truth() -> None:
    """A ground-truth pipeline converges on the first message."""
    personas = [FRESH_USER]
    pipeline = _MockPipeline("ground_truth", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=4,
        adapt_converged_threshold=0.3,
        seed=1,
    )
    result = await evaluator.run()
    cs = result.convergence_speeds[FRESH_USER.name]
    assert cs is not None
    assert cs == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Router-bias check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_bias_detects_elevated_local_rate_for_motor_impaired() -> None:
    """When the mock router routes locally iff accessibility > 0.6, the
    motor-impaired persona (accessibility=0.75) should route entirely
    locally while the fresh user (accessibility=0.05) routes to cloud.
    """
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("access_router", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=3,
        seed=2024,
    )
    result = await evaluator.run()
    bias = result.router_bias
    assert bias["local_rate_fresh_user"] == pytest.approx(0.0)
    assert bias["local_rate_motor_impaired_user"] == pytest.approx(1.0)
    # Only motor_impaired in the accessibility set is present here;
    # evaluator should still compute the delta correctly.
    assert "accessibility_vs_baseline_delta" in bias
    assert bias["accessibility_vs_baseline_delta"] > 0.5


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_recovery_rate_matches_per_persona_mean() -> None:
    """Aggregate recovery rate should equal the mean of the per-session flags."""
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("ground_truth", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=3,
        n_messages_per_session=2,
        seed=9,
    )
    result = await evaluator.run()
    # All sessions recover -> 1.0 per persona -> 1.0 aggregate.
    assert result.aggregate_recovery_rate == pytest.approx(1.0)
    for name in (FRESH_USER.name, MOTOR_IMPAIRED_USER.name):
        assert result.per_persona_recovery_rate[name] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_aggregate_adaptation_error_is_nonnegative() -> None:
    """Aggregate L2 error must be non-negative and finite."""
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("constant", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=3,
        seed=11,
    )
    result = await evaluator.run()
    assert result.aggregate_adaptation_error >= 0.0
    assert math.isfinite(result.aggregate_adaptation_error)
    lo, hi = result.aggregate_adaptation_error_ci
    assert math.isfinite(lo) and math.isfinite(hi)
    # CI should bracket the sample mean (modulo bootstrap jitter).
    assert lo <= result.aggregate_adaptation_error <= hi or (
        # Allow a tiny tolerance on degenerate narrow CIs.
        abs(hi - result.aggregate_adaptation_error) < 1e-6
    )


# ---------------------------------------------------------------------------
# Bootstrap CI coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_ci_bounds_are_valid() -> None:
    """CI lower bound must be <= upper bound for every metric."""
    personas = [FRESH_USER, MOTOR_IMPAIRED_USER]
    pipeline = _MockPipeline("ground_truth", _persona_lookup(*personas))
    evaluator = ClosedLoopEvaluator(
        pipeline=pipeline,
        personas=personas,
        n_sessions_per_persona=2,
        n_messages_per_session=3,
        seed=31,
    )
    result = await evaluator.run()
    for name, (lo, hi) in result.per_persona_recovery_ci.items():
        assert lo <= hi, f"recovery CI inverted for {name}: [{lo}, {hi}]"
    for name, (lo, hi) in result.per_persona_adaptation_error_ci.items():
        assert lo <= hi, f"error CI inverted for {name}: [{lo}, {hi}]"
    lo, hi = result.aggregate_recovery_rate_ci
    assert lo <= hi
    lo, hi = result.aggregate_adaptation_error_ci
    assert lo <= hi


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_adaptation_vector_l2_is_zero_for_identical_vectors() -> None:
    v = AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector(0.5, 0.5, 0.5, 0.5),
        emotional_tone=0.5,
        accessibility=0.5,
    )
    assert _adaptation_vector_l2(v, v) == pytest.approx(0.0)


def test_adaptation_vector_l2_is_positive_for_different_vectors() -> None:
    a = AdaptationVector.default()
    b = AdaptationVector(
        cognitive_load=1.0,
        style_mirror=StyleVector(1.0, 1.0, 1.0, 1.0),
        emotional_tone=1.0,
        accessibility=1.0,
    )
    d = _adaptation_vector_l2(a, b)
    assert d > 0.0
