"""Tests for :mod:`i3.huawei.agentic_core_runtime`.

Covers runtime lifecycle, intent dispatch, privacy guard, telemetry
shape, fallback-pipeline behaviour, and response correlation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from i3.huawei.agentic_core_runtime import (
    HMAFAgentRuntime,
    HMAFIntent,
    HMAFResponse,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _TelemetryCapture:
    """Collects telemetry events for assertion."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


@pytest.fixture()
def capture() -> _TelemetryCapture:
    """Fresh telemetry capture per test."""
    return _TelemetryCapture()


@pytest.fixture()
def runtime(capture: _TelemetryCapture) -> HMAFAgentRuntime:
    """Runtime with an in-memory pipeline and captured telemetry."""
    from i3.huawei.hmaf_adapter import HMAFAgentAdapter
    from i3.huawei.agentic_core_runtime import _MockPipeline

    adapter = HMAFAgentAdapter(
        pipeline=_MockPipeline(),
        telemetry_sink=capture,
    )
    return HMAFAgentRuntime(adapter=adapter)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_then_stop_is_idempotent(runtime: HMAFAgentRuntime) -> None:
    """Calling start/stop multiple times must not raise."""
    await runtime.start()
    await runtime.start()  # idempotent
    await runtime.stop()
    await runtime.stop()  # idempotent


@pytest.mark.asyncio
async def test_receive_intent_before_start_fails(
    runtime: HMAFAgentRuntime,
) -> None:
    """``receive_intent`` must reject calls before ``start``."""
    with pytest.raises(RuntimeError):
        await runtime.receive_intent(
            HMAFIntent(name="get_user_adaptation", source_device="phone")
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_returns_response(
    runtime: HMAFAgentRuntime,
) -> None:
    """A known intent yields an ``ok=True`` response with matching id."""
    await runtime.start()
    try:
        intent = HMAFIntent(
            name="get_user_adaptation",
            source_device="phone",
        )
        resp = await runtime.plan_and_execute(intent)
        assert isinstance(resp, HMAFResponse)
        assert resp.ok is True
        assert resp.correlation_id == intent.correlation_id
        assert "adaptation" in resp.payload
        assert resp.latency_ms >= 0.0
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_unknown_intent_yields_failure_response(
    runtime: HMAFAgentRuntime,
) -> None:
    """Unknown intents return ok=False with ``unknown_intent``."""
    await runtime.start()
    try:
        intent = HMAFIntent(name="does_not_exist", source_device="phone")
        resp = await runtime.plan_and_execute(intent)
        assert resp.ok is False
        assert resp.terminal_action == "unknown_intent"
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_bus_dispatch_records_response(
    runtime: HMAFAgentRuntime,
) -> None:
    """An intent pushed onto the bus is consumed and its response stored."""
    await runtime.start()
    try:
        intent = HMAFIntent(
            name="summarise_session",
            parameters={"turns": 3, "avg_engagement": 0.5},
            source_device="phone",
        )
        await runtime.receive_intent(intent)
        # Wait until the consumer processes the intent.
        for _ in range(50):
            resp = await runtime.last_response(intent.correlation_id)
            if resp is not None:
                break
            await asyncio.sleep(0.01)
        assert resp is not None
        assert resp.ok is True
        assert resp.payload["turns"] == 3
    finally:
        await runtime.stop()


# ---------------------------------------------------------------------------
# Privacy guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_guard_refuses_forbidden_intent(
    runtime: HMAFAgentRuntime, capture: _TelemetryCapture
) -> None:
    """Forbidden intents are refused and logged via telemetry."""
    await runtime.start()
    try:
        intent = HMAFIntent(name="dump_raw_diary", source_device="phone")
        resp = await runtime.plan_and_execute(intent)
        assert resp.ok is False
        assert resp.terminal_action == "deny_request"
        assert any(
            ev.get("event") == "intent.refused" for ev in capture.events
        )
    finally:
        await runtime.stop()


# ---------------------------------------------------------------------------
# Telemetry shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telemetry_events_are_text_free(
    runtime: HMAFAgentRuntime, capture: _TelemetryCapture
) -> None:
    """No telemetry event may carry a forbidden text-like key."""
    await runtime.start()
    try:
        intent = HMAFIntent(
            name="translate",
            parameters={"target_language": "fr", "length_in": 10},
            source_device="glasses",
        )
        await runtime.plan_and_execute(intent)
    finally:
        await runtime.stop()

    forbidden_keys = {"text", "prompt", "response", "body", "content", "raw"}
    for event in capture.events:
        overlap = forbidden_keys & {k.lower() for k in event}
        assert overlap == set(), f"Telemetry event leaked: {event}"


@pytest.mark.asyncio
async def test_telemetry_emits_intent_lifecycle(
    runtime: HMAFAgentRuntime, capture: _TelemetryCapture
) -> None:
    """Each successful intent emits complete lifecycle events."""
    await runtime.start()
    try:
        await runtime.plan_and_execute(
            HMAFIntent(name="explain_adaptation", source_device="phone")
        )
    finally:
        await runtime.stop()
    event_types = {e.get("event") for e in capture.events}
    assert "runtime.start" in event_types
    assert "intent.complete" in event_types
    assert "runtime.stop" in event_types


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------


def test_hmaf_intent_rejects_empty_name() -> None:
    """Pydantic must reject an empty intent name."""
    with pytest.raises(ValueError):
        HMAFIntent(name="", source_device="phone")


def test_hmaf_intent_autogenerates_correlation_id() -> None:
    """Correlation id is auto-generated when omitted."""
    intent = HMAFIntent(name="translate", source_device="phone")
    assert len(intent.correlation_id) >= 8
