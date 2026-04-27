"""Iter 66 — Per-cascade-arm OTel span coverage.

Verifies that the iter-52 cascade arms emit OpenTelemetry spans with
stable names (so dashboards / alerts can pin them).  Tests run against
the in-memory OTel exporter so they don't need a live collector and
finish in <0.1 s.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def memory_exporter():
    """Wire OpenTelemetry to an in-memory span exporter for the test."""
    pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Replace the global tracer provider with our test provider.
    prev = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        yield exporter
    finally:
        trace._TRACER_PROVIDER = prev  # type: ignore[attr-defined]


def test_span_helper_emits_named_span(memory_exporter):
    from i3.observability.tracing import span as i3_span
    with i3_span("cascade.test_emission", arm="A"):
        pass
    spans = memory_exporter.get_finished_spans()
    assert any(s.name == "cascade.test_emission" for s in spans), \
        f"span not emitted; got {[s.name for s in spans]}"


def test_span_attributes_preserved(memory_exporter):
    from i3.observability.tracing import span as i3_span
    with i3_span("cascade.attr_test", arm="B", request_chars=42):
        pass
    spans = memory_exporter.get_finished_spans()
    s = next(s for s in spans if s.name == "cascade.attr_test")
    assert s.attributes.get("arm") == "B"
    assert s.attributes.get("request_chars") == 42


def test_intent_cascade_emits_arm_b_span(memory_exporter):
    """When the intent cascade fires, it must emit a stable
    `cascade.arm_b.qwen_intent` span the dashboard can correlate."""
    from i3.pipeline.engine import Pipeline

    class _StubResult:
        action = "set_timer"
        params = {"duration_seconds": 60}
        valid_action = True
        valid_slots = True
        confidence = 0.95

        def to_dict(self):
            return {"action": self.action, "params": self.params,
                    "valid_action": True, "valid_slots": True,
                    "confidence": self.confidence}

    class _StubParser:
        def parse(self, t):
            return _StubResult()

    p = Pipeline.__new__(Pipeline)
    p._stated_facts = {}
    p._intent_parser_qwen = _StubParser()
    out = p._maybe_handle_intent_command(
        message="set timer for 1 minute",
        user_id="u", session_id="s",
    )
    assert out is not None
    spans = memory_exporter.get_finished_spans()
    names = {s.name for s in spans}
    # Expect the wrapper span + the parse span.  qwen_load span is
    # only emitted on cold-load — won't fire here because the parser
    # is pre-stubbed.
    assert "cascade.arm_b.qwen_intent" in names, \
        f"missing arm_b wrapper span; got {names}"
    assert "cascade.arm_b.qwen_parse" in names, \
        f"missing arm_b parse span; got {names}"


def test_otel_no_op_path_works_without_sdk():
    """Even without OTel installed, the i3.span() helper must not crash."""
    from i3.observability.tracing import span as i3_span
    # No memory_exporter fixture — exercise the no-op path.
    with i3_span("cascade.no_op_test", k="v"):
        pass
