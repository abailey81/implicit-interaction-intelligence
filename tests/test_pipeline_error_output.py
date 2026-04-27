"""Iter 110 — Pipeline._build_error_output / _fallback_response_for_error tests.

The error-output path is the safety net at the bottom of process_message.
A bug here means a single unhandled exception kills the chat instead of
degrading gracefully.
"""
from __future__ import annotations

import pytest

from i3.pipeline.engine import Pipeline
from i3.pipeline.types import PipelineOutput


def _bare_pipeline_with_config():
    """Construct a minimal Pipeline shape for error-output test."""
    from types import SimpleNamespace
    p = Pipeline.__new__(Pipeline)
    # The error path reads pipeline.config.router.arms — fake it.
    p.config = SimpleNamespace(router=SimpleNamespace(arms=("local_slm", "cloud_llm")))
    return p


def test_fallback_response_is_non_empty():
    text = Pipeline._fallback_response_for_error()
    assert isinstance(text, str)
    assert text.strip()


def test_fallback_response_is_user_friendly():
    """Fallback must NOT leak the exception class / stacktrace / Python
    internals into user-visible text."""
    text = Pipeline._fallback_response_for_error().lower()
    forbidden = ["traceback", "exception", "valueerror",
                 "typeerror", "runtimeerror", "file \"", "line "]
    for phrase in forbidden:
        assert phrase not in text, f"fallback leaks {phrase!r}: {text!r}"


def test_build_error_output_returns_pipeline_output():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(100.0, ValueError("inner"))
    assert isinstance(out, PipelineOutput)


def test_build_error_output_response_is_safe():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, RuntimeError("crash details"))
    assert "crash details" not in out.response_text
    assert "RuntimeError" not in out.response_text


def test_build_error_output_route_label():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, Exception())
    assert out.route_chosen == "error_fallback"


def test_build_error_output_engagement_zero():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, Exception())
    # Failed turns get zero engagement so the bandit doesn't reward them.
    assert out.engagement_score == 0.0


def test_build_error_output_messages_in_session_zero():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, Exception())
    assert out.messages_in_session == 0


def test_build_error_output_baseline_not_established():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, Exception())
    assert out.baseline_established is False


def test_build_error_output_routing_confidence_sums_in_unit():
    p = _bare_pipeline_with_config()
    out = p._build_error_output(50.0, Exception())
    assert isinstance(out.routing_confidence, dict)
    s = sum(out.routing_confidence.values())
    assert 0.0 <= s <= 1.0 + 1e-6
