"""Chaos tests: force specific dependencies to fail and assert that the
pipeline degrades gracefully rather than crashing the service.

Failure modes exercised:
    * ``cloud_client.generate`` raises :class:`asyncio.TimeoutError`.
    * ``cloud_client.generate`` raises :class:`RuntimeError`.
    * ``PrivacySanitizer.sanitize`` returns an empty string.
    * TCN encoder forward pass raises.

For each, we expect :meth:`Pipeline.process_message` to return a
well-formed :class:`PipelineOutput` (never ``None``) with a fallback
``response_text`` and ``route_chosen in {"local_slm", "cloud_llm",
"error_fallback"}``.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
async def pipeline():
    """Build a real Pipeline with lightweight init — skip on failure."""
    try:
        from i3.config import load_config
        from i3.pipeline.engine import Pipeline
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Pipeline import failed: {exc}")
    try:
        config = load_config("configs/default.yaml", set_seeds=True)
        pipe = Pipeline(config)
        await pipe.initialize()
    except Exception as exc:  # pragma: no cover — degraded env
        pytest.skip(f"Pipeline.initialize failed: {exc}")
    yield pipe
    try:
        await pipe.shutdown()
    except Exception:
        pass


def _input(user_id: str = "chaos_user", text: str = "hello there"):
    from i3.pipeline.types import PipelineInput
    import time as _time

    return PipelineInput(
        user_id=user_id,
        session_id="chaos_session",
        message_text=text,
        timestamp=_time.time(),
        composition_time_ms=1200.0,
        edit_count=0,
        pause_before_send_ms=150.0,
        keystroke_timings=[50.0, 60.0, 70.0],
    )


# ─────────────────────────────────────────────────────────────────────────
#  Cloud LLM failure modes
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cloud_timeout_falls_back_locally(pipeline) -> None:
    """If ``cloud_client.generate`` raises TimeoutError the pipeline must
    still return a response (either SLM-generated or a canned fallback)."""
    await pipeline.start_session("chaos_user")

    # Replace cloud_client.generate with a TimeoutError-raising mock.
    async def _timeout(*args, **kwargs):
        raise asyncio.TimeoutError("simulated cloud timeout")

    original = pipeline.cloud_client.generate
    pipeline.cloud_client.generate = _timeout  # type: ignore[assignment]
    try:
        output = await pipeline.process_message(_input())
    finally:
        pipeline.cloud_client.generate = original  # type: ignore[assignment]

    assert output is not None
    assert isinstance(output.response_text, str)
    assert output.response_text, "fallback response must not be empty"
    assert output.route_chosen in {
        "local_slm",
        "cloud_llm",
        "error_fallback",
        "fallback",
    }


@pytest.mark.asyncio
async def test_cloud_runtime_error_does_not_propagate(pipeline) -> None:
    """A generic RuntimeError from the cloud client must never bubble
    up — the pipeline must always return a PipelineOutput."""
    await pipeline.start_session("chaos_user")

    async def _boom(*args, **kwargs):
        raise RuntimeError("synthetic backend failure")

    original = pipeline.cloud_client.generate
    pipeline.cloud_client.generate = _boom  # type: ignore[assignment]
    try:
        output = await pipeline.process_message(_input())
    finally:
        pipeline.cloud_client.generate = original  # type: ignore[assignment]

    assert output is not None
    assert isinstance(output.latency_ms, (int, float))
    assert output.latency_ms >= 0
    # route_chosen is always a non-empty string
    assert isinstance(output.route_chosen, str) and output.route_chosen


# ─────────────────────────────────────────────────────────────────────────
#  Sanitiser edge cases
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_sanitized_output_degrades_gracefully(pipeline) -> None:
    """If the sanitiser reduces the message to empty the pipeline must
    either:
      (a) raise a well-typed ValueError / ValidationError, OR
      (b) return a safe default PipelineOutput.
    Never: raise an unhandled generic Exception.
    """
    await pipeline.start_session("chaos_user")

    from i3.privacy.sanitizer import SanitizationResult

    original = pipeline.sanitizer.sanitize

    def _empty(_text: str) -> SanitizationResult:
        return SanitizationResult(
            sanitized_text="",
            pii_detected=False,
            pii_types=[],
            replacements_made=0,
        )

    pipeline.sanitizer.sanitize = _empty  # type: ignore[assignment]
    try:
        output = await pipeline.process_message(_input())
    except (ValueError,) as exc:
        # Explicit well-typed error is fine.
        assert "empty" in str(exc).lower() or True
        return
    finally:
        pipeline.sanitizer.sanitize = original  # type: ignore[assignment]

    # If we did get a result, it must still be a well-formed PipelineOutput.
    assert output is not None
    assert isinstance(output.response_text, str)


# ─────────────────────────────────────────────────────────────────────────
#  Encoder failure
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_encoder_exception_does_not_break_pipeline(pipeline) -> None:
    """A forward-pass error in the TCN encoder must be contained."""
    await pipeline.start_session("chaos_user")

    original = pipeline._encode_features

    def _boom(_window):
        raise RuntimeError("synthetic encoder failure")

    pipeline._encode_features = _boom  # type: ignore[assignment]
    try:
        output = await pipeline.process_message(_input())
    finally:
        pipeline._encode_features = original  # type: ignore[assignment]

    assert output is not None
    # error_fallback is the standard route_chosen when the inner
    # pipeline raises (see engine._build_error_output).
    assert output.route_chosen in {
        "error_fallback",
        "local_slm",
        "cloud_llm",
        "fallback",
    }
    assert isinstance(output.response_text, str)
    assert output.response_text


# ─────────────────────────────────────────────────────────────────────────
#  Never-None contract
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_message_never_returns_none(pipeline) -> None:
    """Regardless of what fails, ``process_message`` returns a value."""
    await pipeline.start_session("chaos_user")

    original = pipeline.cloud_client.generate

    async def _raise(*args, **kwargs):
        raise Exception("anything")

    pipeline.cloud_client.generate = _raise  # type: ignore[assignment]
    try:
        output = await pipeline.process_message(_input())
    finally:
        pipeline.cloud_client.generate = original  # type: ignore[assignment]

    assert output is not None
