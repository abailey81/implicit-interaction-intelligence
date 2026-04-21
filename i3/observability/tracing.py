"""OpenTelemetry tracing configuration for I3.

All OpenTelemetry packages are soft-imported.  If any are missing, the
tracer configuration becomes a no-op and :func:`get_tracer` returns a
lightweight fake that exposes the subset of the tracer API used inside
the pipeline (``start_as_current_span`` as a no-op context manager and
``start_span`` returning a similar stub).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

_CONFIGURED: bool = False
_TRACER_PROVIDER: Any = None


# ---------------------------------------------------------------------------
# No-op fallback
# ---------------------------------------------------------------------------


class _NoopSpan:
    def set_attribute(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None

    def set_status(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None

    def record_exception(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None

    def add_event(self, *_: Any, **__: Any) -> None:  # noqa: D401
        return None

    def end(self) -> None:  # noqa: D401
        return None

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_: Any, **__: Any):
        yield _NoopSpan()

    def start_span(self, *_: Any, **__: Any) -> _NoopSpan:  # noqa: D401
        return _NoopSpan()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_tracing(service_version: str = "0.0.0") -> None:
    """Initialise the global OpenTelemetry tracer provider.

    This function is idempotent.  Missing optional dependencies are
    tolerated — a single warning is logged and the tracer provider
    remains in its default (no-op) state.
    """
    global _CONFIGURED, _TRACER_PROVIDER
    if _CONFIGURED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import (
            ParentBased,
            TraceIdRatioBased,
        )
    except Exception as exc:  # pragma: no cover - import guarded
        logger.warning(
            "OpenTelemetry SDK not installed (%s); tracing disabled", exc
        )
        _CONFIGURED = True
        return

    try:
        sampler_arg = float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    except ValueError:
        sampler_arg = 1.0
    sampler_arg = min(max(sampler_arg, 0.0), 1.0)

    environment = os.environ.get(
        "OTEL_DEPLOYMENT_ENVIRONMENT",
        os.environ.get("I3_ENV", "development"),
    )

    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "i3"),
            "service.version": service_version,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sampler_arg)),
    )

    # ---- exporter ---------------------------------------------------------
    exporter = _build_otlp_exporter()
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    _CONFIGURED = True
    logger.info(
        "OpenTelemetry tracing configured",
        extra={
            "sampler_arg": sampler_arg,
            "environment": environment,
            "exporter": "otlp" if exporter is not None else "none",
        },
    )


def _build_otlp_exporter() -> Any:
    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        insecure = endpoint.startswith("http://")
        return OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    except Exception as exc:
        logger.warning(
            "OTLP exporter unavailable (%s); spans will not be exported",
            exc,
        )
        return None


def instrument_libraries(app: Any = None) -> None:
    """Install OpenTelemetry auto-instrumentation for well-known libraries.

    Each instrumentor is soft-imported so a missing package never blocks
    server startup.
    """
    _instrument_fastapi(app)
    _instrument_httpx()
    _instrument_sqlite3()
    _instrument_logging()


def _instrument_fastapi(app: Any) -> None:
    if app is None:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # pragma: no cover - import guarded
        logger.debug("FastAPI instrumentation unavailable: %s", exc)


def _instrument_httpx() -> None:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # pragma: no cover - import guarded
        logger.debug("httpx instrumentation unavailable: %s", exc)


def _instrument_sqlite3() -> None:
    try:
        from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

        SQLite3Instrumentor().instrument()
    except Exception as exc:  # pragma: no cover - import guarded
        logger.debug("sqlite3 instrumentation unavailable: %s", exc)


def _instrument_logging() -> None:
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception as exc:  # pragma: no cover - import guarded
        logger.debug("logging instrumentation unavailable: %s", exc)


def get_tracer(name: str = "i3") -> Any:
    """Return a tracer.  Falls back to a no-op if OTel is unavailable."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NoopTracer()


@contextmanager
def span(name: str, **attributes: Any):
    """Convenience context manager that opens a span with attributes.

    Falls back to :func:`contextlib.nullcontext` if tracing is disabled.
    """
    tracer = get_tracer()
    try:
        ctx = tracer.start_as_current_span(name)
    except Exception:
        ctx = nullcontext(_NoopSpan())
    with ctx as current:
        try:
            for k, v in attributes.items():
                try:
                    current.set_attribute(k, v)
                except Exception:
                    pass
        except Exception:
            pass
        yield current


def current_trace_id() -> str:
    """Return the current trace id as a 32-char hex string, or ``""``."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return ""


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider if one is configured."""
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is None:
        return
    try:
        _TRACER_PROVIDER.shutdown()
    except Exception:  # pragma: no cover - defensive
        logger.debug("tracer provider shutdown raised", exc_info=True)
    _TRACER_PROVIDER = None
