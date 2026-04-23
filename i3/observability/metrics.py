"""Prometheus + OpenTelemetry metrics for I3.

The metrics module exposes a **single registry** used by both the
Prometheus scrape endpoint (``/api/metrics``) and the OTel meter
provider when the OTel Prometheus exporter is installed.

All metric objects are defined at module scope so they behave like
singletons.  If ``prometheus_client`` is not installed, the module falls
back to ``_NullMetric`` stubs whose ``inc``, ``observe``, ``set``, and
``labels`` methods are no-ops — this keeps the application runnable in
environments without Prometheus.

Convention
----------
* Counters:    ``<subsystem>_<noun>_total``
* Histograms:  ``<subsystem>_<noun>_seconds`` (always in seconds)
* Gauges:      ``<subsystem>_<noun>``
* All I3 metrics are prefixed with ``i3_``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - import guarded
    _PROM_AVAILABLE = False

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class CollectorRegistry:  # type: ignore[no-redef]
        pass

    def generate_latest(_registry: Any = None) -> bytes:  # type: ignore[no-redef]
        return b""


# Dedicated registry so unit tests can create / tear down without mutating
# the global prometheus_client state.
REGISTRY: CollectorRegistry = (
    CollectorRegistry() if _PROM_AVAILABLE else CollectorRegistry()
)


class _NullMetric:
    """No-op stand-in used when prometheus_client is not installed."""

    def labels(self, *_: Any, **__: Any) -> _NullMetric:
        return self

    def inc(self, *_: Any, **__: Any) -> None:
        return None

    def observe(self, *_: Any, **__: Any) -> None:
        return None

    def set(self, *_: Any, **__: Any) -> None:
        return None


def _counter(name: str, desc: str, labels: list[str] | None = None) -> Any:
    if not _PROM_AVAILABLE:
        return _NullMetric()
    return Counter(name, desc, labelnames=labels or [], registry=REGISTRY)


def _histogram(
    name: str,
    desc: str,
    labels: list[str] | None = None,
    buckets: tuple[float, ...] | None = None,
) -> Any:
    if not _PROM_AVAILABLE:
        return _NullMetric()
    kwargs: dict[str, Any] = {
        "labelnames": labels or [],
        "registry": REGISTRY,
    }
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, desc, **kwargs)


def _gauge(name: str, desc: str, labels: list[str] | None = None) -> Any:
    if not _PROM_AVAILABLE:
        return _NullMetric()
    return Gauge(name, desc, labelnames=labels or [], registry=REGISTRY)


# ---------------------------------------------------------------------------
# Bucket presets (seconds)
# ---------------------------------------------------------------------------

_HTTP_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)
_PIPELINE_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
)
_INFERENCE_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


# ---------------------------------------------------------------------------
# HTTP metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = _counter(
    "i3_http_requests_total",
    "Total HTTP requests handled by the I3 server.",
    labels=["method", "route", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = _histogram(
    "i3_http_request_duration_seconds",
    "End-to-end HTTP request latency in seconds.",
    labels=["method", "route", "status"],
    buckets=_HTTP_BUCKETS,
)
HTTP_REQUESTS_IN_PROGRESS = _gauge(
    "i3_http_requests_in_progress",
    "Number of HTTP requests currently being processed.",
    labels=["method"],
)

# ---------------------------------------------------------------------------
# Pipeline stage metrics
# ---------------------------------------------------------------------------

# Keep the label cardinality bounded — pipeline stages are a small fixed set.
PIPELINE_STAGE_LATENCY_SECONDS = _histogram(
    "i3_pipeline_stage_duration_seconds",
    "Latency of a single pipeline stage (sanitize, encode, adapt, route, "
    "generate, postprocess, diary) in seconds.",
    labels=["stage"],
    buckets=_PIPELINE_BUCKETS,
)
PIPELINE_TOTAL_LATENCY_SECONDS = _histogram(
    "i3_pipeline_total_duration_seconds",
    "End-to-end pipeline latency (all stages) in seconds.",
    buckets=_INFERENCE_BUCKETS,
)
PIPELINE_ERRORS_TOTAL = _counter(
    "i3_pipeline_errors_total",
    "Pipeline stages that raised exceptions.",
    labels=["stage", "error"],
)

# ---------------------------------------------------------------------------
# Router metrics
# ---------------------------------------------------------------------------

ROUTER_DECISIONS_TOTAL = _counter(
    "i3_router_decisions_total",
    "Router arm selection counts.",
    labels=["arm", "reason"],
)
ROUTER_ARM_POSTERIOR_MEAN = _gauge(
    "i3_router_arm_posterior_mean",
    "Posterior mean reward for each bandit arm.",
    labels=["arm"],
)

# ---------------------------------------------------------------------------
# SLM / encoder inference metrics
# ---------------------------------------------------------------------------

SLM_INFERENCE_LATENCY_SECONDS = _histogram(
    "i3_slm_inference_duration_seconds",
    "SLM inference latency split by phase (prefill, decode).",
    labels=["phase"],
    buckets=_INFERENCE_BUCKETS,
)
SLM_TOKENS_GENERATED_TOTAL = _counter(
    "i3_slm_tokens_generated_total",
    "Total tokens produced by the local SLM.",
)
TCN_ENCODER_LATENCY_SECONDS = _histogram(
    "i3_tcn_encoder_duration_seconds",
    "TCN encoder forward-pass latency in seconds.",
    buckets=_PIPELINE_BUCKETS,
)

# ---------------------------------------------------------------------------
# WebSocket metrics
# ---------------------------------------------------------------------------

WEBSOCKET_CONNECTIONS = _gauge(
    "i3_websocket_connections",
    "Number of currently-open WebSocket connections.",
)
WEBSOCKET_MESSAGES_TOTAL = _counter(
    "i3_websocket_messages_total",
    "Total WebSocket messages handled.",
    labels=["direction"],
)

# ---------------------------------------------------------------------------
# Privacy metrics
# ---------------------------------------------------------------------------

PII_HITS_TOTAL = _counter(
    "i3_privacy_pii_hits_total",
    "PII patterns matched by the privacy sanitizer.",
    labels=["pattern"],
)
PRIVACY_OVERRIDES_TOTAL = _counter(
    "i3_privacy_overrides_total",
    "Router decisions overridden for privacy reasons.",
    labels=["reason"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def metrics_enabled() -> bool:
    """Return True if ``/api/metrics`` should be served."""
    if not _PROM_AVAILABLE:
        return False
    flag = os.environ.get("I3_METRICS_ENABLED", "1")
    return flag not in {"0", "false", "False", "no"}


def render_prometheus() -> tuple[bytes, str]:
    """Render the current registry as Prometheus exposition text."""
    if not _PROM_AVAILABLE:
        return b"# prometheus_client not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    """Context manager that records pipeline stage latency in seconds.

    Usage::

        with stage_timer("encode"):
            embedding = encoder(features)
    """
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        PIPELINE_ERRORS_TOTAL.labels(stage=stage, error=type(exc).__name__).inc()
        raise
    finally:
        PIPELINE_STAGE_LATENCY_SECONDS.labels(stage=stage).observe(
            time.perf_counter() - start
        )


def record_http(method: str, route: str, status: int, duration_seconds: float) -> None:
    """Convenience recorder used by the middleware."""
    status_str = str(status)
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status=status_str).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method, route=route, status=status_str
    ).observe(duration_seconds)


def record_router_decision(arm: str, reason: str = "bandit") -> None:
    ROUTER_DECISIONS_TOTAL.labels(arm=arm, reason=reason).inc()


def set_router_posterior(arm: str, mean: float) -> None:
    ROUTER_ARM_POSTERIOR_MEAN.labels(arm=arm).set(float(mean))


def record_pii_hit(pattern: str, count: int = 1) -> None:
    if count <= 0:
        return
    PII_HITS_TOTAL.labels(pattern=pattern).inc(count)


def record_slm_inference(phase: str, duration_seconds: float) -> None:
    SLM_INFERENCE_LATENCY_SECONDS.labels(phase=phase).observe(duration_seconds)


def configure_otel_metrics(service_version: str = "0.0.0") -> None:
    """Wire OpenTelemetry meter provider if the SDK is installed.

    This is best-effort: the Prometheus registry is the source of truth
    for the ``/api/metrics`` endpoint.  The OTel bridge is useful when
    the collector should also ingest metrics via OTLP.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
    except Exception as exc:  # pragma: no cover - import guarded
        logger.debug("OTel metrics SDK unavailable (%s); skipping", exc)
        return

    try:
        insecure = endpoint.startswith("http://")
        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=30_000
        )
        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "i3"),
                "service.version": service_version,
            }
        )
        provider = MeterProvider(metric_readers=[reader], resource=resource)
        otel_metrics.set_meter_provider(provider)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("OTel metrics wiring failed: %s", exc)
