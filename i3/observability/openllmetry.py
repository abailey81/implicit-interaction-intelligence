"""OpenLLMetry / Traceloop instrumentation for I3.

`OpenLLMetry <https://github.com/traceloop/openllmetry>`_ is an open
implementation of the **OpenTelemetry GenAI semantic conventions**.
When initialised it auto-instruments every major LLM SDK (Anthropic,
OpenAI, Cohere, Bedrock, …) so spans carry the standardised
``gen_ai.*`` attributes required by any OTel backend.

For I3 this means a single call to :func:`setup_traceloop` gives us:

* ``gen_ai.system = "anthropic"``
* ``gen_ai.operation.name = "chat"``
* ``gen_ai.request.model = "claude-sonnet-4-5"``
* ``gen_ai.usage.input_tokens`` / ``gen_ai.usage.output_tokens``
* request / response latency in milliseconds
* tool-use traces when Claude invokes tools

References:
    * OpenTelemetry GenAI Semantic Conventions (2024-10 — "Anthropic"
      profile): https://opentelemetry.io/docs/specs/semconv/gen-ai/anthropic/
    * Traceloop / OpenLLMetry documentation (2024-11):
      https://www.traceloop.com/docs/openllmetry

Install hint::

    pip install "traceloop-sdk>=0.30" "opentelemetry-instrumentation-anthropic>=0.30"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    from traceloop.sdk import Traceloop  # type: ignore[import-not-found]

    _TRACELOOP_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    Traceloop = None  # type: ignore[assignment, misc]
    _TRACELOOP_AVAILABLE = False


def is_available() -> bool:
    """Return ``True`` iff ``traceloop-sdk`` is importable."""
    return _TRACELOOP_AVAILABLE


# ---------------------------------------------------------------------------
# Resource attributes (OTel GenAI semconv)
# ---------------------------------------------------------------------------

# Canonical values from https://opentelemetry.io/docs/specs/semconv/gen-ai/
GENAI_SYSTEM_ANTHROPIC: str = "anthropic"
GENAI_OPERATION_CHAT: str = "chat"


@dataclass
class TraceloopStatus:
    """Outcome of a :func:`setup_traceloop` call.

    Attributes:
        enabled: ``True`` iff Traceloop was actually initialised.
        reason: Human-readable reason when *enabled* is ``False``.
        endpoint: The OTLP endpoint that was used (``""`` if none).
        resource_attributes: GenAI semconv attributes placed on the
            resource descriptor.
    """

    enabled: bool
    reason: str
    endpoint: str = ""
    resource_attributes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def setup_traceloop(
    *,
    app_name: str = "i3",
    model: str = "claude-sonnet-4-5",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    disable_batch: bool = False,
) -> TraceloopStatus:
    """Initialise Traceloop / OpenLLMetry.

    The function is a no-op if ``traceloop-sdk`` is not installed or if
    neither *endpoint* nor the ``TRACELOOP_BASE_URL`` /
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variables are set — the
    rest of the pipeline remains functional either way.

    Args:
        app_name: Logical application name (becomes ``service.name``).
        model: Default Anthropic model id recorded on every span as
            ``gen_ai.request.model`` unless the individual SDK call
            overrides it.
        endpoint: Explicit OTLP/HTTP or OTLP/gRPC endpoint.  When
            omitted, Traceloop picks it up from the environment.
        api_key: Optional Traceloop cloud API key.  Overrides
            ``TRACELOOP_API_KEY``.
        disable_batch: Export spans synchronously (useful in tests).

    Returns:
        :class:`TraceloopStatus` describing what was initialised.
    """
    if not _TRACELOOP_AVAILABLE:
        return TraceloopStatus(
            enabled=False,
            reason="traceloop-sdk not installed",
        )

    resolved_endpoint = (
        endpoint
        or os.environ.get("TRACELOOP_BASE_URL")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or ""
    )
    resolved_key = api_key if api_key is not None else os.environ.get(
        "TRACELOOP_API_KEY", ""
    )

    if not resolved_endpoint and not resolved_key:
        return TraceloopStatus(
            enabled=False,
            reason="no OTLP endpoint or TRACELOOP_API_KEY configured",
        )

    # Apply GenAI semconv resource attributes via env vars; Traceloop
    # forwards OTEL_RESOURCE_ATTRIBUTES into the OTel SDK resource.
    genai_attrs = {
        "gen_ai.system": GENAI_SYSTEM_ANTHROPIC,
        "gen_ai.operation.name": GENAI_OPERATION_CHAT,
        "gen_ai.request.model": model,
    }
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    pieces = [p for p in existing.split(",") if p]
    for k, v in genai_attrs.items():
        # Only append if not already present; user env wins.
        if not any(p.startswith(f"{k}=") for p in pieces):
            pieces.append(f"{k}={v}")
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(pieces)

    assert Traceloop is not None
    try:
        kwargs: dict[str, object] = {
            "app_name": app_name,
            "disable_batch": disable_batch,
        }
        if resolved_endpoint:
            kwargs["api_endpoint"] = resolved_endpoint
        if resolved_key:
            kwargs["api_key"] = resolved_key
        Traceloop.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Traceloop.init() failed: %s", exc)
        return TraceloopStatus(
            enabled=False,
            reason=f"init_failed:{type(exc).__name__}",
            endpoint=resolved_endpoint,
            resource_attributes=genai_attrs,
        )

    logger.info(
        "Traceloop / OpenLLMetry enabled (app=%s, model=%s, endpoint=%s)",
        app_name,
        model,
        resolved_endpoint or "<traceloop-cloud>",
    )
    return TraceloopStatus(
        enabled=True,
        reason="ok",
        endpoint=resolved_endpoint,
        resource_attributes=genai_attrs,
    )


__all__ = [
    "GENAI_OPERATION_CHAT",
    "GENAI_SYSTEM_ANTHROPIC",
    "TraceloopStatus",
    "is_available",
    "setup_traceloop",
]
