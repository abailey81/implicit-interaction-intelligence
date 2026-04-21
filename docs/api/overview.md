# API Overview

I³ exposes three API surfaces: a **REST** API for request/response queries,
a **WebSocket** channel for real-time streaming, and a **Python SDK** that
orchestrates the same pipeline in-process.

!!! tip "Which API do I want?"
    - **Real-time UI**: WebSocket for streaming keystrokes + responses.
    - **Batch / background**: REST for profile, diary, stats.
    - **Embedded**: Python SDK for in-process use.

## Transports at a glance { #glance }

| Surface | Use case | Auth | Docs |
|:--------|:---------|:-----|:-----|
| **REST**      | Profile, diary, stats, profiling, demo utilities | Path regex + rate limit (see [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)) | [REST](rest.md) |
| **WebSocket** | Keystroke streaming + message handling            | Origin allow-list + path regex + rate limit         | [WebSocket](websocket.md) |
| **Python SDK**| Scripting, training, evaluation                    | In-process (no auth)                                | [Python SDK](python.md) |

!!! warning "Demo authentication posture"
    The shipped build has **no caller authentication** on REST or WebSocket.
    Any client that knows a `user_id` may claim it. This is acceptable for
    the on-device demo scope but must be wrapped in JWT or mTLS before any
    multi-tenant deployment — see [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md).

## Error shape { #errors }

Both REST and WebSocket emit a uniform error envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "detail": "limit must be between 1 and 100",
    "trace_id": "6f2a0b6a6b3a4a2f"
  }
}
```

- `code` is from a small, stable enumeration (`VALIDATION_ERROR`,
  `NOT_FOUND`, `RATE_LIMITED`, `POLICY_VIOLATION`, `INTERNAL`).
- `detail` **never** echoes user input, internal paths, stack traces, or
  class names.
- `trace_id` is the OpenTelemetry span id for server-side correlation.

## Rate limiting { #rate-limit }

Shared per-user sliding-window limiter across both REST and WebSocket:

| Bucket | Limit |
|:-------|:------|
| Default  | 600 req / min / user |
| `/demo/*`| 60 req / min, demo mode only |

Responses over quota carry `Retry-After` and a `RATE_LIMITED` envelope.

## Content types { #content }

- **REST**: `application/json` request and response, UTF-8.
- **WebSocket**: JSON frames, text only. Binary frames are rejected.

## Versioning { #versioning }

The REST API is currently at `1.0.0` (see `/health`). Breaking changes
will be version-gated via path prefix (`/v2/...`). The WebSocket protocol
carries its own `protocol_version` in the `session_started` frame.

## Where to go next { #next }

<div class="i3-grid" markdown>

<div class="i3-card" markdown>
### :material-connection: REST
Every endpoint, every parameter, every error. [REST](rest.md).
</div>

<div class="i3-card" markdown>
### :material-lan-connect: WebSocket
Frame-level protocol reference with the security caps. [WebSocket](websocket.md).
</div>

<div class="i3-card" markdown>
### :material-language-python: Python SDK
Module reference rendered via `mkdocstrings`. [Python SDK](python.md).
</div>

<div class="i3-card" markdown>
### :material-chart-timeline-variant: Telemetry
OTEL spans & Prometheus metrics: [Observability](../operations/observability.md).
</div>

</div>
