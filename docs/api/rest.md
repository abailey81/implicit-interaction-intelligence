# REST API

The REST surface complements the [WebSocket stream](websocket.md) with
request/response queries for profiles, diary entries, statistics, and the
gated demo utilities.

!!! warning "Demo authentication posture"
    The shipped build has **no caller authentication**. Any client that
    knows a `user_id` may read it. See [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md).

## Conventions { #conventions }

- **Base URL**: `http://127.0.0.1:8000` (demo) — configurable via
  `server.host` / `server.port` in `configs/default.yaml`.
- **Content type**: `application/json; charset=utf-8`.
- **User id format**: `^[a-zA-Z0-9_-]{1,64}$` — anchored, path-traversal-safe.
- **Pagination**: `limit ∈ [1, 100]`, `offset ∈ [0, 10000]`, explicit
  `int` coercion.
- **Rate limiting**: 600 req/min/user, shared with WebSocket; responses
  over quota carry `Retry-After`.

All endpoints emit a `trace_id` header mirroring the OpenTelemetry span id.

## Endpoint index { #index }

| Method | Path | Purpose |
|:------:|:-----|:--------|
| GET  | `/health`                     | Liveness probe |
| GET  | `/user/{user_id}/profile`     | Embedded user profile |
| GET  | `/user/{user_id}/diary`       | Paginated diary entries |
| GET  | `/user/{user_id}/stats`       | Aggregate statistics |
| GET  | `/profiling/report`           | Edge-feasibility profiling |
| POST | `/demo/reset`                 | Wipe all state (demo only) |
| POST | `/demo/seed`                  | Seed demo profiles (demo only) |

---

## `GET /health` { #health }

Liveness probe. Always returns `200 OK` if the process is up. Intentionally
minimal — it does **not** expose pipeline internals, memory, hostname, or
config. For detailed diagnostics, use [`/profiling/report`](#profiling-report)
or the OTEL collector.

=== "Request"

    ```http
    GET /health HTTP/1.1
    Host: 127.0.0.1:8000
    ```

=== "Response"

    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {"status": "healthy", "version": "1.0.0"}
    ```

=== "curl"

    ```bash
    curl -s http://127.0.0.1:8000/health
    ```

---

## `GET /user/{user_id}/profile` { #get-profile }

Return the user's persisted profile — embedding baseline, long-term style,
relationship-strength score. **Raw user text is never in the response.**

### Path parameters

| Name | Type | Pattern | Description |
|:-----|:-----|:--------|:------------|
| `user_id` | `str` | `^[a-zA-Z0-9_-]{1,64}$` | Alphanumeric id |

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | `UserProfile` JSON | |
| `404`  | `{"detail": "Profile not found"}` | Generic — does not echo `user_id` |
| `429`  | `{"detail": "rate_limited"}` | Retry-After set |
| `500`  | `{"detail": "Internal error"}` | Stack traces never exposed |
| `503`  | `{"detail": "Service unavailable"}` | Pipeline not initialised |

=== "curl"

    ```bash
    curl -s http://127.0.0.1:8000/user/alice/profile | jq
    ```

=== "Body (200)"

    ```json
    {
      "user_id": "alice",
      "baseline": {
        "mean_keystroke_interval_ms": 142.3,
        "std_keystroke_interval_ms": 38.1,
        "mean_message_length": 87.5,
        "vocabulary_diversity": 0.58
      },
      "style": {"formality": 0.62, "verbosity": 0.71, "emotionality": 0.48, "directness": 0.55},
      "sessions_observed": 41,
      "messages_observed": 612,
      "relationship_strength": 0.73,
      "last_seen_ts": 1712534421.1
    }
    ```

---

## `GET /user/{user_id}/diary` { #get-diary }

Paginated diary entries — **embeddings, topic TF-IDF, and metrics only**.

### Parameters

| In    | Name    | Type | Range | Default | Description |
|:------|:--------|:-----|:------|:--------|:------------|
| path  | `user_id` | `str` | regex above | — | |
| query | `limit`  | `int` | 1–100   | 10 | Max rows |
| query | `offset` | `int` | 0–10000 | 0  | Rows to skip |

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | `{"entries": […], "count": N}` | Newest first |
| `404`  | `{"detail": "Diary not found"}` | |
| `422`  | Pydantic validation error | Bad `limit` / `offset` |

=== "curl"

    ```bash
    curl -s "http://127.0.0.1:8000/user/alice/diary?limit=5" | jq
    ```

=== "Body (200)"

    ```json
    {
      "count": 2,
      "entries": [
        {
          "entry_id": "018f3a…",
          "ts": 1712534420.8,
          "embedding": [0.11, -0.23, …],
          "topics_tfidf": [["tcn", 0.42], ["receptive", 0.31]],
          "adaptation": [0.74, 0.55, 0.43, 0.62, 0.20, 0.80, 0.10, 0.30],
          "route": "local",
          "latency_ms": 148,
          "engagement": 0.81
        }
      ]
    }
    ```

---

## `GET /user/{user_id}/stats` { #get-stats }

Aggregate statistics — sessions, message count, average engagement,
baseline-deviation history, top routing categories.

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | Aggregate dict | |
| `404`  | `{"detail": "Stats not found"}` | |

=== "Body (200)"

    ```json
    {
      "sessions_total": 41,
      "messages_total": 612,
      "avg_engagement": 0.72,
      "deviation_history_p95": 1.82,
      "route_share": {"local": 0.78, "cloud": 0.22},
      "sensitive_override_count": 3
    }
    ```

---

## `GET /profiling/report` { #profiling-report }

Edge-feasibility profiling — per-component latency, memory footprint, and
whether the full pipeline fits the 200 ms budget on-device.

!!! note "Field allow-list"
    The response is filtered through an allow-list (see `server/routes.py`
    `_PROFILING_ALLOWED_FIELDS`). Hostname, OS, Python version, file paths,
    environment variables, and model artefact paths are dropped even if
    the pipeline accidentally includes them.

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | Filtered profiling report | |
| `500`  | Typed internal error | |

=== "Body (200)"

    ```json
    {
      "components": {
        "encoder": {"latency_ms_p50": 3, "memory_mb": 0.2},
        "adaptation": {"latency_ms_p50": 1, "memory_mb": 0.05},
        "router": {"latency_ms_p50": 2, "memory_mb": 0.1},
        "slm": {"latency_ms_p50": 143, "memory_mb": 7.1}
      },
      "total_latency_ms": 149,
      "memory_mb": 7.45,
      "fits_budget": true,
      "budget_ms": 200,
      "device_class": "phone"
    }
    ```

---

## `POST /demo/reset` { #demo-reset }

**Demo mode only.** Wipes every user profile, session, and diary entry.
Gated behind `I3_DEMO_MODE` so a misconfigured production instance cannot
be wiped by an anonymous `POST`.

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | `{"status": "reset"}` | |
| `403`  | `{"detail": "Demo mode disabled"}` | `I3_DEMO_MODE` not set |

=== "curl"

    ```bash
    I3_DEMO_MODE=true \
      curl -sX POST http://127.0.0.1:8000/demo/reset
    ```

---

## `POST /demo/seed` { #demo-seed }

**Demo mode only.** Seeds the demo with pre-built user profiles and diary
entries so live demos show adaptation from the very first message.
Idempotent (upserts keyed on `demo_user`).

### Responses

| Status | Body | Notes |
|:------:|:-----|:------|
| `200`  | `{"status": "seeded"}` | |
| `403`  | `{"detail": "Demo mode disabled"}` | |
| `503`  | `{"detail": "Demo data unavailable"}` | Demo module missing |

## OpenAPI / Swagger { #openapi }

FastAPI auto-generates both:

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**:     `http://127.0.0.1:8000/redoc`
- **Raw schema**:`http://127.0.0.1:8000/openapi.json`

The schema reflects the Pydantic models for every path above, including
field-level regex validation.

## Security-relevant middleware { #security }

Enforced by `server/middleware.py`:

- **CORS**: `I3_CORS_ORIGINS` env overrides `configs/default.yaml ::
  server.cors_origins`; no wildcard in prod.
- **Rate limit**: sliding-window, per-user.
- **Request size cap**: 64 KiB JSON body.
- **Response hardening**: `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, strict `Cache-Control` on API paths.
- **Trace propagation**: `traceparent` / `tracestate` honoured.

## Further reading { #further }

- [WebSocket API](websocket.md)
- [Python SDK](python.md)
- [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)
- [Runbook](../operations/runbook.md)
