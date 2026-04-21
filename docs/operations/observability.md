# Observability

I³ emits **OpenTelemetry** traces, **Prometheus** metrics, and structured
**JSON** logs. This page documents what is instrumented, how to point it
at your collector, and the dashboards that make an I³ instance legible.

!!! tip "Related reading"
    [ADR 0007 — OpenTelemetry for observability](../adr/0007-opentelemetry-for-observability.md).

## Signals at a glance { #signals }

| Signal | Transport | Where enabled |
|:-------|:----------|:--------------|
| Traces  | OTLP gRPC / HTTP | `i3/observability/traces.py` |
| Metrics | Prometheus `/metrics` endpoint | `server/middleware.py` |
| Logs    | `stdout` JSON via `logging` | `configs/logging.yaml` |

## 1. Traces { #traces }

### Instrumented spans

| Span name | Layer | Attributes |
|:----------|:------|:-----------|
| `pipeline.process`        | Pipeline   | `user_id`, `session_id`, `route`, `latency_ms` |
| `privacy.sanitize`        | Privacy    | `redaction_count`, `categories` |
| `interaction.features`    | Layer 1    | `message_chars`, `keystrokes` |
| `encoder.forward`         | Layer 2    | `seq_len`, `device` |
| `user_model.update`       | Layer 3    | `baseline_established`, `z_score_p95` |
| `adaptation.controller`   | Layer 4    | `cognitive_load`, `tone`, `a11y` |
| `router.decide`           | Layer 5    | `arm_chosen`, `override`, `posterior_*` |
| `slm.generate`            | Layer 6a   | `max_tokens`, `int8`, `finish_reason` |
| `cloud.complete`          | Layer 6b   | `model`, `input_tokens`, `output_tokens`, `retries` |
| `diary.log`               | Layer 7    | `topics_count` |

### Export

```bash
export OTEL_SERVICE_NAME=i3
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1   # 10 % sampling
```

Disabled by default when no endpoint is set — I³ falls back to a no-op
tracer so the pipeline has zero overhead.

### Trace correlation

- Each REST response carries the `trace_id` in the `X-Trace-Id` header.
- Each WebSocket `response` frame carries `trace_id` alongside `latency_ms`.
- Logs include `trace_id` and `span_id` at every level.

## 2. Metrics { #metrics }

Scraped at `GET /metrics` (Prometheus exposition format).

### Request / response

```
i3_requests_total{method,path,status}     counter
i3_request_duration_seconds{method,path}  histogram
i3_requests_in_flight                     gauge
i3_rate_limit_exceeded_total{user_id}     counter
```

### Pipeline

```
i3_pipeline_latency_seconds{layer}        histogram   # per-layer time
i3_pipeline_errors_total{layer,class}     counter
i3_pipeline_messages_total{route}         counter     # local / cloud
i3_session_active                         gauge
i3_session_duration_seconds               histogram
```

### Router

```
i3_router_decisions_total{arm,override}   counter
i3_router_reward                          histogram   # 0–1
i3_router_laplace_refit_seconds           histogram
i3_router_posterior_mean{arm}             gauge
```

### SLM

```
i3_slm_tokens_generated_total             counter
i3_slm_generation_latency_seconds         histogram
i3_slm_memory_mb                          gauge
i3_slm_quantized                          gauge       # 0/1
```

### Cloud

```
i3_cloud_requests_total{status}           counter
i3_cloud_retries_total                    counter
i3_cloud_latency_seconds                  histogram
i3_cloud_tokens_in_total                  counter
i3_cloud_tokens_out_total                 counter
```

### Privacy

```
i3_pii_redactions_total{category}         counter
i3_privacy_override_total                 counter
i3_privacy_auditor_hits_total             counter     # should stay at 0
```

## 3. Logs { #logs }

JSON, one line per event, to `stdout`. Example:

```json
{
  "ts": "2026-04-22T10:18:44.312Z",
  "level": "INFO",
  "event": "pipeline.process",
  "user_id": "alice",
  "session_id": "3b1b9d18-…",
  "route": "local",
  "latency_ms": 148,
  "trace_id": "6f2a0b6a6b3a4a2f",
  "span_id": "01020304"
}
```

- **Never**: raw text, Anthropic key, Fernet key.
- **Always**: `trace_id`, `span_id`, `event`, `level`, `ts`.
- **Optionally**: `user_id` (always the validated regex-matching value).

### Log levels

| Level | Use |
|:------|:----|
| `DEBUG`   | Per-step pipeline timings, tensor shapes (dev only) |
| `INFO`    | Session start/end, router decisions, cloud calls, demo ops |
| `WARNING` | Rate-limit hits, validation rejections, degraded paths |
| `ERROR`   | Pipeline exceptions, cloud failures, privacy auditor hits |
| `CRITICAL`| Fernet key missing or invalid |

## 4. Dashboards { #dashboards }

Recommended Grafana panels (a JSON export is in `scripts/grafana/`):

1. **Pipeline latency** — P50/P95/P99 per layer, stacked.
2. **Route share** — local vs cloud over time, with privacy-override
   share annotated.
3. **Reward** — histogram heatmap by route.
4. **Cloud cost** — tokens/sec and rolling \$/min estimate.
5. **Privacy** — PII redaction counts by category; auditor hits as
   RED-annotated alert.
6. **Session health** — active sessions, avg duration, P95 messages/session.

## 5. Alerts { #alerts }

```yaml title="prometheus-alerts.yaml (excerpt)"
groups:
  - name: i3
    rules:
      - alert: I3PipelineP95Over500ms
        expr: histogram_quantile(0.95, sum(rate(i3_pipeline_latency_seconds_bucket[5m])) by (le)) > 0.5
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: Pipeline P95 > 500 ms

      - alert: I3PrivacyAuditorHit
        expr: increase(i3_privacy_auditor_hits_total[5m]) > 0
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: Privacy auditor detected a potential raw-text leak
          runbook: https://example.org/docs/operations/runbook/#privacy-auditor

      - alert: I3RouterStuckOnCloud
        expr: sum(rate(i3_router_decisions_total{arm="cloud"}[5m]))
              / sum(rate(i3_router_decisions_total[5m])) > 0.9
        for: 15m
        labels: {severity: warning}
        annotations:
          summary: Router routed >90 % to cloud for 15 min

      - alert: I3CloudErrorRate
        expr: sum(rate(i3_cloud_requests_total{status="error"}[5m]))
              / sum(rate(i3_cloud_requests_total[5m])) > 0.1
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: Cloud error rate > 10 %
```

See the [Runbook](runbook.md) for response steps to each alert.

## 6. Profiling reports { #profiling }

On start-up the server writes
`/var/lib/i3/reports/feasibility-YYYYMMDD.md` with the edge feasibility
matrix. Ops can inspect it for regressions after a deploy:

```markdown
## I3 feasibility report — 2026-04-22

| component  | P50 ms | P95 ms | memory MB |
|-----------|-------:|-------:|----------:|
| encoder   | 3      | 4      | 0.2       |
| adaptation| 1      | 2      | 0.05      |
| router    | 2      | 3      | 0.1       |
| slm       | 143    | 181    | 7.1       |
| **total** | **149**| **190**| **7.45**  |
```

Enforce a CI gate: P95 regression > 25 % fails the deploy.

## 7. Further reading { #further }

- [Deployment](deployment.md) — env var catalogue.
- [Runbook](runbook.md) — alert → action map.
- [ADR 0007](../adr/0007-opentelemetry-for-observability.md).
