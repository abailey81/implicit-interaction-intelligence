# ADR-0007 — OpenTelemetry for observability

- **Status**: Accepted
- **Date**: 2026-02-18
- **Deciders**: Tamer Atesyakar
- **Technical area**: cross-cutting (observability)

## Context and problem statement { #context }

We need traces and metrics from every layer of the pipeline — perception,
encoder, adaptation, router, SLM, cloud — for both development
(understanding why a response was slow) and operations (alerting on
regressions and privacy-auditor hits).

We also want to ship to whatever backend the operator prefers (Jaeger,
Tempo, Honeycomb, Datadog, Grafana Cloud) without coupling the codebase
to any one vendor.

## Decision drivers { #drivers }

- Vendor-neutral protocol.
- Mature Python SDK with working FastAPI / httpx / SQLAlchemy
  instrumentations.
- Configurable sampling (we do not want to sample 100 % in production).
- Zero overhead when no collector is configured.
- Structured logs with `trace_id` / `span_id` for cross-signal
  correlation.

## Considered options { #options }

1. **OpenTelemetry** (OTel) — traces + metrics, OTLP exporter.
2. **Prometheus only** (no traces).
3. **Datadog / Honeycomb SDK** direct integration.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — OpenTelemetry with OTLP export, plus a
> Prometheus exposition endpoint for metrics. This covers traces,
> metrics, and log correlation through one standard.

### Consequences — positive { #pos }

- Operators choose their backend; we are not locked in.
- `traceparent` propagation lets us correlate client `trace_id` → server
  spans → downstream cloud call.
- A no-op tracer is used when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset —
  zero overhead by default.
- Prometheus scraping at `/metrics` is unchanged; OTel just adds traces
  on top.

### Consequences — negative { #neg }

- OTel's Python SDK is large and has had churn across releases.
  *Mitigation*: pin to a known-good release line and update on a
  quarterly cadence.
- Another dependency surface for supply-chain review. *Mitigation*:
  OTel is CNCF-graduated and reviewed broadly.
- Configuration is env-variable heavy. *Mitigation*: `docs/operations/
  observability.md` lists every one.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — Prometheus only { #opt-2 }

- Yes Very simple.
- No No distributed traces.
- No Harder to answer "why was *this specific request* slow?"
- No No native correlation with logs.

### Option 3 — Vendor SDK { #opt-3 }

- Yes Richer out-of-the-box dashboards from the vendor.
- No Vendor lock-in.
- No Forces every operator to adopt the same backend.
- No Violates the "edge-ready" posture — most vendor SDKs have opinions
  about egress.

## References { #refs }

- [Observability](../operations/observability.md)
- [Runbook](../operations/runbook.md)
- OpenTelemetry Python: <https://opentelemetry.io/docs/languages/python/>
