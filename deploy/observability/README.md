# I3 Observability Stack

A self-contained Docker Compose bundle that gives the I3 server a
production-grade observability backend: **Prometheus** for metrics,
**Tempo** for distributed traces, **OpenTelemetry Collector** as the
receive/forward hub, and **Grafana** for visualisation with a
pre-provisioned overview dashboard.

```
  I3 server  --OTLP gRPC :4317-->  OTel Collector  ----->  Tempo  (traces)
      |                                     \-----> Prometheus exporter :8889
      |
      \--/api/metrics (Prometheus) --- scraped by --->   Prometheus

                                      Grafana   reads from  Prometheus + Tempo
```

## Quick start

```bash
# From the repository root:
docker compose -f deploy/observability/docker-compose.observability.yml up -d

# Then run the I3 server on the host:
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export I3_LOG_FORMAT=json
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Once everything is up:

| Service          | URL                                | Notes                                |
|------------------|------------------------------------|--------------------------------------|
| Grafana          | http://localhost:3000              | admin / admin (change via env vars)  |
| Prometheus       | http://localhost:9090              | UI + target health                   |
| Tempo (API)      | http://localhost:3200              | Grafana uses it as a datasource      |
| OTel Collector   | grpc://localhost:4317              | OTLP ingest                          |
| I3 `/metrics`    | http://localhost:8000/api/metrics  | Scraped by Prometheus                |

## Environment variables

Set these in the I3 server process:

| Variable                         | Default                      | Purpose                                   |
|----------------------------------|------------------------------|-------------------------------------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://localhost:4317`      | Where OTLP traces/metrics are shipped     |
| `OTEL_SERVICE_NAME`              | `i3`                         | Resource attribute                        |
| `OTEL_DEPLOYMENT_ENVIRONMENT`    | `development`                | Resource attribute                        |
| `OTEL_TRACES_SAMPLER_ARG`        | `1.0`                        | Head-based sampling ratio                 |
| `I3_LOG_FORMAT`                  | `json`                       | `json` or `console`                       |
| `I3_LOG_LEVEL`                   | `INFO`                       | `DEBUG`, `INFO`, `WARNING`, `ERROR`       |
| `I3_METRICS_ENABLED`             | `1`                          | Set to `0` to disable `/api/metrics`      |
| `SENTRY_DSN`                     | *(unset)*                    | Enables Sentry when non-empty             |
| `SENTRY_TRACES_SAMPLE_RATE`      | `0.1`                        | Sentry performance sampling               |

Grafana admin credentials are controlled by `GF_ADMIN_USER` and
`GF_ADMIN_PASSWORD` on the compose command / an `.env` file next to
the compose file.

## What's in the dashboard

The provisioned *I3 — Runtime Overview* dashboard contains:

1. **HTTP request rate** per route.
2. **HTTP latency** P50 / P95 / P99.
3. **Pipeline stage latency** (stacked P95 for sanitize / encode / adapt /
   route / generate / postprocess / diary).
4. **Router arm distribution** (donut).
5. **Bandit arm posterior mean** time series.
6. **SLM inference latency** (P50/P95/P99 by prefill / decode).
7. **PII sanitizer hits** per pattern type.
8. **WebSocket concurrent connections** gauge.
9. **TCN encoder latency** (P95).
10. **HTTP 5xx rate** per route.

## Files

| File                                          | Role                                                  |
|-----------------------------------------------|-------------------------------------------------------|
| `docker-compose.observability.yml`            | Stack definition (4 services + 1 network + 3 volumes) |
| `otel-collector-config.yaml`                  | Collector: OTLP receivers, batch, redact, exporters   |
| `prometheus.yml`                              | Scrape config (I3, collector self-metrics)            |
| `tempo.yaml`                                  | Tempo single-binary config (local file backend)       |
| `grafana/provisioning/datasources/default.yaml` | Auto-wires Prometheus + Tempo                       |
| `grafana/provisioning/dashboards/default.yaml`  | Auto-discovers JSON dashboards                       |
| `grafana/dashboards/i3-overview.json`         | Main dashboard                                        |

## Local development without the full stack

The I3 server will run with zero observability infrastructure — all
optional dependencies are soft-imported.  Without an OTel collector at
`OTEL_EXPORTER_OTLP_ENDPOINT`, spans are simply dropped (the server
logs a single warning at boot).

## Tearing down

```bash
docker compose -f deploy/observability/docker-compose.observability.yml down -v
```

The `-v` flag also removes persisted Prometheus / Tempo / Grafana
volumes, which is the right thing during demo resets.
