# Docker

The repository ships a multi-stage `Dockerfile` and a `docker/` directory
with a `docker-compose.yaml` for local stack-up. This page covers image
structure, invocation patterns, and hardening.

## Image structure { #image }

```dockerfile title="Dockerfile (shipped)"
# ---- builder stage ---------------------------------------------------
FROM python:3.12-slim AS builder
ENV POETRY_NO_INTERACTION=1 POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app
RUN pip install poetry==1.8.3 && \
    apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY . .
RUN poetry install --only main

# ---- runtime stage ---------------------------------------------------
FROM python:3.12-slim AS runtime
RUN groupadd -r i3 && useradd -r -g i3 -m -d /app i3
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app
RUN chown -R i3:i3 /app
USER i3
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health').read()" || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key properties:

- **Multi-stage** — builder has Poetry + `build-essential`; runtime is
  slim (≈ 180 MB final).
- **Non-root** — runs as `i3`.
- **Health-checked** — Docker/Compose can observe the liveness probe.
- **Deterministic** — locked via `poetry.lock`.

## Build { #build }

```bash
docker build -t i3:1.0.0 .
# Tag for registry:
docker tag i3:1.0.0 ghcr.io/abailey81/implicit-interaction-intelligence:1.0.0
docker push        ghcr.io/abailey81/implicit-interaction-intelligence:1.0.0
```

## Run { #run }

### Minimal

```bash
docker run -it --rm \
    -p 8000:8000 \
    -e I3_ENCRYPTION_KEY="$(cat fernet.key)" \
    -e I3_CORS_ORIGINS=http://localhost:8000 \
    -v i3-data:/app/data \
    i3:1.0.0
```

### Full (with cloud)

```bash
docker run -d --name i3 \
    -p 8000:8000 \
    -e I3_ENCRYPTION_KEY="$(cat fernet.key)" \
    -e ANTHROPIC_API_KEY="$(cat anthropic.key)" \
    -e I3_CORS_ORIGINS="https://i3.example.org" \
    -e OTEL_EXPORTER_OTLP_ENDPOINT="http://otel:4317" \
    --read-only \
    --tmpfs /tmp \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    -v i3-data:/app/data \
    -v i3-checkpoints:/app/checkpoints:ro \
    i3:1.0.0
```

!!! warning "Never pass secrets as build args"
    `ARG` and `ENV` at build time are baked into layer metadata. Pass
    secrets only at `docker run` or via Docker/compose secrets.

## Compose (full stack) { #compose }

```yaml title="docker/docker-compose.yaml (excerpt)"
services:
  i3:
    image: i3:1.0.0
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      I3_CORS_ORIGINS: "http://localhost:8000"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
      OTEL_SERVICE_NAME: "i3"
    secrets:
      - i3_fernet
      - anthropic_key
    volumes:
      - i3-data:/app/data
      - i3-checkpoints:/app/checkpoints:ro
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop: [ALL]
    depends_on: [otel-collector]

  otel-collector:
    image: otel/opentelemetry-collector:0.98.0
    command: ["--config=/etc/otel-config.yaml"]
    configs:
      - source: otel_config
        target: /etc/otel-config.yaml
    ports:
      - "4317:4317"    # OTLP gRPC
      - "9464:9464"    # Prometheus

volumes:
  i3-data:
  i3-checkpoints:

configs:
  otel_config:
    file: ./otel-config.yaml

secrets:
  i3_fernet:
    file: ./secrets/fernet.key
  anthropic_key:
    file: ./secrets/anthropic.key
```

Run:

```bash
docker compose -f docker/docker-compose.yaml up -d
```

## Hardening checklist { #hardening }

- [x] Runs as non-root (`USER i3`).
- [x] `--read-only` root filesystem with `tmpfs` for `/tmp`.
- [x] `--cap-drop=ALL` and `--security-opt=no-new-privileges`.
- [x] No SSH / shell / extraneous tooling in the runtime image.
- [x] Slim base image; rebuilt monthly for CVE patching.
- [x] Health check points at `/health`, never at `/demo/*`.
- [x] Secrets via Docker secrets, never ENV in compose YAML.
- [x] Volume for `checkpoints/` mounted `:ro`.
- [x] Periodic `docker scout cves i3:1.0.0` in CI.

## Image size { #size }

| Layer | Size |
|:------|-----:|
| `python:3.12-slim` base | ~120 MB |
| Torch + site-packages   | ~60 MB |
| App code                | ~2 MB  |
| **Total runtime image** | **~180 MB** |

If you need a smaller image, swap the base for `python:3.12-alpine` and
compile Torch from wheels matching `musl` — not recommended for production.

## Further reading { #further }

- [Deployment](deployment.md) — systemd variant.
- [Kubernetes](kubernetes.md) — multi-replica rollouts.
