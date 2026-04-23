# I³ — Container usage

This directory contains the supporting scripts used by the I³ images:

| File              | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| `entrypoint.sh`   | PID-1 launcher (via `tini`). Validates config, execs uvicorn. |
| `healthcheck.sh`  | `HEALTHCHECK` probe. Calls `/api/health` via `urllib`.     |
| `init.d/*.sh`     | Optional bootstrap hooks, sourced before uvicorn starts.   |

The image itself is defined by:

- **`Dockerfile`** — multi-stage production build (Python 3.11-slim, non-root
  `i3:10001`, tini, CPU-only torch, no pip/poetry/curl in runtime).
- **`docker/Dockerfile.dev`** — single-stage dev image with hot-reload + full Poetry
  toolchain, used via `docker-compose.override.yml`.

---

## Quick start

### 1. Build the production image

```sh
DOCKER_BUILDKIT=1 docker build \
    --tag i3:1.0.0 \
    --build-arg IMAGE_VERSION=1.0.0 \
    --build-arg IMAGE_REVISION="$(git rev-parse --short HEAD)" \
    --build-arg IMAGE_CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    .
```

### 2. Run it (dev defaults)

```sh
# Required: a 32-byte (base64) encryption key for the ModelEncryptor.
export I3_ENCRYPTION_KEY="$(openssl rand -base64 32)"

docker compose up --build
```

The server binds on `http://localhost:8000`. Visit `/api/health` to confirm
liveness — the `HEALTHCHECK` probe uses the same path.

### 3. Production deploy

```sh
docker compose \
    -f docker-compose.yml \
    -f docker-compose.prod.yml \
    up -d
```

The prod overlay:

- enforces `read_only: true` on the root filesystem (tmpfs for `/tmp` and
  `/app/data/tmp`),
- drops ALL capabilities and sets `no-new-privileges`,
- runs 2 replicas by default (`I3_REPLICAS=…`),
- fronts the server with nginx for TLS termination.

---

## Passing the encryption key

The entrypoint **refuses to start** without one of:

- `I3_ENCRYPTION_KEY` — the key material itself, **or**
- `I3_ENCRYPTION_KEY_FILE` — a path (Docker secret / mounted file) whose
  contents will be loaded into `I3_ENCRYPTION_KEY`.

### Option A — environment file (`.env`)

```sh
echo "I3_ENCRYPTION_KEY=$(openssl rand -base64 32)" >> .env
docker compose up
```

### Option B — Docker secret / bind mount

```sh
docker run --rm \
    -v "$PWD/secrets/i3.key:/run/secrets/i3_key:ro" \
    -e I3_ENCRYPTION_KEY_FILE=/run/secrets/i3_key \
    -p 8000:8000 \
    i3:1.0.0
```

### Option C — Swarm / Kubernetes secret

Mount the secret at any readable path inside the container and point
`I3_ENCRYPTION_KEY_FILE` at it.

---

## Mounting custom configs

All YAML under `configs/` is mounted **read-only** at `/app/configs`. To
override the default without rebuilding:

```sh
docker compose run --rm \
    -v "$PWD/my-configs:/app/configs:ro" \
    -e I3_CONFIG=/app/configs/my-profile.yaml \
    i3-server
```

Or, for recurring use, add a `docker-compose.override.yml`:

```yaml
services:
  i3-server:
    volumes:
      - ./my-configs:/app/configs:ro
    environment:
      I3_CONFIG: /app/configs/my-profile.yaml
```

---

## Hot-reload dev loop

```sh
cp docker-compose.override.yml.example docker-compose.override.yml
docker compose up --build
```

Code edits under `server/` or `i3/` reload immediately; `ipython`, `pytest`,
and `ruff` are all available inside the container:

```sh
docker compose exec i3-server ruff check .
docker compose exec i3-server pytest -q
docker compose exec i3-server ipython
```

---

## Signal handling & shutdown

- `tini` is always PID 1 — zombies are reaped automatically.
- `entrypoint.sh` uses `exec` so `SIGTERM` from `docker stop` is delivered
  directly to uvicorn.
- `stop_grace_period: 30s` in `docker-compose.yml` gives in-flight requests
  time to complete before `SIGKILL`.

---

## Image layers / size budget

| Layer                              | Approx size |
| ---------------------------------- | ----------- |
| `python:3.11-slim-bookworm`        | ~45 MiB     |
| `libgomp1` + `ca-certificates`     | ~2 MiB      |
| `tini` (copied from builder)       | <1 MiB      |
| CPU-only torch + fastapi + numpy   | ~300 MiB    |
| application source                 | <2 MiB      |
| **target total**                   | **< 400 MiB** |

CUDA wheels are intentionally excluded — pull a GPU-enabled image from a
separate Dockerfile if you need accelerated inference.
