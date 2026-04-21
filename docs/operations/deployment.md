# Deployment

A practical guide to getting an I³ instance into production. Covers
environment preparation, secrets management, the reverse-proxy layer, the
production-hardening checklist, and a zero-downtime rollout pattern.

!!! tip "Scope"
    This page deploys a **single-node** I³ instance on Linux.
    [Kubernetes](kubernetes.md) covers multi-replica rollouts;
    [Docker](docker.md) covers the container image.

## 1. Prerequisites { #prereq }

- Linux 64-bit with Python 3.10+.
- A dedicated user (`i3svc`, for example), not root.
- A reverse proxy (Nginx / Caddy / Traefik) terminating TLS.
- Persistent storage under `/var/lib/i3/` (SQLite DB, Fernet-encrypted).
- Outbound HTTPS egress (only if cloud routing is enabled).
- A secrets store for `I3_ENCRYPTION_KEY` and `ANTHROPIC_API_KEY`
  (Vault, KMS, systemd-creds, Docker secrets — not the shell history).

## 2. Install { #install }

```bash
sudo useradd -r -m -d /var/lib/i3 i3svc
sudo -u i3svc bash -lc '
  git clone https://github.com/abailey81/implicit-interaction-intelligence.git /var/lib/i3/app
  cd /var/lib/i3/app
  python3 -m venv .venv
  .venv/bin/pip install poetry
  .venv/bin/poetry install --only main
'
```

## 3. Secrets { #secrets }

!!! warning "Never commit any of these"
    The `.env` file, the Fernet key, or the Anthropic key.

=== "systemd-creds"

    ```bash
    sudo systemd-creds encrypt --name=i3-fernet -p - /etc/i3/fernet.cred
    # paste the Fernet key, Ctrl-D
    ```

=== "Docker secrets"

    ```bash
    echo "$FERNET_KEY" | docker secret create i3_fernet -
    ```

=== "Kubernetes"

    ```bash
    kubectl create secret generic i3-fernet --from-literal=key="$FERNET_KEY"
    ```

The application reads secrets from environment variables:

| Variable | Required | Purpose |
|:---------|:--------:|:--------|
| `I3_ENCRYPTION_KEY`       | yes | Fernet key for profile at-rest encryption |
| `ANTHROPIC_API_KEY`       | cloud only | Claude access |
| `I3_CORS_ORIGINS`         | yes | Comma-separated allowed origins |
| `I3_CONFIG`               | optional | Path to config, default `configs/default.yaml` |
| `I3_LOG_LEVEL`            | optional | `INFO` (default) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | optional | OTLP collector |

## 4. Systemd unit { #systemd }

```ini title="/etc/systemd/system/i3.service"
[Unit]
Description=I3 — Implicit Interaction Intelligence
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=i3svc
Group=i3svc
WorkingDirectory=/var/lib/i3/app
Environment=PATH=/var/lib/i3/app/.venv/bin
EnvironmentFile=/etc/i3/env
LoadCredential=fernet:/etc/i3/fernet.cred
ExecStart=/var/lib/i3/app/.venv/bin/uvicorn server.app:app \
    --host 127.0.0.1 --port 8000 \
    --workers 1 --lifespan on \
    --loop uvloop --http httptools \
    --log-config /etc/i3/logging.yaml
Restart=on-failure
RestartSec=5s

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/i3
PrivateDevices=true
CapabilityBoundingSet=
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
LockPersonality=true
SystemCallArchitectures=native
MemoryMax=2G
TasksMax=256

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now i3.service
sudo systemctl status i3.service
```

## 5. Reverse proxy (Nginx) { #nginx }

```nginx title="/etc/nginx/sites-available/i3"
upstream i3 {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name i3.example.org;

    # --- TLS (Let's Encrypt / ACM / your CA) -------------------------------
    ssl_certificate     /etc/letsencrypt/live/i3.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/i3.example.org/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # --- Security headers --------------------------------------------------
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options    "nosniff"         always;
    add_header X-Frame-Options           "DENY"            always;
    add_header Referrer-Policy           "no-referrer"     always;

    # --- Bounds ------------------------------------------------------------
    client_max_body_size 128k;
    proxy_read_timeout   75s;
    proxy_send_timeout   75s;

    # --- REST --------------------------------------------------------------
    location / {
        proxy_pass         http://i3;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }

    # --- WebSocket ---------------------------------------------------------
    location /ws/ {
        proxy_pass           http://i3;
        proxy_http_version   1.1;
        proxy_set_header     Upgrade   $http_upgrade;
        proxy_set_header     Connection "upgrade";
        proxy_set_header     Host      $host;
        proxy_set_header     Origin    $http_origin;
        proxy_read_timeout   3600s;
    }

    # --- Deny demo endpoints in production --------------------------------
    location ~* ^/demo/ { return 404; }
}

server {
    listen 80;
    server_name i3.example.org;
    return 301 https://$host$request_uri;
}
```

!!! warning "Block `/demo/*` at the edge"
    The server already gates `/demo/*` behind `I3_DEMO_MODE`, but the
    belt-and-braces approach is to deny at the proxy too.

## 6. Production-hardening checklist { #hardening }

- [x] Run as non-root with `NoNewPrivileges` and tight `ReadWritePaths`.
- [x] `I3_DEMO_MODE` is **not** set.
- [x] `I3_CORS_ORIGINS` is a concrete list — never `*`.
- [x] TLS with HSTS + OCSP stapling.
- [x] `Strict-Transport-Security`, `X-Content-Type-Options`,
      `X-Frame-Options`, `Referrer-Policy` set.
- [x] OTel exporter points at your collector (not the default localhost).
- [x] SQLite file is on an encrypted volume (defence in depth over Fernet).
- [x] Backups of `/var/lib/i3/data/` scheduled (profiles.sqlite).
- [x] Log aggregation ingesting the JSON access log.
- [x] Rate limiting validated (send 650 req/min; expect 429s).

## 7. Zero-downtime rollout { #rollout }

I³ is a single-process worker (pipeline state is in-process). A
zero-downtime rollout uses **blue/green** with a shared SQLite file:

```bash
# 1. Start green on an alternate port.
sudo systemctl start i3-green.service      # listens on 127.0.0.1:8001

# 2. Flip the nginx upstream to point at 8001.
sudo nginx -s reload

# 3. Drain blue (wait for open WS to close).
sudo systemctl stop i3.service

# 4. Promote green to "blue" on the next deploy.
```

!!! tip "Sessions are ephemeral"
    WebSocket sessions do not survive rollouts. The client JS has
    exponential-backoff reconnect; a typical drop lasts ~2 s.

## 8. Scaling beyond one node { #scaling }

- **Horizontal**: multiple replicas behind a sticky-session load balancer.
  SQLite per node, plus a periodic sync to an S3-compatible store for
  disaster recovery.
- **Vertical**: the SLM fits comfortably on 2 GB RAM. Larger memory buys
  more concurrent WS connections (each session holds a ring buffer).
- **GPU**: the SLM generates on CPU in <200 ms P95. A GPU buys throughput
  for batch scoring but does not help a single-user UX.

See [Kubernetes](kubernetes.md) for multi-replica patterns.

## 9. Further reading { #further }

- [Observability](observability.md) — what to watch post-deploy.
- [Runbook](runbook.md) — incident response.
- [Troubleshooting](troubleshooting.md) — known symptoms and fixes.
