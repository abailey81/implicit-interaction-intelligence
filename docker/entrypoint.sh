#!/bin/sh
# =============================================================================
#  I³ container entrypoint
# -----------------------------------------------------------------------------
#  * POSIX sh (no bashisms — the runtime image has no bash)
#  * `set -eu` : fail fast on errors and undefined variables
#  * `exec`    : replaces this shell with uvicorn so that tini (PID 1) forwards
#                SIGTERM / SIGINT directly to the Python process. Without
#                `exec`, Ctrl-C / `docker stop` would hang for the 10-second
#                grace period before Docker SIGKILLs the container.
# =============================================================================
set -eu

# ---------------------------------------------------------------------------
#  Configuration defaults (override via environment / docker-compose).
# ---------------------------------------------------------------------------
: "${I3_PORT:=8000}"
: "${I3_HOST:=0.0.0.0}"
: "${I3_WORKERS:=1}"
: "${I3_FORWARDED_IPS:=127.0.0.1}"
: "${I3_LOG_LEVEL:=info}"
: "${I3_APP_MODULE:=server.app:app}"

# ---------------------------------------------------------------------------
#  Pre-flight: encryption key MUST be set before the app boots.
#  The model-encryptor subsystem refuses to load checkpoints without it.
# ---------------------------------------------------------------------------
if [ -z "${I3_ENCRYPTION_KEY:-}" ] && [ -z "${I3_ENCRYPTION_KEY_FILE:-}" ]; then
    echo "[entrypoint] FATAL: neither I3_ENCRYPTION_KEY nor I3_ENCRYPTION_KEY_FILE is set." >&2
    echo "[entrypoint]        Refusing to start — model artefacts would be loaded unencrypted." >&2
    exit 78  # EX_CONFIG
fi

# If a *_FILE variant is provided (Docker-secret style), read it once and
# export the real value so downstream code can use it uniformly.
if [ -z "${I3_ENCRYPTION_KEY:-}" ] && [ -n "${I3_ENCRYPTION_KEY_FILE:-}" ]; then
    if [ ! -r "${I3_ENCRYPTION_KEY_FILE}" ]; then
        echo "[entrypoint] FATAL: I3_ENCRYPTION_KEY_FILE=${I3_ENCRYPTION_KEY_FILE} is not readable." >&2
        exit 78
    fi
    I3_ENCRYPTION_KEY="$(cat "${I3_ENCRYPTION_KEY_FILE}")"
    export I3_ENCRYPTION_KEY
fi

# ---------------------------------------------------------------------------
#  Banner — emitted to stdout so `docker logs` shows the runtime profile.
# ---------------------------------------------------------------------------
cat <<BANNER
╔══════════════════════════════════════════════════════════════════════════╗
║       Implicit Interaction Intelligence  (I³)  —  container runtime      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  app module     : ${I3_APP_MODULE}
║  bind address   : ${I3_HOST}:${I3_PORT}
║  workers        : ${I3_WORKERS}
║  forwarded-ips  : ${I3_FORWARDED_IPS}
║  log level      : ${I3_LOG_LEVEL}
║  python         : $(python --version 2>&1)
║  pid 1          : tini  (signal-forwarding reaper)
╚══════════════════════════════════════════════════════════════════════════╝
BANNER

# ---------------------------------------------------------------------------
#  Optional bootstrap hook — users can bind-mount additional init scripts at
#  /app/docker/init.d/*.sh and they will run before uvicorn starts.
# ---------------------------------------------------------------------------
if [ -d /app/docker/init.d ]; then
    for f in /app/docker/init.d/*.sh; do
        [ -r "$f" ] || continue
        echo "[entrypoint] running $f"
        # shellcheck disable=SC1090
        . "$f"
    done
fi

# ---------------------------------------------------------------------------
#  Launch uvicorn via `exec` so signal handling works correctly under tini.
# ---------------------------------------------------------------------------
exec uvicorn "${I3_APP_MODULE}" \
    --host "${I3_HOST}" \
    --port "${I3_PORT}" \
    --workers "${I3_WORKERS}" \
    --log-level "${I3_LOG_LEVEL}" \
    --proxy-headers \
    --forwarded-allow-ips "${I3_FORWARDED_IPS}" \
    "$@"
