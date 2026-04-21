#!/bin/sh
# =============================================================================
#  Container HEALTHCHECK for the I³ FastAPI server.
# -----------------------------------------------------------------------------
#  Uses Python's standard library (urllib) so we do NOT have to ship a curl
#  binary in the runtime image. Any non-2xx response — or a timeout — results
#  in a non-zero exit, which Docker interprets as "unhealthy".
# =============================================================================
set -eu

: "${I3_HEALTH_HOST:=127.0.0.1}"
: "${I3_HEALTH_PORT:=${I3_PORT:-8000}}"
: "${I3_HEALTH_PATH:=/api/health}"
: "${I3_HEALTH_TIMEOUT:=3}"

URL="http://${I3_HEALTH_HOST}:${I3_HEALTH_PORT}${I3_HEALTH_PATH}"

exec python - "$URL" "$I3_HEALTH_TIMEOUT" <<'PY'
import sys
import urllib.error
import urllib.request

url, timeout = sys.argv[1], float(sys.argv[2])
req = urllib.request.Request(url, headers={"User-Agent": "i3-healthcheck/1.0"})

try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (loopback only)
        if 200 <= resp.status < 300:
            sys.exit(0)
        print(f"unhealthy: HTTP {resp.status}", file=sys.stderr)
        sys.exit(1)
except urllib.error.HTTPError as exc:
    print(f"unhealthy: HTTP {exc.code}", file=sys.stderr)
    sys.exit(1)
except (urllib.error.URLError, TimeoutError, OSError) as exc:
    print(f"unhealthy: {exc}", file=sys.stderr)
    sys.exit(1)
PY
