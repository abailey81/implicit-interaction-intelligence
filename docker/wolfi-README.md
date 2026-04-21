# Wolfi distroless container — Implicit Interaction Intelligence (I³)

This directory documents the **Chainguard Wolfi** variant of the I³ production
container, built from `Dockerfile.wolfi` at the repository root. It is an
opt-in, zero-known-CVE alternative to the default Debian-slim image.

## Why Wolfi?

Wolfi is an **undistribution** authored by Chainguard: it is an OCI-native
Linux userland built from scratch specifically for containers. Unlike Debian
or Alpine, Wolfi:

1. **Tracks upstream daily** — the package build pipeline re-emits every
   package from source the moment a new tag lands. There is no 6-to-18-month
   "stable release" lag that leaves CVEs sitting in the distro.
2. **Has no legacy baggage** — no `sysvinit`, no dbus, no `setuid` shims.
   Images routinely contain fewer than 15 executables.
3. **Ships with verifiable provenance** — every package is signed with
   Sigstore, every image ships with an in-toto attestation and SBOM, and
   every build is reproducible under SLSA L3.
4. **Is distroless-first** — the `:latest` tag has **no shell, no package
   manager, and no `libc` surface area** beyond what your binary needs.

The end state: when `trivy`, `grype`, or `snyk` scans the image, they report
**zero known CVEs** at build time, and Chainguard's pipeline delivers a new
rebuild within ~hours of any CVE being disclosed upstream.

## CVE comparison — typical Python production images

| Base image                            | Size (uncompressed) | High/Critical CVEs (Trivy, 2026-04) |
|---------------------------------------|---------------------|-------------------------------------|
| `python:3.11`                         | ~1 020 MB           | 120–180                             |
| `python:3.11-slim`                    | ~150 MB             | 30–60                               |
| `python:3.11-alpine`                  | ~60 MB              | 10–25                               |
| `cgr.dev/chainguard/python:latest`    | ~55 MB              | **0**                               |
| `cgr.dev/chainguard/python:latest-dev`| ~180 MB             | **0**                               |

The Debian-slim numbers fluctuate hour-by-hour as Debian's security team
triages backports; Alpine's are usually lower but the musl toolchain frequently
causes issues with the `manylinux` wheels torch publishes. Wolfi solves both
problems: glibc-compatible, zero CVEs, and a clean SBOM.

## How to build

From the repository root:

```bash
docker build -f Dockerfile.wolfi -t i3:wolfi .
```

For a multi-arch build that works on both `x86_64` and `arm64`:

```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f Dockerfile.wolfi \
    -t ghcr.io/i3/i3:wolfi-$(git rev-parse --short HEAD) \
    --push .
```

The build is a two-stage affair:

1. **Builder stage** uses `cgr.dev/chainguard/python:latest-dev` — this
   image *does* have `apk`, `pip`, `uv`, and a shell. It compiles the
   requirements lockfile via `uv pip compile --universal`, installs torch
   (CPU index) and the project wheels into a venv at `/opt/venv`.
2. **Runtime stage** uses the distroless `cgr.dev/chainguard/python:latest`
   image. It has **no shell**. We copy only `/opt/venv`, the app source, and
   configs, then exec `python -m uvicorn` directly.

Because the runtime has no shell, there is **no `CMD`** — the `ENTRYPOINT` is
an exec-form `python -m uvicorn …` invocation. You cannot `docker exec -it
<container> sh` into it; that is the point.

## How to push

```bash
docker tag i3:wolfi ghcr.io/i3/i3:wolfi-latest
docker push ghcr.io/i3/i3:wolfi-latest
```

CI publishes the Wolfi image to the same registry as the default build under
the `-wolfi` suffix. Consumers opt-in by pinning that tag in their Helm
values or docker-compose overrides. See `docker-compose.prod.yml` for the
default (Debian-slim) wiring.

## How to scan

```bash
# Trivy — CVE scan against the distro metadata and the SBOM.
trivy image i3:wolfi --severity HIGH,CRITICAL

# grype — alternative scanner, useful for cross-validation.
grype i3:wolfi --only-fixed

# syft — extract the SBOM for downstream tooling.
syft i3:wolfi -o spdx-json > sbom.wolfi.spdx.json
```

Expected scan output: **0 high, 0 critical** on first build. Chainguard
publishes the image SBOM alongside each tag, and our `sbom.yml` CI workflow
attests to it with cosign.

## How to run

Locally, for a smoke test:

```bash
docker run --rm -p 8000:8000 \
    -e I3_MODEL_PATH=/app/configs/model.pt \
    -e I3_LOG_LEVEL=info \
    i3:wolfi
```

In production the container expects the standard I³ environment contract —
see `SECURITY.md` and `docs/operations/` for the full variable list.

## Known trade-offs

- **No debugging shell in production.** If you need to inspect a running
  container, build the `-dev` variant or use `kubectl debug --image=` with a
  debug sidecar. This is by design — attackers can't drop into a shell that
  doesn't exist.
- **`apt-get install` recipes don't apply.** If a new native dependency is
  required (e.g. an audio codec), add it via the Wolfi `apk add` path in the
  builder stage and copy the resulting shared objects into `/opt/venv/lib`.
- **Image tags rotate.** Chainguard rebuilds `:latest` whenever upstream
  Python or a linked OpenSSL lands a fix. For a pin-stable deployment use a
  hash-pinned tag (`@sha256:…`) instead of a rolling tag.

## Further reading

- Chainguard Academy — *Working with the Python distroless images*.
- Chainguard — *The zero-CVE image story*, 2025.
- Reproducible Builds — `https://reproducible-builds.org`.
- SLSA framework, level 3 — `https://slsa.dev/spec/v1.0/levels#build-l3`.
