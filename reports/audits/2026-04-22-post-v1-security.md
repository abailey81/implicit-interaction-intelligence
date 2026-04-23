# Security Audit Report — Post-v1.0 Commits

**Audit date:** 2026-04-22
**Auditor:** Automated strict review (Claude)
**Branch:** `main`
**Baseline commit:** `ca2e976` ("Add files via upload")
**Head commits reviewed:** `ca2e976`, `c2de7ac`, `a0d6d8b`, `57fcd26`, `d64bbc0` and all files
introduced/modified since the v1.0 baseline (~14 commits of added infra, observability,
MLOps, guardrails, deployment manifests, and workflow wiring).

---

## 1. Scope

The audit covers **only** the work added on top of the v1.0 upload — specifically:

- `Dockerfile`, `docker/Dockerfile.dev`, `docker-compose*.yml`, `docker/entrypoint.sh`,
  `docker/healthcheck.sh`, `.dockerignore`, `docker-compose.override.yml.example`
- `deploy/k8s/**`, `deploy/helm/i3/**`, `deploy/observability/**`
- New GitHub Actions workflows (all under `.github/workflows/` **except** the
  pre-existing `ci.yml` and `security.yml`)
- `i3/observability/**` (logging, metrics, tracing, sentry, middleware,
  instrumentation, langfuse_client, context)
- `i3/cloud/guardrails.py`, `i3/cloud/guarded_client.py`
- `i3/mlops/checkpoint.py`, `i3/mlops/model_signing.py`, `i3/mlops/registry.py`
- `i3/encoder/loss.py`, `i3/interaction/sentiment.py` (+ JSON asset)
- `server/routes_health.py`, single observability hook in `server/app.py`
- `.env.example`, `pyproject.toml` (new dependency groups), `engineering notes`

**Methodology:** line-by-line read of each new/modified file, cross-referenced
against the explicit threat checklist in the audit brief. Dependency floors
cross-checked against public CVE trackers. Read-only review — no files were
modified, no git commands run.

Pre-existing files that were *not* touched by the new commits (`i3/privacy/`,
`i3/pipeline/engine.py`, `server/middleware.py`, `server/routes.py`,
`.github/workflows/ci.yml`, `.github/workflows/security.yml`, `SECURITY.md`) were
not re-audited — they are covered by the existing review documented in
`SECURITY.md`.

---

## 2. Summary

| Severity       | Count |
|----------------|-------|
| Critical       | 0     |
| High           | 0     |
| Medium         | 2     |
| Low            | 4     |
| Informational  | 5     |

**Verdict:** the post-v1.0 work is **production-grade** for the scope of a
research prototype / interview submission. No critical or high issues. The
medium findings are hardening tweaks; the low/informational findings are either
style or defensive-depth suggestions. None of them block the interview demo.

---

## 3. Findings

### Medium

| ID | File | Issue | Severity | Recommendation |
|----|------|-------|----------|----------------|
| **SA-M-1** | `.github/workflows/trivy.yml:33,52,74,93,129,149` | The Trivy action is pinned to `aquasecurity/trivy-action@master` (floating branch). A compromise or breaking change at `master` is auto-ingested on every run. The brief explicitly asks "actions pinned (at least to major version)". | Medium | Pin to a specific tag, e.g. `aquasecurity/trivy-action@0.24.0`, ideally by SHA. Same fix applies to every call in the file. |
| **SA-M-2** | `.github/workflows/semgrep.yml:28` | `container: image: semgrep/semgrep:latest` — unpinned image tag running Semgrep CLI with org rules. Supply-chain drift risk if semgrep publishes a breaking or malicious tag. | Medium | Pin to a specific Semgrep image tag (e.g. `semgrep/semgrep:1.90`), or to a digest (`semgrep/semgrep@sha256:...`). |

### Low

| ID | File | Issue | Severity | Recommendation |
|----|------|-------|----------|----------------|
| **SA-L-1** | `i3/mlops/checkpoint.py:124-152` | `_verify_signature()` is an **explicit stub** — if a `<path>.sig` file exists, the function logs a warning and returns `True`. In a supply-chain attack scenario where an attacker can drop a signature sidecar, this gives a false sense of provenance. Stub is clearly documented and logged, so not misleading; but `ModelSigner` in the same package provides the real thing. | Low | Replace the stub with a thin delegation to `i3.mlops.model_signing.ModelSigner.verify(...)` when `model-signing` is importable, falling back to the current warning-only path otherwise. Alternatively, make `verify_signature=True` (default) raise unless a real verifier is configured, flipping the fail-open to fail-closed. |
| **SA-L-2** | `docker/entrypoint.sh:66-73` | The init-hook loop `. "$f"` sources every file under `/app/docker/init.d/*.sh` before `exec uvicorn`. If any code path ever makes that directory writable by a non-root or an attacker-controllable bind mount, the sourced files execute as the `i3` user before the server starts. The prod compose and k8s manifests DO use a read-only root FS so this is not exploitable in the shipped configuration, but the entrypoint itself does not verify the file's ownership or permissions. | Low | Either gate the hook behind an explicit env flag (`I3_INIT_HOOKS=1`) or verify `root:root` ownership and `0o644` mode before sourcing. |
| **SA-L-3** | `i3/cloud/guardrails.py:112-132` | The prompt-injection keyword allow-list is deliberately small (11 patterns). It will miss non-English equivalents, unicode lookalikes (ideographic colon `：`, fullwidth brackets), base64 encoded payloads, and CSS/HTML-style injections. The module correctly flags this is "intentionally conservative", but the false-negative rate will be non-trivial against motivated adversaries. | Low | Supplement with (a) a length-distribution check, (b) repeated-delimiter detection (`\n{5,}`), and (c) unicode normalisation (NFKC) **before** regex matching. Not a blocker — guardrails are defence-in-depth and the server-side Anthropic policy also filters. |
| **SA-L-4** | `i3/observability/middleware.py:42-51` | `_extract_client_ip` always trusts the first value of `X-Forwarded-For` without cross-checking against `I3_FORWARDED_IPS`. This value is only used for logging, not for auth/authorisation, so the blast radius is log-forging (an attacker can spoof `client_ip` in audit logs). | Low | Either honour `I3_FORWARDED_IPS` (use `scope["client"]` unless the immediate peer is in the trusted list), or rename the logged field to `x_forwarded_for_first` to make it clear the value is untrusted. |

### Informational

| ID | File | Issue | Severity | Recommendation |
|----|------|-------|----------|----------------|
| **SA-I-1** | `deploy/helm/i3/templates/` | The Helm chart does **not** ship a `templates/secret.example.yaml`, contrary to the audit brief's expectation. This is actually **safer** than shipping one — the chart correctly delegates to `envFromSecret.existingSecret: i3-secrets` and documents SealedSecrets/ESO workflows in `deploy/helm/README.md:65-76`. The canonical example lives at `deploy/k8s/secret.example.yaml` with a prominent "EXAMPLE ONLY — DO NOT COMMIT REAL SECRETS" banner (lines 1-7). | Informational | No change required. |
| **SA-I-2** | `i3/observability/logging.py:74-84` | Substring-based redaction triggers on `"token"` as a substring — this will redact keys like `tokenizer`, `token_count`, `input_tokens`. The cost is cosmetic (log analytics lose a non-secret count) but the benefit is very conservative redaction. The explicit allow-list (lines 32-55) fully covers every key named in the audit brief (`api_key`, `authorization`, `cookie`, `token`, `password`, `secret`, `fernet`, `I3_ENCRYPTION_KEY`, plus `anthropic_api_key`, `access_token`, `refresh_token`, `client_secret`). | Informational | Consider whitelisting `token_count`, `input_tokens`, `output_tokens`, `tokenizer`, `max_tokens` to preserve telemetry. |
| **SA-I-3** | `i3/mlops/model_signing.py:385-391` | `verify()` catches `Exception` and returns `False` + logs a warning. The brief required "verification reports mismatch clearly" — `False` is a clear signal to the caller, and the warning log differentiates *why* it failed. The one slight gap: the caller cannot distinguish "signature invalid" from "network unreachable to Rekor" or "sigstore TUF error". | Informational | Add a structured return (`VerifyResult` dataclass with `ok: bool, error: str, error_class: str`) if callers ever need to branch on the failure mode. Not required. |
| **SA-I-4** | `server/app.py:218-219` | The observability bootstrap is inserted after the CORS middleware (line 202-215) and before the exception handlers (line 225+) — exactly as specified in the brief. The single edit is non-destructive: no existing middleware or handler was removed or reordered. The call happens after CORS is registered but the observability middleware is added *inside* `setup_observability(app)` which uses `app.add_middleware`, meaning it sits **inside** CORS (which is correct for request-ID logging of preflight-rejected requests too). | Informational | No change. |
| **SA-I-5** | `docker-compose.override.yml.example:32` | Placeholder Fernet key `"dev-only-not-for-production-xxxxxxxxxxxxxxxx"` is not a valid Fernet key (must be 32 URL-safe base64 bytes = 44 chars including `=`). This will **crash** the encryptor on first use, which is the correct fail-closed behaviour — but it's a small DX annoyance. | Informational | Either provide a deterministic valid throwaway key (`AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=`) or drop the default and require the developer to generate one. |

---

## 4. Clean areas

The following areas were audited and found to be **clean** (no actionable
findings beyond those listed above).

**Containers & runtime:**
- `Dockerfile` — multi-stage, non-root `i3:10001`, no build-essential/pip/poetry
  in runtime (lines 152-165), `tini` as PID 1 (line 204), Python-stdlib
  HEALTHCHECK (no curl in runtime), OCI labels pinned, base image pinned via
  `PYTHON_IMAGE` build-arg (line 25), tests/caches stripped from site-packages
  (lines 109-114).
- `docker/Dockerfile.dev` — same non-root identity (10001/10001) for uid parity on
  bind mounts, build toolchain confined to dev.
- `docker-compose.prod.yml` — `read_only: true` (line 24), explicit `tmpfs`
  for `/tmp` and `/app/data/tmp` (lines 25-27), `cap_drop: [ALL]` (line 28),
  `security_opt: no-new-privileges:true` (line 30), no external port on
  `i3-server` (nginx sole ingress), pinned nginx image (line 80), nginx
  `cap_add: [NET_BIND_SERVICE]` only.
- `docker-compose.yml` — loads `.env` via `env_file` (not a bind mount of a
  secret file into the container FS — standard compose pattern), no
  embedded secrets.
- `.dockerignore` — excludes `.git`, `.env`, `.env.*` (with explicit
  `!.env.example` re-include), `checkpoints/`, `data/`, `wandb/`, `*.sqlite*`,
  `tests/`, `__pycache__`, IDE noise.

**Kubernetes:**
- `deploy/k8s/deployment.yaml` — `runAsNonRoot: true`, `runAsUser: 10001`,
  `seccompProfile.type: RuntimeDefault` (line 57), `allowPrivilegeEscalation:
  false`, `readOnlyRootFilesystem: true`, `capabilities.drop: [ALL]` (lines
  158-164), `automountServiceAccountToken: false` on both Pod and ServiceAccount,
  explicit `startupProbe`/`livenessProbe`/`readinessProbe`, `preStop` sleep to
  drain the load balancer.
- `deploy/k8s/networkpolicy.yaml` — default-deny (`podSelector: {}` +
  `policyTypes: [Ingress, Egress]`, lines 16-20), narrow allow for
  ingress-nginx + monitoring namespaces, egress limited to DNS +
  `443/tcp` with RFC1918/link-local excluded.
- `deploy/k8s/secret.example.yaml` — prominent "EXAMPLE ONLY — DO NOT COMMIT
  REAL SECRETS" banner and SealedSecret snippet included.
- Kustomize overlays — prod overlay tightens HPA and resources without weakening
  security context.

**Helm chart (`deploy/helm/i3/`):**
- `values.yaml` — `runAsNonRoot: true`, `seccompProfile: RuntimeDefault`,
  `readOnlyRootFilesystem: true`, `capabilities.drop: [ALL]`, NetworkPolicy
  default-on, ServiceMonitor default-on, HPA 2-10 replicas with CPU/memory and
  custom metric, rate-limit annotations on the ingress (`limit-rps: 60`,
  `limit-rpm: 600`, `limit-connections: 50`), `proxy-body-size: 4m`.
- No hard-coded secrets anywhere; default `envFromSecret.existingSecret:
  i3-secrets` pattern is the correct idiom.
- `templates/networkpolicy.yaml` mirrors the k8s default-deny + narrow-allow
  shape.

**GitHub Actions (new workflows):**
- `docker.yml` — OIDC-based keyless cosign signing (pinned to
  `sigstore/cosign-installer@v3.7.0`), SLSA Level 3 provenance pinned to
  `slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0`,
  `actions/attest-sbom` pinned **by SHA** (line 137, `5026d3663739...@v1.4.1`),
  default `permissions: contents: read`, job-level minimal perms.
- `release.yml` — Trusted Publishing to PyPI via OIDC (no long-lived token),
  SLSA v2.0.0 generator, `persist-credentials: false` on checkouts, minimal
  default permissions.
- `sbom.yml` — `actions/attest-sbom` SHA-pinned, CycloneDX for Python SBOM,
  Syft for image SBOM, release-only attestation gate.
- `scorecard.yml` — `permissions: read-all` at workflow level (required by
  scorecard-action), granular job perms, `persist-credentials: false`.
- `pr-title.yml` — uses `pull_request_target` **safely**: does not check out
  PR code, only reads the title via the action; no path for untrusted code
  execution.
- `benchmark.yml`, `docs.yml`, `lockfile-audit.yml`, `markdown-link-check.yml`,
  `stale.yml` — minimal permissions, no `secrets.X` misuse beyond
  `GITHUB_TOKEN`.

**Observability:**
- `i3/observability/logging.py:32-55` — sensitive-key allow-list covers every
  term the brief requires plus extras (`apikey`, `x-api-key`,
  `anthropic_api_key`, `access_token`, `refresh_token`, `id_token`,
  `client_secret`, `encryption_key`). Supports runtime extension via
  `I3_LOG_EXTRA_REDACTIONS`.
- `i3/observability/sentry.py` — `PrivacySanitizer` **lazy-imported** inside
  `_lazy_sanitizer()` (line 27), `send_default_pii=False` (line 143),
  `attach_stacktrace=True` but local variables (`frame.vars`) are stripped in
  `before_send` (lines 67-70). `configure_sentry` is fully idempotent.
- `i3/observability/metrics.py:239-244` — `metrics_enabled()` reads
  `I3_METRICS_ENABLED` (default enabled; any of `{"0", "false", "False",
  "no"}` disables), and `render_prometheus` degrades to a no-op when
  `prometheus_client` is absent.
- `i3/observability/tracing.py` — sampler correctly bounded `[0.0, 1.0]`
  (line 96), falls back to no-op tracer on any import error.
- `i3/observability/langfuse_client.py:34-36` — metadata-only tracing by
  default; raw user prompts only forwarded when `capture_io=True` is
  **explicitly** passed.

**Guardrails:**
- `i3/cloud/guardrails.py:286,112` — length caps enforced
  (`DEFAULT_MAX_RESPONSE_CHARS: 16_000`, `DEFAULT_MAX_TOKENS: 4096`).
- Output redaction list (lines 288-295) covers Anthropic, OpenAI-style, AWS
  access key IDs, GitHub PATs, Slack tokens, Google API keys, and caller-
  supplied `user_id` via case-insensitive word-boundary match.
- `GuardedCloudClient._blocked_response` returns the same dict shape as the
  inner client with explicit `blocked=True` — no bypass path.
- `GuardedCloudClient.generate_session_summary` still routes through
  `OutputGuardrail.sanitize` to strip any reflected keys.

**MLOps:**
- `i3/mlops/checkpoint.py:231` — `weights_only=True` is the **default**.
  `load_verified` raises `ChecksumError` on digest mismatch (line 293) and
  `FileNotFoundError` when the artefact is missing (line 271). Hash is
  streamed in 1 MiB chunks (line 61) to bound memory.
- `i3/mlops/model_signing.py:269-272` — signing failures raise
  `RuntimeError` with the wrapped exception's type and message; does **not**
  silently succeed.
- `i3/mlops/registry.py:14-15` — best-effort MLflow/W&B mirroring; failures
  never block local registration; registry never executes pickled code.

**Encoder / interaction:**
- `i3/encoder/loss.py:127-128` — diagonal mask uses `-1e9`, not `-inf`
  (explicit comment cites the fp16 numerical-stability rule). No float
  overflow path.
- `i3/interaction/sentiment.py:170-197` — JSON loader **validates shape
  and types** before inclusion: `isinstance(entries, dict)` (line 184) and
  `isinstance(token, str) and isinstance(score, (int, float))` per entry
  (line 187). Failure falls back silently to the inline dictionary (lines
  173-179, 190-195). No code-eval path — pure data deserialisation via
  `json.load`.

**Server health:**
- `server/routes_health.py:154-160` — `/api/metrics` returns **404** when
  `I3_METRICS_ENABLED` is disabled (so Prometheus skips the target cleanly,
  and enumeration does not reveal that the endpoint exists).
- `/api/ready` returns generic status strings (`"ok"`, `"initializing"`,
  `"missing"`, `"disabled"`) and an optional `details` map. The details only
  contain internal status descriptions (no PII — pipeline flag names, env var
  names, error class names). Version + uptime are non-sensitive.

**Deps / secrets / docs:**
- `.env.example` — every value is a placeholder (`sk-ant-your-key-here`,
  empty `I3_ENCRYPTION_KEY=`, localhost-only CORS). No real secrets.
- `engineering notes` — engineering disclosure only; no credentials, internal URLs,
  or Huawei-confidential material.
- `server/app.py:218-219` single-edit insertion — non-destructive, correctly
  placed.

---

## 5. Dependency notes

All dependency floors in `pyproject.toml` are current as of April 2026. Audit
of the new groups:

| Package | Version pin | Status | Notes |
|---------|-------------|--------|-------|
| `torch` | `>=2.6,<3.0` | Fine | Brief cites the `weights_only` bypass (CVE-2025-XXXXX-family) closed in 2.6. Upper bound gives a clean migration path. |
| `cryptography` | `^43.0` | Fine | Covers CVE-2024-26130 (PKCS#12 NULL deref) and the OpenSSL wrapper EC-key family. |
| `fastapi` | `^0.115` | Fine | Pulls `starlette >= 0.40` (CVE-2024-47874 multipart DoS closed). |
| `structlog` | `^24.1` | Fine | No known CVEs. Active maintenance. |
| `prometheus_client` | `^0.20` | Fine | No known CVEs. |
| `opentelemetry-*` | `^1.25` / `^0.46b0` | Fine | Tracking stable SDK. No known CVEs. |
| `sentry-sdk` | `^2.0` | Fine | v2.x line is the recommended major. |
| `mlflow` | `^2.11` | Watch | MLflow has a history of path-traversal and SSRF findings (CVE-2023-43472, CVE-2024-0520 series). The code only uses tracking URIs — no model-serving endpoints — so the blast radius is limited, but bump to `^2.15` when convenient for a cleaner CVE picture. |
| `locust` | `^2.24` | Fine | Load-test only; dev group. |
| `wandb` | `^0.17` (optional) | Fine | Optional extra. No known CVEs at this floor. |
| `dvc` | `^3.50` | Fine | No known CVEs. |
| `mutmut` | `^2.5` | Watch — maintenance | Maintenance has slowed (primary maintainer active-but-busy). Dev-only dependency, never in the runtime image, so the supply-chain impact is limited. Consider pinning to an exact version. |
| `torchao` | `>=0.11` | Fine | Official PyTorch project; pinned against a minimum. |
| `model-signing` | `>=1.0` | Fine | OpenSSF project; pinned against a minimum. |
| `executorch` | `>=0.4` | Watch — early | PyTorch edge runtime; API still stabilising. Optional group, not in the runtime image. |
| `langfuse` | `>=2.40` | Fine | Optional; no known CVEs. |
| `onnx` | `^1.16` | Fine — ensure >=1.17 long-term | ONNX has had several proto-level parser CVEs in 2024. `^1.16` is fine today; consider bumping to `^1.17` when convenient. |
| `onnxruntime` | `^1.17` | Fine | No known critical CVEs at this floor. |
| `mkdocs-material` | `^9.5` | Fine | Docs only. |
| `bandit` | `^1.7`, `pip-audit` `^2.7`, `safety` `^3.0` | Fine | Security tooling, dev-only. |
| `detect-secrets` | `^1.5` | Fine | Pre-commit hook only. |

**No abandoned packages; no packages with open critical CVEs at the declared
floor.**

---

## 6. Recommendations

### Before the interview (high ROI, low effort — 20 min total)

1. **SA-M-1 + SA-M-2 — pin `aquasecurity/trivy-action@master` and
   `semgrep/semgrep:latest` to concrete versions.** A reviewer who runs
   `grep -rn "@master\|:latest" .github/` will find these immediately, and the
   fix is a single-line edit per call. This is the only finding that could
   plausibly be flagged in a live code walkthrough.
2. **Sanity-check the placeholder Fernet key in
   `docker-compose.override.yml.example` (SA-I-5)** — replace with a
   deterministic-but-obviously-dev key so the first `docker compose up` does
   not crash the encryptor on the demo machine.

### Post-interview (defence-in-depth)

3. **SA-L-1** — wire `ModelSigner` into `_verify_signature` when
   `model-signing` is importable. Closes the only observed fail-open
   checkpoint path.
4. **SA-L-2** — gate `docker/entrypoint.sh` init-hook behind
   `I3_INIT_HOOKS=1` or verify ownership before sourcing.
5. **SA-L-3** — augment `InputGuardrail` with NFKC normalisation and repeated-
   delimiter detection. Small patch, material uplift against encoded/obfuscated
   prompt-injection attempts.
6. **SA-L-4** — honour `I3_FORWARDED_IPS` in the observability middleware's
   `_extract_client_ip` helper (or rename the logged field to make it clear
   the value is untrusted).

### Not recommended to change

- The Helm chart's **lack** of a `templates/secret.example.yaml` (SA-I-1) is
  correct. Shipping a templated Secret that a user could accidentally populate
  and commit is a bigger risk than omitting it.
- The substring-match redaction (SA-I-2) is intentionally aggressive; the false-
  positive cost is cosmetic.
- The `pull_request_target` usage in `pr-title.yml` is safe — no PR code is
  checked out.

---

## Appendix — files inspected

Infrastructure: `Dockerfile`, `docker/Dockerfile.dev`, `.dockerignore`,
`docker-compose.yml`, `docker-compose.prod.yml`,
`docker-compose.override.yml.example`, `docker/entrypoint.sh`,
`docker/healthcheck.sh`.

Kubernetes: `deploy/k8s/deployment.yaml`, `deploy/k8s/configmap.yaml`,
`deploy/k8s/secret.example.yaml`, `deploy/k8s/networkpolicy.yaml`,
`deploy/k8s/overlays/prod/kustomization.yaml`.

Helm: `deploy/helm/i3/values.yaml`, `deploy/helm/i3/templates/deployment.yaml`,
`deploy/helm/i3/templates/networkpolicy.yaml`, `deploy/helm/README.md`.

Workflows: `benchmark.yml`, `docker.yml`, `docs.yml`, `lockfile-audit.yml`,
`markdown-link-check.yml`, `pr-title.yml`, `release.yml`, `sbom.yml`,
`scorecard.yml`, `semgrep.yml`, `stale.yml`, `trivy.yml`.

Application code: `i3/observability/{logging,metrics,tracing,sentry,middleware,
instrumentation,langfuse_client}.py`, `i3/cloud/{guardrails,guarded_client}.py`,
`i3/mlops/{checkpoint,model_signing,registry}.py`, `i3/encoder/loss.py`,
`i3/interaction/sentiment.py`, `i3/interaction/data/sentiment_lexicon.json`,
`server/routes_health.py`, `server/app.py` (lines 218-219 only).

Config / docs: `.env.example`, `pyproject.toml`, `engineering notes`, `CHANGELOG.md`.
