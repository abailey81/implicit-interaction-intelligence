# Supply Chain Security

This document describes how Implicit Interaction Intelligence (I3) secures its
software supply chain — from source, through build, to consumer verification —
and how we respond when a vulnerability is reported.

For the SLSA-specific justification and verification steps, see `docs/security/slsa.md`.
For private vulnerability reporting, see `SECURITY.md`.

---

## 1. Principles

1. **Every artifact we publish is signed, reproducible, and has provenance.**
2. **Every artifact we consume is version-pinned, hash-verified, and continuously
   scanned.**
3. **No single human is a cryptographic root of trust.** Signing keys are
   ephemeral, issued by Sigstore Fulcio against GitHub OIDC identities.
4. **Transparency is mandatory.** Signatures land in Rekor; SBOMs ship with
   every release; CVE responses are public.

---

## 2. Sources of trust

| Layer                  | Trust anchor                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| Source code            | Protected branch `main`, required reviews, CODEOWNERS, signed commits (encouraged).                        |
| Build platform         | GitHub-hosted runners + SLSA reusable workflows pinned by tag.                                             |
| Signing                | Sigstore (Fulcio CA + Rekor transparency log) using short-lived OIDC-bound keys.                           |
| Artifact provenance    | `slsa-framework/slsa-github-generator` v2.0.0.                                                             |
| Dependency manifest    | `pyproject.toml` + `poetry.lock` with hashes; Renovate + Dependabot for updates; OSSF Scorecard for audit. |

---

## 3. What we publish

| Artifact                     | Where                                | Signed              | Provenance       | SBOM             |
| ---------------------------- | ------------------------------------ | ------------------- | ---------------- | ---------------- |
| Container image              | `ghcr.io/abailey81/i3:<semver>`      | Cosign (keyless)    | SLSA L3          | SPDX (Syft)      |
| Python wheel + sdist         | `pypi.org/project/i3/`               | Sigstore (via PyPI) | SLSA L3          | CycloneDX        |
| GitHub Release tarball       | `github.com/abailey81/i3/releases`   | Release-please      | SLSA L3          | CycloneDX        |

### 3.1 SBOM generation & distribution

- **Python SBOM** (`sbom.yml` / `sbom-python`): `cyclonedx-py poetry` emits
  CycloneDX JSON + XML. On published releases, the SBOM is attested via
  `actions/attest-sbom@v1` so consumers can retrieve it directly from the
  GitHub Attestations API.
- **Image SBOM** (`docker.yml` / `anchore/sbom-action@v0`): Syft emits SPDX
  JSON, which is attached to the image as an OCI referrer and attested with
  `actions/attest-sbom@v1`. Consumers discover it with:

  ```bash
  cosign download sbom ghcr.io/abailey81/i3@sha256:<digest>
  ```

- **Retention**: artifact SBOMs are retained for 90 days in Actions. The
  permanent copy lives alongside the signed image or release.

### 3.2 Image signing & verification

Image signing is keyless: at build time the workflow receives a short-lived
OIDC token, exchanges it for a Fulcio certificate, signs the digest, and
writes the signature + certificate to Rekor. The signed reference is
`image@sha256:<digest>`. Consumers verify with:

```bash
cosign verify \
  --certificate-identity-regexp '^https://github.com/abailey81/i3/\.github/workflows/docker\.yml@refs/(heads/main|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/abailey81/i3@sha256:<digest>
```

### 3.3 Reproducible builds

- Python wheels are built from a locked dependency graph (`poetry.lock`
  hashes) on a pinned Python minor version.
- Container images install only the locked requirements set exported from
  Poetry; system packages are pinned by apt snapshot where practical.
- Renovate's `lockFileMaintenance` keeps the lockfile fresh without
  unsolicited semver drift.
- CI's `lockfile-audit` workflow fails any PR where `poetry.lock` drifts from
  `pyproject.toml`.
- Timestamps in image layers are normalised via buildx's `SOURCE_DATE_EPOCH`
  wherever possible.

Reproducibility is best-effort for images that depend on upstream base-image
updates — we prioritise _provenance_ (who built it, from what commit, with
what inputs) over byte-identical reproducibility.

---

## 4. Dependency vetting & management

### 4.1 Ingestion policy

- **Direct dependencies** are reviewed by a CODEOWNER before merge.
- **Transitive dependencies** are pinned by hash in `poetry.lock`.
- **New dependencies** must:
  1. be actively maintained (commit within the last 12 months),
  2. have a license compatible with Apache-2.0,
  3. have no open CRITICAL CVEs at the pinned version,
  4. have a reasonable OSSF Scorecard score (>= 5.0) or a justification.

### 4.2 Update cadence

- **Renovate** opens grouped PRs weekly (Monday before 06:00 Europe/London).
  - `ml-core`: torch / numpy / scipy / scikit-learn / pandas
  - `web-stack`: fastapi / starlette / uvicorn / httpx / pydantic
  - `dev-tooling`: pytest / ruff / mypy / black / pre-commit / coverage / hypothesis
  - `github-actions`: non-major action updates (auto-merged, digest-pinned)
- **Dependabot** is also enabled (superseded by Renovate — left in place as a
  secondary net).
- **Vulnerability PRs** bypass the schedule and are labelled `security`.
- **Major runtime upgrades** are never auto-merged.

### 4.3 Continuous scanning

| Tool            | Scope                                  | Trigger                  | Output             |
| --------------- | -------------------------------------- | ------------------------ | ------------------ |
| Bandit          | Python source                          | PR, push, weekly         | SARIF (existing)   |
| pip-audit       | Python dependency tree                 | PR, push, weekly         | SARIF (existing)   |
| Safety          | Python dependency tree                 | PR, push, weekly         | SARIF (existing)   |
| CodeQL          | Python + actions                       | PR, push, weekly         | SARIF (existing)   |
| Gitleaks        | Secrets                                | PR, push                 | SARIF (existing)   |
| Semgrep         | Python, OWASP Top 10, FastAPI, JWT…    | PR, push, weekly         | SARIF              |
| Trivy (fs)      | Source tree vulns                      | PR, push, weekly         | SARIF              |
| Trivy (config)  | Dockerfile / k8s / helm misconfig      | PR, push, weekly         | SARIF              |
| Trivy (image)   | Final container                        | push main/tag            | SARIF              |
| OSSF Scorecard  | Supply-chain posture                   | Weekly, push main        | SARIF + badge      |

All SARIF streams feed GitHub code-scanning so findings appear in the
Security tab with deduplication.

---

## 5. Vulnerability response policy

### 5.1 Reporting

- **Exploitable / unfixed** vulnerabilities must be reported privately via
  GitHub's
  [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  or the email in `SECURITY.md`.
- Public / informational reports may use the `Security Report` issue template.

### 5.2 Triage SLA

| Severity (CVSSv3.1)    | Acknowledge | First fix target | Disclosure target |
| ---------------------- | ----------- | ---------------- | ----------------- |
| Critical (9.0 – 10.0)  | 24 hours    | 72 hours         | 7 days            |
| High (7.0 – 8.9)       | 72 hours    | 14 days          | 30 days           |
| Medium (4.0 – 6.9)     | 7 days      | 30 days          | 60 days           |
| Low (< 4.0)            | 14 days     | next minor       | next minor        |

### 5.3 Workflow

1. **Acknowledge** — respond to the reporter within SLA.
2. **Reproduce & classify** — assign CVSS, determine affected versions.
3. **Fix** — develop the patch on a private security branch or GitHub Security
   Advisory draft. All fixes follow the normal PR + CI + signing pipeline.
4. **Pre-release** — publish a patched release (e.g. `X.Y.Z+1`) with SLSA
   provenance and updated SBOM.
5. **Disclose** — file a GHSA, request a CVE, update `CHANGELOG.md`,
   notify subscribers.
6. **Postmortem** — document root cause + prevention action items.

### 5.4 Consumer guidance

- Always verify the digest + signature before deploying (see §3.2, `docs/security/slsa.md`).
- Subscribe to
  [`abailey81/i3` security advisories](https://github.com/abailey81/i3/security/advisories).
- Pin images by digest, not tag, in production.
- Re-scan your deployment's SBOM against the OSV database weekly.

---

## 6. Signing keys & identity rotation

Because signing is keyless (Fulcio-issued ephemeral certificates), there is no
long-lived private key on any developer or CI machine. Rotation happens
_per-build_ as a property of the OIDC token exchange.

If a repository compromise is suspected:

1. Revoke the suspect OIDC-bound workflow's permissions in GitHub.
2. Rotate any repository/org/environment secrets (PyPI trusted publisher
   settings, GHCR tokens if any were issued).
3. Yank the affected releases from PyPI.
4. Republish a clean build from a known-good commit; the new signatures will
   have different Rekor log-index entries. Any previously-published but
   unverifiable signatures will be publicly visible in Rekor — this is the
   intended design.
5. Post a GHSA documenting scope, affected digests, and remediation.

---

## 7. Third-party action trust

- All actions from **high-trust categories** (signing, provenance, attestation)
  are pinned to a **commit SHA** — `actions/attest-sbom`,
  `sigstore/cosign-installer`, `slsa-framework/slsa-github-generator/*`.
- Other actions are pinned by **major tag**; Renovate upgrades them in the
  `github-actions` group with `pinDigests: true` so that every PR records
  the exact digest the tag resolves to.
- New actions require CODEOWNER review of `/.github/`.

---

## 8. Appendix — quick commands

### Verify an image
```bash
cosign verify \
  --certificate-identity-regexp '^https://github.com/abailey81/i3/\.github/workflows/docker\.yml@refs/(heads/main|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/abailey81/i3@sha256:<digest>
```

### Verify SLSA provenance for an image
```bash
slsa-verifier verify-image \
  --source-uri github.com/abailey81/i3 \
  --source-branch main \
  ghcr.io/abailey81/i3@sha256:<digest>
```

### Download an image SBOM
```bash
cosign download sbom ghcr.io/abailey81/i3@sha256:<digest> > sbom.spdx.json
```

### Verify a PyPI wheel
```bash
slsa-verifier verify-artifact \
  --provenance-path i3-X.Y.Z-py3-none-any.whl.intoto.jsonl \
  --source-uri github.com/abailey81/i3 \
  --source-tag vX.Y.Z \
  i3-X.Y.Z-py3-none-any.whl
```

### Validate lockfile locally
```bash
poetry check --lock
```

### Re-run all supply-chain scans locally
```bash
# Semgrep
docker run --rm -v "$PWD:/src" semgrep/semgrep:latest semgrep ci \
  --config p/python --config p/owasp-top-ten --config p/security-audit \
  --config p/docker --config p/github-actions --config p/bandit \
  --config p/secrets --config p/fastapi --config p/jwt

# Trivy filesystem
trivy fs --severity CRITICAL,HIGH .

# CycloneDX SBOM
cyclonedx-py poetry --output-format JSON --output-file sbom-python.json .
```
