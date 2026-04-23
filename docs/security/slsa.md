# SLSA Build Level 3 Posture

Implicit Interaction Intelligence (I3) publishes SLSA Level 3 provenance for
**every container image** pushed to GHCR and **every Python distribution**
published to PyPI. This document explains how we meet each requirement of
[SLSA v1.0 Build L3](https://slsa.dev/spec/v1.0/levels#build-l3) and how
consumers can verify it before installing or running our artifacts.

- Container registry: `ghcr.io/abailey81/i3`
- PyPI project: `i3`
- Source of truth: `github.com/abailey81/i3`
- Build platform: GitHub-hosted runners (`ubuntu-latest`) driven by the
  reusable workflows in `slsa-framework/slsa-github-generator` pinned to tag
  `v2.0.0`.

---

## 1. SLSA requirements — mapping

| Requirement (Build L3)                            | How we satisfy it                                                                                                                                                                                                                                                           |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Provenance exists**                             | Produced by the reusable workflows `generator_container_slsa3.yml` (images) and `generator_generic_slsa3.yml` (PyPI wheels/sdist) in [slsa-framework/slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator).                                        |
| **Provenance is authentic**                       | Signed with Sigstore Fulcio certificates issued against the `token.actions.githubusercontent.com` OIDC issuer; signature entries are recorded in the public [Rekor transparency log](https://rekor.sigstore.dev).                                                            |
| **Provenance has non-falsifiable fields**         | The build identity (repository, workflow path, ref, commit SHA) is asserted by GitHub's OIDC token, not user-controllable. The generator runs in an **isolated reusable workflow** where the calling workflow cannot tamper with the signing keys.                           |
| **Build is hermetic & isolated**                  | Each build runs on a fresh, ephemeral GitHub-hosted runner. The signing step is executed inside `slsa-github-generator`'s reusable workflow, which cannot be altered by the calling repository and does not share its token with prior steps.                                |
| **Dependencies are declared**                     | `pyproject.toml` + `poetry.lock` pin every Python dependency by hash. GitHub Actions are pinned by major tag (and by commit SHA for all high-trust actions — attest, cosign, SLSA generator).                                                                                |
| **Provenance is distributed with the artifact**   | Provenance is uploaded as a release asset for wheels/sdists and stored as an OCI referrer in GHCR for container images, discoverable via `cosign download attestation` or `gh attestation list`.                                                                              |
| **SBOM is attached**                              | A CycloneDX SBOM is generated for each Python release (`sbom.yml` job `sbom-python`); a Syft SPDX JSON SBOM is generated for every image and pushed to GHCR as an OCI artifact (`docker.yml` job `build-and-push` via `actions/attest-sbom@v1`).                               |

---

## 2. Build platform

- **Runner**: GitHub-hosted `ubuntu-latest` (x86_64) and `ubuntu-latest` with
  QEMU for `linux/arm64` images.
- **Caller workflows**: `.github/workflows/release.yml`, `.github/workflows/docker.yml`.
- **Provenance generator**:
  - Images: `slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0`
  - Wheels/sdist: `slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.0.0`
- **OIDC issuer**: `https://token.actions.githubusercontent.com`
- **Signing CA**: Sigstore Fulcio (`https://fulcio.sigstore.dev`).
- **Transparency log**: Sigstore Rekor (`https://rekor.sigstore.dev`).

Reusable workflows run in a separate job and receive a fresh OIDC token whose
subject identifies _them_, not the caller. This means the caller cannot
impersonate the generator even if it is compromised — the non-falsifiability
requirement is enforced by GitHub's token scoping plus Sigstore's identity-based
verification.

---

## 3. Source provenance

Every build records, inside the provenance envelope:

- the repository URL (`github.com/abailey81/i3`),
- the commit SHA (immutable),
- the ref (`refs/heads/main` or `refs/tags/vX.Y.Z`),
- the workflow path and filename that triggered the build,
- the event name (`push`, `release`, `workflow_dispatch`),
- the invocation parameters (builder inputs) in the `buildConfig` section.

Because all of these are asserted by GitHub's OIDC claims, they cannot be
falsified by malicious workflow content in the calling repo.

---

## 4. Dependency tracking

- **Python**: `poetry.lock` pins every transitive dependency by version and hash.
  The `lockfile-audit` workflow fails any PR where `poetry.lock` is out of sync
  with `pyproject.toml`.
- **GitHub Actions**: all high-trust actions (`actions/attest-sbom`,
  `sigstore/cosign-installer`, `slsa-framework/slsa-github-generator/*`) are
  pinned by commit SHA with a version comment. Other actions are pinned by
  major tag and updated via Renovate with `pinDigests: true`.
- **Container base images**: pinned by tag; Renovate rule
  `docker-base-images` enforces a 7-day `minimumReleaseAge` and requires
  manual review.
- **SBOM**: a CycloneDX SBOM is produced on every push to `main` and attested
  on every release, giving consumers a reproducible view of the dependency
  closure at the moment of the build.

---

## 5. Verification — for consumers

### 5.1 Verify a container image

Use `cosign` to verify the image's Sigstore signature (pinned by digest, not tag):

```bash
cosign verify \
  --certificate-identity-regexp '^https://github.com/abailey81/i3/\.github/workflows/docker\.yml@refs/(heads/main|tags/v.*)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/abailey81/i3@sha256:<digest>
```

Then verify the SLSA provenance with
[`slsa-verifier`](https://github.com/slsa-framework/slsa-verifier):

```bash
slsa-verifier verify-image \
  --source-uri github.com/abailey81/i3 \
  --source-branch main \
  ghcr.io/abailey81/i3@sha256:<digest>
```

Optional — inspect the attached SBOM:

```bash
cosign download attestation \
  --predicate-type=https://spdx.dev/Document \
  ghcr.io/abailey81/i3@sha256:<digest> \
  | jq -r '.payload' | base64 -d | jq .
```

### 5.2 Verify a PyPI wheel

```bash
pip download --no-deps i3==X.Y.Z
slsa-verifier verify-artifact \
  --provenance-path i3-X.Y.Z-py3-none-any.whl.intoto.jsonl \
  --source-uri github.com/abailey81/i3 \
  --source-tag vX.Y.Z \
  i3-X.Y.Z-py3-none-any.whl
```

The `.intoto.jsonl` provenance file is attached to the GitHub release.

### 5.3 Check GitHub's attestation UI

All attestations are also viewable at:

```
https://github.com/abailey81/i3/attestations
```

and via the `gh` CLI:

```bash
gh attestation verify ghcr.io/abailey81/i3@sha256:<digest> --repo abailey81/i3
```

---

## 6. Threat model — what this does and does not protect against

SLSA Level 3 **does** protect against:

- a malicious maintainer secretly swapping the contents of a release artifact
  after it is signed (provenance would not match),
- a tampered build step (since the signing runs in the isolated reusable
  workflow, a compromised caller step cannot forge the provenance),
- silent changes to the source repository (commit SHA is in the provenance),
- reusing a provenance from another repo (the `source-uri` is asserted by OIDC).

SLSA Level 3 **does not** protect against:

- an attacker with push access committing malicious source that is then
  legitimately built (mitigated by CODEOWNERS + signed commits + review),
- supply-chain attacks on declared dependencies (mitigated by `poetry.lock`
  hashes + Trivy + Renovate + vulnerability alerts),
- a compromise of the GitHub Actions platform itself (mitigated by Sigstore's
  transparency log — any unauthorised signature is publicly visible).

---

## 7. Key rotation & incident response

- Signing uses **ephemeral keys** issued by Fulcio per-run; there is no long-lived
  private key to rotate.
- If an identity used by the workflow is ever suspected of compromise (for
  example a repo takeover), we:
  1. Invalidate the affected tags and yank the published artifacts from PyPI.
  2. Re-issue a clean release from a known-good commit.
  3. File an advisory on
     [`github.com/abailey81/i3/security/advisories`](https://github.com/abailey81/i3/security/advisories)
     referencing the affected digests.
- See `docs/security/supply-chain.md` for the full vulnerability response workflow.

---

## 8. References

- SLSA v1.0 specification: <https://slsa.dev/spec/v1.0/>
- `slsa-github-generator` docs: <https://github.com/slsa-framework/slsa-github-generator>
- `slsa-verifier`: <https://github.com/slsa-framework/slsa-verifier>
- Cosign: <https://docs.sigstore.dev/cosign/overview/>
- GitHub Artifact Attestations: <https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds>
