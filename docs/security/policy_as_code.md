# Policy-as-Code — The Four-Layer Security Story

_Design doc, ~3,000 words, versioned alongside the codebase._

This document explains how Implicit Interaction Intelligence (I³) composes
four distinct "policy-as-code" layers into a single defensible posture:
**supply chain**, **cluster policy**, **application authorization**, and
**runtime security**. Each layer has its own threat model, its own tool,
and its own evaluation semantics. Together they cover the end-to-end
lifecycle of a release, from the moment a developer's commit is signed
to the moment a syscall fires on a production node.

This doc is the *normative reference* — if you need to know why a given
rule exists, where it lives, which NIST control it maps to, or which
CVE family motivated it, start here.

---

## 1. Motivation

A single piece of code travels a surprising distance on its way to end
users. A diff becomes a commit, a commit becomes a container image, an
image becomes a pod, a pod makes syscalls. At each hop an attacker might
substitute code, smuggle a dependency, grant excessive permissions,
exfiltrate data, or persist a foothold. Historically projects have tried
to defend this chain with one or two monolithic controls — a network
firewall plus a WAF, say, or just "use TLS everywhere". That approach
under-protects the developer → artifact half of the pipeline and
over-trusts the shared runtime.

The CNCF community has converged on a different model: **many small,
declarative, machine-checkable policies**, each scoped to a narrow
layer, version-controlled in the repo, tested in CI, and enforced at
runtime by a policy engine purpose-built for that layer. Violations
become pull-request failures, admission rejections, or runtime alerts
— never opaque security reviewer opinions. This document is I³'s
expression of that model.

---

## 2. The four layers

```
┌───────────────────────────────────────────────────────────────┐
│ 1. Supply chain     — SBOM, cosign, SLSA, model-signing       │  already shipped
├───────────────────────────────────────────────────────────────┤
│ 2. Cluster policy   — Kyverno, OPA Gatekeeper, Sigstore PC    │  this commit
├───────────────────────────────────────────────────────────────┤
│ 3. App authorization — Cedar + CedarAuthorizer Python adapter │  this commit
├───────────────────────────────────────────────────────────────┤
│ 4. Runtime security — Falco rules, Tracee policy, Allstar     │  this commit
└───────────────────────────────────────────────────────────────┘
```

### 2.1 Layer 1 — Supply chain (_already shipped_)

The existing `.github/workflows/docker.yml` pipeline produces an image
accompanied by:

- a **CycloneDX SBOM** (`sbom.yml` workflow),
- a **cosign keyless signature** from the GitHub Actions OIDC token,
- a **SLSA Level 3 provenance attestation** (`docs/security/slsa.md` documents the
  journey),
- an **OpenSSF model-signing v1.0** envelope for PyTorch weights.

The threat this covers: *the bits that run in production must be the
bits built from the signed commit tree.* Everything downstream depends
on trusting this chain; without it, every later control is defending a
potentially already-compromised image.

### 2.2 Layer 2 — Cluster policy (_this commit_)

Even a signed image can be misconfigured. Cluster policy enforces the
"what this pod is allowed to be" invariants at admission time:

- **Kyverno** — native Kubernetes policy engine. Five `ClusterPolicy`
  manifests in `deploy/policy/kyverno/` enforce:
    1. every image admitted to the `i3` namespace carries a cosign
       signature from the canonical GitHub OIDC identity
       (`require-signed-images.yaml`);
    2. no image uses `:latest` or an empty tag anywhere in the cluster
       (`deny-latest-tag.yaml`);
    3. every pod runs non-root with a read-only root filesystem, the
       `RuntimeDefault` seccomp profile, `allowPrivilegeEscalation=false`,
       and drops all Linux capabilities (`require-non-root.yaml`);
    4. pods cannot schedule in a namespace that lacks a NetworkPolicy
       (`network-policy-required.yaml`);
    5. every new namespace is automatically seeded with a default-deny
       NetworkPolicy (`generate-defaults.yaml`).

- **OPA Gatekeeper** — for clusters that standardise on OPA instead of
  Kyverno, the Rego bundle in `deploy/policy/opa/` implements the same
  contract. `admission.rego` packages the deny rules; `data.rego` holds
  allow-lists; `test_admission.rego` is the unit-test suite.

- **Sigstore Policy Controller** —
  `deploy/policy/sigstore/policy-controller-config.yaml` provides a
  **second** signature check at admission, independent of Kyverno. The
  two engines form a belt-and-braces: if one fails open (a bug, a CRD
  misapply) the other still blocks.

Threat covered: *unsigned, misconfigured, or unprovenanced workloads
landing in the cluster.*

### 2.3 Layer 3 — Application authorization (_this commit_)

Even a perfectly-hardened pod must then decide, per request, whether
user X can read resource Y. I³'s data — diaries, user profiles,
adaptation vectors — is deeply personal, and authorization bugs at the
application layer are responsible for some of the industry's loudest
breaches (the IDOR class, specifically).

We model application authorization with **Cedar** (AWS, Rust, OOPSLA
2024). The shipped assets are:

- `deploy/policy/cedar/i3.cedar` — the complete I³ policy set:
    - "only owner can read own diary",
    - "only owner can write own diary",
    - "export requires MFA",
    - "admin delete requires a non-empty reason",
    - "admin bulk delete requires reason AND a trusted-IP session",
    - "admin may never read diary contents" (explicit `forbid`),
    - "cloud service may read Adaptation only when `sanitized=true`",
    - "cloud service may never touch Diary" (explicit `forbid`),
    - "user may read/write only their own profile".
- `deploy/policy/cedar/schema.cedarschema.json` — the Cedar schema
  that types entities, actions, and context attributes.
- `deploy/policy/cedar/tests/test_cases.json` — 21 scenario test
  cases covering happy paths, IDOR attempts, missing MFA, missing
  reasons, cloud-path leakage, etc.
- `i3/authz/cedar_adapter.py` — a 100%-typed Python adapter exposing
  `CedarAuthorizer.is_authorized(principal, action, resource, context)`.
  The adapter soft-imports `cedarpy` so the service still boots when
  the policy engine isn't installed.

**Why Cedar?** Cedar was designed from day one for provable
authorization: its core evaluator has a Lean proof; its static
analyzer can answer questions like "does policy P _ever_ grant
`bulkDelete` to a non-admin?" without running traffic through it; and
its policy language is deliberately simpler than Rego (no higher-order
functions, no unbounded iteration), which makes formal analysis
tractable. For a privacy-sensitive system those properties outweigh
Rego's broader applicability.

Threat covered: *per-request authorization bugs — IDOR, missing access
checks, role confusion, silent privilege escalation at the REST/RPC
layer.*

### 2.4 Layer 4 — Runtime security (_this commit_)

Once the pod is running and requests are authorized, something can
still go wrong: a dependency with an undisclosed RCE, a container
escape, a compromised developer laptop with `kubectl exec` rights.
Runtime security answers the question *are the syscalls this container
is making consistent with what I³ does?*

Two complementary tools:

- **Falco** (`deploy/policy/falco/i3_rules.yaml`) — alerts on four
  specific patterns:
    1. the diary SQLite file is opened by a process other than
       `uvicorn` / `python` (indicates side-channel access or a
       debugger),
    2. the container makes an outbound connection to anything other
       than `api.anthropic.com` or the Sigstore transparency log,
    3. *any* exec into the i3 container fires an alert (the service
       never needs a shell),
    4. any write to `/app` violates the read-only root filesystem and
       indicates tampering.
  Falco is deployed via the Helm values in `helm-values.yaml`.

- **Tracee** (`deploy/policy/tracee/policies/i3_policy.yaml`) — traces
  everything Falco alerts on *plus* kernel events (ptrace, BPF attach,
  module loads), feeding a longer-horizon forensic stream into a SIEM.
  Falco fires alerts; Tracee builds the timeline you use to answer
  "what did the attacker do next?".

- **OSSF Allstar** (`.github/allstar/*.yaml`) — continuously verifies
  repo-level properties: branch protection with signed commits,
  required reviewers, required status checks, presence of SECURITY.md.
  The runtime here is the repo itself.

Threat covered: *post-admission compromise, tampering, exfiltration,
and drift in the developer workflow.*

---

## 3. Control matrix

Mapping each shipped artifact to NIST SP 800-53 Rev. 5 and the CIS
Kubernetes Benchmark v1.9.

| Artifact | NIST 800-53 | CIS K8s Benchmark |
|---|---|---|
| `deploy/policy/kyverno/require-signed-images.yaml` | SR-4(3), SI-7(15), CM-14 | 5.7.2 |
| `deploy/policy/kyverno/deny-latest-tag.yaml` | CM-8, CM-14 | 5.1.4 |
| `deploy/policy/kyverno/require-non-root.yaml` | AC-6(9), AC-6(10), SI-16 | 5.2.6, 5.2.5, 5.7.3 |
| `deploy/policy/kyverno/network-policy-required.yaml` | SC-7(5), AC-4(21) | 5.3.2 |
| `deploy/policy/kyverno/generate-defaults.yaml` | SC-7(5), SC-7(11) | 5.3.2 |
| `deploy/policy/opa/admission.rego` | SI-7(15), AC-6, CM-14 | 5.2.*, 5.7.2 |
| `deploy/policy/cedar/i3.cedar` | AC-3, AC-3(3), AC-6, AC-16 | n/a (app-layer) |
| `deploy/policy/falco/i3_rules.yaml` | SI-4, SI-4(5), SI-4(23), AU-6 | 5.6.3, 6.1 |
| `deploy/policy/tracee/policies/i3_policy.yaml` | AU-2, AU-12, SI-4 | 6.2 |
| `deploy/policy/sigstore/policy-controller-config.yaml` | SR-4(3), SR-11 | 5.7.2 |
| `.github/allstar/branch_protection.yaml` | CM-3, CM-5, SA-10 | n/a (repo) |
| `.github/allstar/security.yaml` | IR-6, PM-31 | n/a (repo) |

Gaps worth flagging:

- **AC-16 (Security & Privacy Attributes):** Cedar context attributes
  (`mfa`, `sanitized`, `ip_trusted`) carry the policy-relevant security
  metadata; the adapter's caller is responsible for populating them
  truthfully. A future enhancement is to sign those attributes at the
  API gateway to prevent forgery by a compromised middleware.
- **SC-13 (Cryptographic Protection):** mostly inherited from the
  existing `i3/privacy/encryption.py` module and the Sigstore chain; not
  duplicated here.

---

## 4. Formal methods — Cedar's provable authorization

A brief note on why Cedar is singled out for the application layer.

Cedar was announced in 2023 alongside Amazon Verified Permissions, and
its semantics were published formally in *"Cedar: A New Language for
Expressive, Fast, Safe, and Analyzable Authorization"* (Cutler, Eykholt,
Finkbeiner, Losa, Tacchella, Tan, Thakur, Varghese & Wells, OOPSLA 2024).
The paper proves three properties that matter for I³:

1. **Soundness of the SMT encoding** — Cedar policies can be compiled
   to SMT, and the analyzer's answers are sound with respect to the
   evaluator. This is what makes queries like "find every request the
   policy set allows for a non-admin principal on a Diary resource"
   tractable.
2. **Decidability** — because the language intentionally omits
   unbounded iteration and higher-order functions, every analysis
   terminates. Rego does not guarantee this in general.
3. **Total evaluation** — every Cedar expression either returns an
   authorization decision or raises a structural error; there is no
   undefined behaviour.

Concretely: we can answer "does `i3.cedar` ever grant `delete` to a
non-admin without a `reason` context attribute?" with
`cedar analyze --query ...` *without* a single live request. The
existing scenario tests in `tests/test_cedar_authz.py` exercise the
happy and unhappy paths empirically; the analyzer closes the remaining
quantifier. Both are in scope for CI.

See also AWS's 2023 announcement:
<https://aws.amazon.com/verified-permissions/>.

---

## 5. Updated threat model

The threats below were identified during the A17 supply-chain audit
(see `reports/audits/2026-04-22-post-v1-security.md`) and the new policy layer. Each maps to
an owning control.

| ID | Threat | Layer | Control |
|----|--------|-------|---------|
| T1 | Malicious dependency injected via `poetry install` | 1 | SBOM + pip-audit + model-signing |
| T2 | Compromised maintainer pushes unreviewed code | 1, 4 | Branch protection (Allstar) + signed commits |
| T3 | Attacker substitutes image tag at pull time | 2 | Kyverno `verifyImages` + Sigstore PC + `mutateDigest` |
| T4 | Operator deploys `:latest` or untagged image | 2 | `deny-latest-tag.yaml` |
| T5 | Pod scheduled as root / with writable rootfs | 2 | `require-non-root.yaml` |
| T6 | Pod scheduled into a namespace with open networking | 2 | `network-policy-required.yaml` + `generate-defaults.yaml` |
| T7 | IDOR — user X reads user Y's diary | 3 | `i3.cedar` rule 1 |
| T8 | Admin abuses privileges for covert reads | 3 | `i3.cedar` rule 6 explicit `forbid` |
| T9 | Cloud path leaks raw un-sanitized state | 3 | `i3.cedar` rule 7 + 8 |
| T10 | Exfiltration via DNS / outbound connection | 4 | Falco egress rule + Tracee `net_packet_dns_request` |
| T11 | Container escape / tampering with `/app` | 4 | Falco "write to /app" + Tracee `magic_write` |
| T12 | SRE uses `kubectl exec` to access production data | 4 | Falco "I3 container exec" (CRITICAL) |
| T13 | Kernel module loaded at runtime | 4 | Tracee `init_module`/`finit_module` |

T1 – T13 are the controls tested in CI (`.github/workflows/policy-test.yml`)
and monitored in production via Falco / Tracee exporters.

---

## 6. Operational notes

### 6.1 Choosing Kyverno *or* OPA

Install **one** cluster-policy engine, not both. Kyverno is the
default because its YAML is closer to Kubernetes' own idiom and its
`verifyImages` primitive has the best Sigstore integration out of the
box. Operators standardised on OPA can apply the Rego bundle instead;
the two sets are functionally equivalent.

### 6.2 Kyverno + Sigstore Policy Controller coexist

Running both is intentional redundancy. Kyverno's signature check is
part of a broader validation pass (tags, securityContext, …); Sigstore
PC is a single-purpose verifier from the signing project itself. If one
engine fails open, the other still blocks.

### 6.3 Falco vs. Tracee

Falco alerts, Tracee traces. Route Falco output into the on-call
pager; route Tracee output into the forensic archive. Most operators
will deploy both; budget-constrained environments should choose Falco
first because alerts yield action faster than traces.

### 6.4 Updating Cedar policies

Cedar policies are code. Treat them as such:

1. Open a PR modifying `deploy/policy/cedar/i3.cedar`.
2. Add / update scenarios in `tests/test_cases.json`.
3. CI runs `cedar validate`, `cedar authorize` over the scenarios,
   *and* the Python adapter tests.
4. On merge, the service picks up the new policy at next restart.

No policy is ever deployed without a green `policy-test.yml` run.

### 6.5 Adding new cluster policies

New Kyverno / OPA rules follow the same lifecycle: add the manifest,
add a Kyverno CLI test under `tests/` (TODO: shipped incrementally),
update the NIST matrix in this document, merge.

---

## 7. References

- **Kyverno** — <https://kyverno.io/docs/>
- **Kyverno `verifyImages`** —
  <https://kyverno.io/docs/writing-policies/verify-images/sigstore/>
- **OPA Gatekeeper** — <https://open-policy-agent.github.io/gatekeeper/>
- **Rego v1 reference** —
  <https://www.openpolicyagent.org/docs/latest/policy-language/>
- **Cedar** — <https://www.cedarpolicy.com/>
- **Cedar paper** — Cutler et al., *"Cedar: A New Language for
  Expressive, Fast, Safe, and Analyzable Authorization"*, OOPSLA 2024.
- **Amazon Verified Permissions announcement (2023)** —
  <https://aws.amazon.com/verified-permissions/>
- **Falco** — <https://falco.org/docs/>
- **Tracee** — <https://aquasecurity.github.io/tracee/>
- **Sigstore Policy Controller** —
  <https://docs.sigstore.dev/policy-controller/overview/>
- **OSSF Allstar** — <https://github.com/ossf/allstar>
- **NIST SP 800-53 Rev. 5** —
  <https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final>
- **CIS Kubernetes Benchmark v1.9** —
  <https://www.cisecurity.org/benchmark/kubernetes>
- Companion docs: `SECURITY.md`, `docs/security/slsa.md`, `docs/security/supply-chain.md`.

---

## 8. Glossary

- **Admission control** — the Kubernetes phase where a request to
  create / update a resource is inspected and possibly rejected, before
  it is persisted.
- **Attestation** — a signed statement about a subject (e.g. "this
  image was built from this commit by this GitHub Actions workflow").
- **Default deny** — a policy whose absence of an explicit rule means
  "reject". The opposite of fail-open.
- **IDOR** — Insecure Direct Object Reference; an app-layer bug where a
  user can access another user's resource by guessing or tampering with
  an identifier.
- **OIDC** — OpenID Connect. GitHub Actions issues an OIDC token for
  every workflow run; cosign uses that token to request a short-lived
  Fulcio certificate and sign the image keylessly.
- **Rekor** — the Sigstore transparency log; every signature is
  recorded there, enabling after-the-fact detection of unauthorised
  signing.
- **SLSA** — Supply-chain Levels for Software Artifacts
  (<https://slsa.dev/>). I³ targets Level 3.
