# Kyverno Policies for Implicit Interaction Intelligence (I³)

Kyverno is a Kubernetes-native policy engine that validates, mutates, and
generates resources using declarative YAML (no Rego required). These
[`ClusterPolicy`][kyverno-cp] manifests enforce supply-chain and workload
hardening controls across the `i3` namespace and (where noted) the entire
cluster.

## Install

Install the Kyverno controller (v1.11+ recommended for `verifyImages` with
keyless cosign support):

```bash
helm repo add kyverno https://kyverno.github.io/kyverno/
helm repo update
helm install kyverno kyverno/kyverno \
  --namespace kyverno --create-namespace \
  --set admissionController.replicas=3 \
  --set admissionController.rbac.create=true
```

## Apply policies

All manifests in this directory are `ClusterPolicy` objects and are safe to
apply together:

```bash
kubectl apply -f deploy/policy/kyverno/
```

## What each policy enforces

| File                             | Scope                | Effect                                                                     |
| -------------------------------- | -------------------- | -------------------------------------------------------------------------- |
| `require-signed-images.yaml`     | `i3` namespace       | every image must carry a valid cosign signature from the GitHub OIDC issuer |
| `deny-latest-tag.yaml`           | cluster-wide         | blocks admission of `:latest` (or tagless) images                          |
| `require-non-root.yaml`          | `i3` namespace       | runAsNonRoot + readOnlyRootFilesystem + RuntimeDefault seccomp + no PE    |
| `network-policy-required.yaml`   | `i3` namespace       | denies pod creation if the namespace has no NetworkPolicy                 |
| `generate-defaults.yaml`         | new namespaces       | auto-generates a default-deny NetworkPolicy per new namespace             |

## Relationship to OPA

A functionally-equivalent Rego bundle lives in [`../opa/`](../opa/README.md)
for clusters using OPA Gatekeeper instead of Kyverno. The two are mutually
exclusive — install one.

## References

- Kyverno docs: <https://kyverno.io/docs/>
- Keyless cosign in Kyverno: <https://kyverno.io/docs/writing-policies/verify-images/sigstore/>
- Pod Security Standards: <https://kubernetes.io/docs/concepts/security/pod-security-standards/>

[kyverno-cp]: https://kyverno.io/docs/kyverno-policies/policy-types/
