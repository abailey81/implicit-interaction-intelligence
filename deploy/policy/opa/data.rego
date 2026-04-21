# ─── data.rego ────────────────────────────────────────────────────────────
#
# Allow-lists consumed by `admission.rego`. Split out so that operators can
# tune the governance surface (add a registry, bump a severity threshold)
# without touching rule logic.
#
# Rego v1 syntax.

package i3.policy.data

import rego.v1

# Acceptable cosign/Fulcio signer identities (GitHub Actions OIDC).
allowed_signers := {
    "https://github.com/abailey81/implicit-interaction-intelligence/.github/workflows/docker.yml@refs/heads/main",
}

# Permitted registry prefixes for I³ images.
allowed_registries := {
    "ghcr.io/abailey81/implicit-interaction-intelligence",
    ".dkr.ecr.",
}

# CVE severities that should block admission when reported by the scanner.
blocking_severities := {"CRITICAL", "HIGH"}

# Namespaces exempt from the default-deny / non-root rules.
exempt_namespaces := {
    "kube-system",
    "kube-public",
    "kube-node-lease",
    "kyverno",
    "gatekeeper-system",
}

# The namespace I³ workloads run in.
i3_namespace := "i3"
