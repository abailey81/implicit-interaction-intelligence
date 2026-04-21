# I³ Helm Chart

This chart installs the Implicit Interaction Intelligence (I³) FastAPI + PyTorch
server with production-grade defaults.

## Requirements

- Kubernetes >= 1.26
- Helm >= 3.12
- Optional: cert-manager (for automatic TLS via ingress)
- Optional: ingress-nginx (for the default ingress resource)
- Optional: Prometheus Operator (for `ServiceMonitor`)
- Optional: Prometheus Adapter (for HPA custom `http_requests_per_second` metric)

## Quickstart

```bash
# Add the chart from a local checkout
helm dependency update ./deploy/helm/i3   # no deps yet; safe no-op

# Install (dev)
helm upgrade --install i3 ./deploy/helm/i3 \
  --namespace i3-dev --create-namespace \
  --values ./deploy/helm/i3/values-dev.yaml \
  --set envFromSecret.data.I3_ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  --set envFromSecret.data.ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"

# Install (prod)
# Create a SealedSecret named `i3-secrets` in the prod namespace first, then:
helm upgrade --install i3 ./deploy/helm/i3 \
  --namespace i3-prod --create-namespace \
  --values ./deploy/helm/i3/values-prod.yaml \
  --set image.tag="1.0.0"
```

## Validate

```bash
helm lint ./deploy/helm/i3
helm template i3 ./deploy/helm/i3 --values ./deploy/helm/i3/values-prod.yaml | kubectl apply --dry-run=client -f -
helm test i3 --namespace i3-dev
```

## Key values

| Key | Default | Description |
| --- | --- | --- |
| `image.repository` | `ghcr.io/abailey81/i3` | Container image repository |
| `image.tag` | `1.0.0` | Image tag (fallback: `.Chart.AppVersion`) |
| `replicaCount` | `2` | Static replica count when autoscaling disabled |
| `autoscaling.enabled` | `true` | Enable the HPA |
| `autoscaling.minReplicas` / `maxReplicas` | `2` / `10` | HPA bounds |
| `resources` | requests 500m/1Gi, limits 2/2Gi | Container resources |
| `envFromSecret.existingSecret` | `i3-secrets` | Pre-existing Secret with `I3_ENCRYPTION_KEY`, `ANTHROPIC_API_KEY` |
| `envFromSecret.data` | `{}` | Inline secret values (dev only — never commit) |
| `ingress.enabled` | `true` | Create an Ingress resource |
| `ingress.className` | `nginx` | IngressClass |
| `podDisruptionBudget.enabled` | `true` | Create a PDB (`minAvailable: 1`) |
| `networkPolicy.enabled` | `true` | Default-deny + narrow allowlist |
| `serviceMonitor.enabled` | `true` | Prometheus Operator integration |
| `config.*` | see `values.yaml` | Non-secret env vars rendered into a ConfigMap |

See `values.yaml` for the full set.

## Secrets

Never commit real secrets. The chart wires `envFromSecret.existingSecret` (default
`i3-secrets`) into the Deployment via `envFrom`. Production clusters should
populate this Secret via:

- [SealedSecrets](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- A KMS-backed operator (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)

See `deploy/k8s/secret.example.yaml` for a SealedSecret snippet.

## Uninstall

```bash
helm uninstall i3 --namespace i3-prod
kubectl delete namespace i3-prod   # optional; removes the Secret too
```
