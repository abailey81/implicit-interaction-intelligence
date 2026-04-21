# I³ Deployment Manifests

Production-grade deployment artifacts for **Implicit Interaction Intelligence
(I³)** — a FastAPI + PyTorch service exposed on port 8000 whose entry point is
`server.app:app`.

## Layout

```
deploy/
  k8s/                   Raw Kubernetes manifests + kustomize overlays
    overlays/{dev,staging,prod}/
  helm/i3/               Helm chart (primary install path)
  terraform/             Reference Terraform for AWS EKS
    modules/i3-app/      Reusable Helm release module
  skaffold.yaml          Local dev loop
  argocd/                GitOps manifests (Application + AppProject)
  observability/         (Owned by another agent — not documented here)
```

## Install paths

### 1. Helm (recommended for most environments)

```bash
helm lint ./deploy/helm/i3
helm template i3 ./deploy/helm/i3 --values ./deploy/helm/i3/values-prod.yaml \
  | kubectl apply --dry-run=client -f -

helm upgrade --install i3 ./deploy/helm/i3 \
  --namespace i3-prod --create-namespace \
  --values ./deploy/helm/i3/values-prod.yaml \
  --set image.tag=1.0.0
```

See [`helm/README.md`](helm/README.md).

### 2. Kustomize

```bash
kubectl apply --dry-run=client -k ./deploy/k8s/overlays/prod
kubectl apply               -k ./deploy/k8s/overlays/prod
```

Overlays set namespace, replicas, image tag, HPA bounds, and ingress host.
The base intentionally **does not include** `secret.example.yaml`; provide
your own Secret (preferably a SealedSecret) before applying.

### 3. Terraform

For AWS EKS environments. See [`terraform/README.md`](terraform/README.md).

### 4. ArgoCD

Apply once to bootstrap GitOps:

```bash
kubectl apply -f deploy/argocd/appproject.yaml
kubectl apply -f deploy/argocd/application.yaml
```

### 5. Skaffold (local dev)

```bash
cd deploy && skaffold dev   # watch + rebuild + redeploy + port-forward :8000
```

## What gets deployed

| Resource | Purpose |
| --- | --- |
| `Deployment` | 2 replicas, rolling update, readOnlyRootFilesystem, drop-ALL capabilities |
| `Service` (ClusterIP) | Port 80 → container 8000; Prometheus scrape annotations |
| `Ingress` (nginx + cert-manager) | TLS, rate-limits, 4MB body cap |
| `HorizontalPodAutoscaler` | min 2 / max 10; CPU 70% + `http_requests_per_second` custom metric |
| `PodDisruptionBudget` | `minAvailable: 1` |
| `NetworkPolicy` | Default-deny; allow from ingress + monitoring; allow DNS + HTTPS egress |
| `ServiceMonitor` | Prometheus Operator scrape of `/api/metrics`, 30s interval |
| `ConfigMap` | Non-secret env (`I3_HOST`, `I3_CORS_ORIGINS`, `I3_DB_PATH`, …) |
| `Secret` (`i3-secrets`) | `I3_ENCRYPTION_KEY`, `ANTHROPIC_API_KEY` — user-supplied |
| `ServiceAccount` | `automountServiceAccountToken: false` |

## Health endpoints

| Path | Usage |
| --- | --- |
| `/api/live` | Liveness + startup probe |
| `/api/ready` | Readiness probe |
| `/api/health` | General health (used by Helm test pod) |
| `/api/metrics` | Prometheus scrape target |

## Secrets

`deploy/k8s/secret.example.yaml` documents the expected Secret shape and a
SealedSecret snippet. **Never commit populated secrets.** Production
installations should source secrets from one of:

- [SealedSecrets](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- A KMS-backed operator (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)

## Validation

```bash
helm lint deploy/helm/i3
helm template i3 deploy/helm/i3 | kubectl apply --dry-run=client -f -

kubectl apply --dry-run=client -k deploy/k8s
kubectl apply --dry-run=client -k deploy/k8s/overlays/dev
kubectl apply --dry-run=client -k deploy/k8s/overlays/staging
kubectl apply --dry-run=client -k deploy/k8s/overlays/prod
```
