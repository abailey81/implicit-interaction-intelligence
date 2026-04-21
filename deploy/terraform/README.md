# I³ Terraform (AWS EKS reference)

Minimal Terraform scaffolding that deploys the I³ Helm chart onto a **pre-existing**
AWS EKS cluster. Cluster provisioning itself is out of scope — plug in your
preferred EKS module (e.g., `terraform-aws-modules/eks/aws`) upstream.

## Layout

| Path | Purpose |
| --- | --- |
| `versions.tf` | Required Terraform + provider versions |
| `variables.tf` | Root inputs |
| `main.tf` | Providers + call into the `i3-app` module |
| `outputs.tf` | Release metadata |
| `modules/i3-app/` | Reusable module: Secret + Helm release |

## Requirements

- Terraform `>= 1.6`
- AWS provider `~> 5.40`
- Kubernetes provider `~> 2.27`
- Helm provider `~> 2.13`
- Network access to the EKS control plane (VPN/bastion or public endpoint)
- IAM permissions to describe the target EKS cluster

## Usage

```hcl
terraform init

terraform apply \
  -var aws_region=us-east-1 \
  -var cluster_name=my-eks \
  -var namespace=i3-prod \
  -var image_tag=1.0.0 \
  -var ingress_host=i3.example.com \
  -var anthropic_api_key="$ANTHROPIC_API_KEY" \
  -var encryption_key="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

To pass arbitrary chart overrides, supply `extra_values`:

```bash
terraform apply \
  ... \
  -var 'extra_values={"replicaCount":"5","autoscaling.maxReplicas":"30"}'
```

## Secret management

The root module materialises a Kubernetes Secret named `i3-secrets` in the
release namespace and wires the Helm chart's `envFromSecret.existingSecret`
at it. Sensitive variables are marked `sensitive = true` so they are redacted
in plan/apply output, but they **do** land in Terraform state — use a remote
backend with encryption at rest (e.g., S3 + KMS + DynamoDB locking).

For higher assurance, replace the `kubernetes_secret` resource with a
[SealedSecret](https://github.com/bitnami-labs/sealed-secrets) or an
[ExternalSecret](https://external-secrets.io/) and drop the sensitive
variables from Terraform entirely.

## Outputs

| Output | Description |
| --- | --- |
| `namespace` | Release namespace |
| `release_name` | Helm release name |
| `release_status` | `deployed` on success |
| `ingress_host` | Public hostname (echoed for convenience) |
