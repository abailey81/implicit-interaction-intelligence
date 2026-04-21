#
# Reference Terraform wiring to deploy the I3 Helm chart onto an existing AWS
# EKS cluster. This file is intentionally minimal — consumers should adapt
# provider authentication, backend state, and IAM to their own environment.
#
# Usage:
#   terraform init
#   terraform apply \
#       -var cluster_name=my-eks \
#       -var ingress_host=i3.example.com \
#       -var anthropic_api_key=$(pass anthropic/api-key) \
#       -var encryption_key=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
#

provider "aws" {
  region = var.aws_region
}

data "aws_eks_cluster" "this" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "this" {
  name = var.cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.this.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.this.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

module "i3_app" {
  source = "./modules/i3-app"

  release_name      = var.release_name
  namespace         = var.namespace
  chart_path        = var.chart_path
  chart_version     = var.chart_version
  image_tag         = var.image_tag
  ingress_host      = var.ingress_host
  anthropic_api_key = var.anthropic_api_key
  encryption_key    = var.encryption_key
  extra_values      = var.extra_values
}
