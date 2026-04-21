variable "aws_region" {
  description = "AWS region where the EKS cluster lives."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Name of the existing EKS cluster to deploy into."
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for the I3 release."
  type        = string
  default     = "i3-prod"
}

variable "release_name" {
  description = "Helm release name."
  type        = string
  default     = "i3"
}

variable "chart_path" {
  description = "Path to the I3 Helm chart on disk (relative to this module)."
  type        = string
  default     = "../helm/i3"
}

variable "chart_version" {
  description = "Helm chart version to install. Leave empty to use the local path chart version."
  type        = string
  default     = "0.1.0"
}

variable "image_tag" {
  description = "Container image tag (e.g. '1.0.0' or a git SHA)."
  type        = string
  default     = "1.0.0"
}

variable "ingress_host" {
  description = "Public DNS hostname for the I3 ingress."
  type        = string
}

variable "anthropic_api_key" {
  description = "Anthropic API key (sensitive). Stored in-cluster as a Kubernetes Secret."
  type        = string
  sensitive   = true
}

variable "encryption_key" {
  description = "Fernet-compatible encryption key used by I3 ModelEncryptor (sensitive)."
  type        = string
  sensitive   = true
}

variable "extra_values" {
  description = "Arbitrary Helm chart overrides as a map of string key/value."
  type        = map(string)
  default     = {}
}
