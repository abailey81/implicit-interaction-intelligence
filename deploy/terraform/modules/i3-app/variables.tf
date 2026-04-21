variable "release_name" {
  description = "Helm release name."
  type        = string
  default     = "i3"
}

variable "namespace" {
  description = "Kubernetes namespace. Created if it does not exist."
  type        = string
}

variable "chart_path" {
  description = "Filesystem path to the Helm chart (relative to the caller)."
  type        = string
}

variable "chart_version" {
  description = "Chart version. Must match Chart.yaml when installing from a local path."
  type        = string
}

variable "image_tag" {
  description = "Container image tag."
  type        = string
}

variable "ingress_host" {
  description = "Public DNS hostname for the ingress."
  type        = string
}

variable "anthropic_api_key" {
  description = "Anthropic API key (sensitive)."
  type        = string
  sensitive   = true
}

variable "encryption_key" {
  description = "Fernet encryption key (sensitive)."
  type        = string
  sensitive   = true
}

variable "extra_values" {
  description = "Additional Helm --set overrides as a map."
  type        = map(string)
  default     = {}
}

variable "values_files" {
  description = "Additional Helm -f files (absolute or module-relative paths)."
  type        = list(string)
  default     = []
}

variable "timeout_seconds" {
  description = "Helm release timeout in seconds."
  type        = number
  default     = 600
}

variable "create_namespace" {
  description = "Whether to create the namespace via the helm_release resource."
  type        = bool
  default     = true
}
