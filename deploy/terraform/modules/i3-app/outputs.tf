output "namespace" {
  description = "Namespace of the release."
  value       = var.namespace
}

output "release_name" {
  description = "Helm release name."
  value       = helm_release.i3.name
}

output "release_status" {
  description = "Final release status as reported by Helm."
  value       = helm_release.i3.status
}

output "ingress_host" {
  description = "Public ingress hostname."
  value       = var.ingress_host
}

output "secret_name" {
  description = "Name of the Kubernetes Secret containing sensitive env vars."
  value       = kubernetes_secret.i3_secrets.metadata[0].name
}
