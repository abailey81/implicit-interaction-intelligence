output "namespace" {
  description = "Namespace in which the release was installed."
  value       = module.i3_app.namespace
}

output "release_name" {
  description = "Helm release name."
  value       = module.i3_app.release_name
}

output "release_status" {
  description = "Final status of the Helm release."
  value       = module.i3_app.release_status
}

output "ingress_host" {
  description = "Public ingress hostname."
  value       = module.i3_app.ingress_host
}
