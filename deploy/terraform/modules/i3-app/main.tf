terraform {
  required_version = ">= 1.6"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13"
    }
  }
}

# ---------------------------------------------------------------------------
# In-cluster Secret with the sensitive env vars. Rendered out-of-band so we
# can reference it from values (envFromSecret.existingSecret) without passing
# plaintext through Helm --set.
# ---------------------------------------------------------------------------
resource "kubernetes_namespace" "this" {
  count = var.create_namespace ? 1 : 0
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"    = "i3"
      "app.kubernetes.io/part-of" = "i3"
    }
  }
}

resource "kubernetes_secret" "i3_secrets" {
  metadata {
    name      = "i3-secrets"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"       = "i3"
      "app.kubernetes.io/part-of"    = "i3"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  type = "Opaque"

  data = {
    I3_ENCRYPTION_KEY = var.encryption_key
    ANTHROPIC_API_KEY = var.anthropic_api_key
  }

  depends_on = [kubernetes_namespace.this]
}

# ---------------------------------------------------------------------------
# Helm release — the chart under deploy/helm/i3 with prod-flavour overrides.
# ---------------------------------------------------------------------------
resource "helm_release" "i3" {
  name             = var.release_name
  namespace        = var.namespace
  chart            = var.chart_path
  version          = var.chart_version
  create_namespace = false
  atomic           = true
  cleanup_on_fail  = true
  wait             = true
  timeout          = var.timeout_seconds

  values = [
    for f in var.values_files : file(f)
  ]

  set {
    name  = "image.tag"
    value = var.image_tag
  }

  set {
    name  = "envFromSecret.existingSecret"
    value = kubernetes_secret.i3_secrets.metadata[0].name
  }

  set {
    name  = "ingress.hosts[0].host"
    value = var.ingress_host
  }

  set {
    name  = "ingress.hosts[0].paths[0].path"
    value = "/"
  }

  set {
    name  = "ingress.hosts[0].paths[0].pathType"
    value = "Prefix"
  }

  set {
    name  = "ingress.tls[0].secretName"
    value = "i3-tls"
  }

  set {
    name  = "ingress.tls[0].hosts[0]"
    value = var.ingress_host
  }

  dynamic "set" {
    for_each = var.extra_values
    content {
      name  = set.key
      value = set.value
    }
  }

  depends_on = [
    kubernetes_namespace.this,
    kubernetes_secret.i3_secrets,
  ]
}
