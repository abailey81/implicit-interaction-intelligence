# Kubernetes

A reference Kubernetes deployment for I³: a `Deployment`, a `Service`, a
sticky-session `Ingress`, `Secrets` for the Fernet/Anthropic keys, and
`ServiceMonitor` for Prometheus.

!!! warning "One logical user per pod"
    The in-process pipeline caches per-user session state. Route each
    `user_id` to a stable pod via sticky sessions (cookie or hash) — see
    the ingress block below.

## Namespace and secrets { #secrets }

```yaml title="01-namespace.yaml"
apiVersion: v1
kind: Namespace
metadata:
  name: i3
---
apiVersion: v1
kind: Secret
metadata:
  name: i3-secrets
  namespace: i3
type: Opaque
stringData:
  I3_ENCRYPTION_KEY: "REPLACE_WITH_32_URL_SAFE_BASE64_BYTES"
  ANTHROPIC_API_KEY: "sk-ant-..."
```

!!! tip "Secret management"
    Prefer `ExternalSecrets`, `SealedSecrets`, or `SOPS` over raw
    `Secret` manifests.

## ConfigMap { #configmap }

```yaml title="02-configmap.yaml"
apiVersion: v1
kind: ConfigMap
metadata:
  name: i3-config
  namespace: i3
data:
  I3_CORS_ORIGINS: "https://i3.example.org"
  I3_LOG_LEVEL: "INFO"
  OTEL_SERVICE_NAME: "i3"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.obs.svc:4317"
```

## Deployment { #deployment }

```yaml title="03-deployment.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: i3
  namespace: i3
  labels: {app: i3}
spec:
  replicas: 3
  revisionHistoryLimit: 3
  strategy:
    type: RollingUpdate
    rollingUpdate: {maxSurge: 1, maxUnavailable: 0}
  selector: {matchLabels: {app: i3}}
  template:
    metadata:
      labels: {app: i3}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      serviceAccountName: i3
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: {type: RuntimeDefault}
      containers:
        - name: i3
          image: ghcr.io/abailey81/implicit-interaction-intelligence:1.0.0
          imagePullPolicy: IfNotPresent
          ports:
            - {name: http, containerPort: 8000}
          envFrom:
            - configMapRef: {name: i3-config}
            - secretRef:    {name: i3-secrets}
          resources:
            requests: {cpu: 500m, memory: 512Mi}
            limits:   {cpu: "2",  memory: 2Gi}
          readinessProbe:
            httpGet:  {path: /health, port: http}
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:  {path: /health, port: http}
            periodSeconds: 20
            failureThreshold: 5
            initialDelaySeconds: 15
          startupProbe:
            httpGet:  {path: /health, port: http}
            periodSeconds: 5
            failureThreshold: 60
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: [ALL]}
          volumeMounts:
            - {name: data,  mountPath: /app/data}
            - {name: tmp,   mountPath: /tmp}
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: i3-data}
        - name: tmp
          emptyDir: {sizeLimit: 64Mi}
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: {matchLabels: {app: i3}}
```

## PVC, Service, Ingress { #pvc-svc-ing }

```yaml title="04-pvc-svc-ingress.yaml"
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: i3-data, namespace: i3}
spec:
  accessModes: [ReadWriteOnce]
  resources: {requests: {storage: 5Gi}}
---
apiVersion: v1
kind: Service
metadata: {name: i3, namespace: i3}
spec:
  selector: {app: i3}
  ports:
    - {name: http, port: 80, targetPort: http}
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP: {timeoutSeconds: 3600}
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: i3
  namespace: i3
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "i3-sticky"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      location ~* ^/demo/ { return 404; }
spec:
  ingressClassName: nginx
  tls:
    - hosts: [i3.example.org]
      secretName: i3-tls
  rules:
    - host: i3.example.org
      http:
        paths:
          - {path: /, pathType: Prefix, backend: {service: {name: i3, port: {number: 80}}}}
```

## ServiceMonitor { #servicemonitor }

```yaml title="05-servicemonitor.yaml"
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: i3
  namespace: i3
  labels: {release: kube-prometheus-stack}
spec:
  selector: {matchLabels: {app: i3}}
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

## PodDisruptionBudget and HPA { #pdb-hpa }

```yaml title="06-pdb-hpa.yaml"
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: {name: i3, namespace: i3}
spec:
  minAvailable: 2
  selector: {matchLabels: {app: i3}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: i3, namespace: i3}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: i3
  minReplicas: 3
  maxReplicas: 12
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
    - type: Pods
      pods:
        metric: {name: i3_session_active}
        target: {type: AverageValue, averageValue: "50"}
```

## NetworkPolicy { #network-policy }

```yaml title="07-networkpolicy.yaml"
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: i3, namespace: i3}
spec:
  podSelector: {matchLabels: {app: i3}}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - namespaceSelector: {matchLabels: {name: ingress-nginx}}
      ports:
        - {protocol: TCP, port: 8000}
  egress:
    - to:
        - namespaceSelector: {matchLabels: {name: obs}}
      ports:
        - {protocol: TCP, port: 4317}
    - to:
        - ipBlock: {cidr: 0.0.0.0/0}   # Anthropic API egress
      ports:
        - {protocol: TCP, port: 443}
    - ports:
        - {protocol: UDP, port: 53}    # DNS
```

## Rollout and rollback { #rollout }

```bash
# Apply
kubectl apply -n i3 -f .

# Status
kubectl -n i3 rollout status deploy/i3

# Rollback
kubectl -n i3 rollout undo deploy/i3
```

## Operational checks { #checks }

- [x] All pods `Ready`.
- [x] `i3_privacy_auditor_hits_total` stays at 0.
- [x] `/health` on each pod returns 200.
- [x] HPA cpu target between 30–70 %.
- [x] Sticky sessions route consistent `user_id` → pod.
- [x] Backups: periodic `velero backup create i3-data` or equivalent.

## Further reading { #further }

- [Deployment](deployment.md) — systemd / non-K8s variant.
- [Observability](observability.md) — dashboards + alerts.
- [Runbook](runbook.md) — incident response.
