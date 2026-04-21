{{/*
Common template helpers for the I3 chart.
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "i3.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
If release name contains chart name, it is used as a full name.
*/}}
{{- define "i3.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart name + version label.
*/}}
{{- define "i3.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "i3.labels" -}}
helm.sh/chart: {{ include "i3.chart" . }}
{{ include "i3.selectorLabels" . }}
app.kubernetes.io/component: server
app.kubernetes.io/part-of: i3
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "i3.selectorLabels" -}}
app.kubernetes.io/name: {{ include "i3.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
ServiceAccount name.
*/}}
{{- define "i3.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "i3.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Secret name to reference via envFrom. If .Values.envFromSecret.data is set,
the chart creates its own Secret named <fullname>-env; otherwise it uses the
existingSecret value.
*/}}
{{- define "i3.envSecretName" -}}
{{- if and .Values.envFromSecret.data (gt (len .Values.envFromSecret.data) 0) -}}
{{- printf "%s-env" (include "i3.fullname" .) -}}
{{- else -}}
{{- required "envFromSecret.existingSecret must be set when envFromSecret.data is empty" .Values.envFromSecret.existingSecret -}}
{{- end -}}
{{- end -}}

{{/*
Image reference with a safe fallback to Chart.AppVersion when tag is empty.
*/}}
{{- define "i3.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
