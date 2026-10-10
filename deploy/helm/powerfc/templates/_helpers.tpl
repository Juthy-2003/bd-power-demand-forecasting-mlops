{{- define "powerfc.labels" -}}
app.kubernetes.io/part-of: powerfc
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "powerfc.pipelineEnv" -}}
- name: MLFLOW_TRACKING_URI
  value: http://mlflow:5000
- name: POWERFC_DB
  value: postgresql://powerfc:{{ .Values.postgres.password }}@postgres:5432/powerfc
- name: POWERFC_DATA
  value: /data
{{- end }}
