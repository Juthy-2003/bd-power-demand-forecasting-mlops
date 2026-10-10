# 0007 — MLflow serves its own artifacts (no MinIO)

**Status:** Accepted (2026-10-10)

## Context
The plan used MinIO as an S3-compatible artifact store. On 2026-10-10 the `minio/minio` and `minio/mc` images were no longer available on Docker Hub ("repository does not exist").

## Decision
`mlflow server --artifacts-destination /mlartifacts --serve-artifacts`: artifacts live on a volume (a PVC on Kubernetes) and clients upload/download through the tracking server, so no client needs storage credentials. Postgres holds runs and the registry.

## Consequences
- ✅ One fewer service and no credentials to distribute.
- ⚠️ Artifact traffic goes through the MLflow server. Moving to S3 or OCI Object Storage later changes only `--artifacts-destination`.
