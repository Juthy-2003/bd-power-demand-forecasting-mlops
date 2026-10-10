# 0005 — k3s on one Oracle ARM VM; CronJobs instead of Airflow on the node

**Status:** Accepted for design (2026-10-10). Not applied: no Oracle account yet.

**Amended (2026-10-10, trim, ADR 0009):** the node no longer runs Prometheus, Alertmanager or Grafana. The daily CronJobs were merged into one (`replay-next && forecast && drift`, mirroring Airflow's `daily_pipeline`), and the weekly training CronJob now ends with `powerfc reload-api` so the API shows the new champion. On Kubernetes, drift is checked and logged daily but does not trigger retraining (only Airflow closes the loop); retraining there is weekly. Project 3 no longer shares this node or MLflow.

## Decision
- One OCI Always Free Ampere A1 VM (2 OCPU / 12 GB, the 2026 limit reported by a secondary source — verify on Oracle's page) running k3s, provisioned with Terraform (`infra/terraform`, `terraform validate` passes).
- The same Helm chart (`deploy/helm/powerfc`) deploys locally to k3d and to the node.
- On the node, the pipeline runs as Kubernetes **CronJobs** calling the same `powerfc` CLI (daily: ingest → forecast → drift; weekly: train → promote → forecast → reload the API). Airflow stays the orchestrator in docker compose. This answers v01 §18 Q10: Airflow 3 standalone needs a 4.7 GB image and its own scheduler, API server and DAG processor processes on a 12 GB node shared with Project 3.
- Images are built for linux/amd64 and linux/arm64 in CI.

## Alternatives
EKS (no free control plane), GKE (free tier covers the management fee, not compute), Airflow on the node (memory).

## Consequences
- ✅ ≈$0/month if the Always Free limits hold.
- ⚠️ CronJobs have no DAG view, retries per task or backfills; the trade-off is written down rather than hidden.
