# Memory budget (measured)

## docker compose, after the trim (2026-10-10)

Measured with `docker stats` on an Apple M4 laptop (Docker Desktop VM with 7.75 GiB), after the closed-loop re-run: 4 containers, fresh volumes, 17 replayed days and 6 model versions in MLflow.

| Container | Idle | Peak seen | Note |
|---|---|---|---|
| Airflow 3.3 (`standalone`: scheduler, API server, DAG processor, triggerer) | 1.30 GiB | 1.74 GiB during a `train_evaluate_promote` run; 1.62 GiB during a `daily_pipeline` run | the ML tasks run inside this container (`@task.external_python`) |
| MLflow server (1 worker, job execution off) | 453 MiB | 456 MiB | |
| Postgres 17 (airflow, mlflow, powerfc DBs) | 203 MiB | 208 MiB | |
| Forecast API | 135 MiB | 145 MiB | |
| **Total** | **≈ 2.1 GiB** | **≈ 2.5 GiB** (sum of per-container peaks) | |

Peaks are the largest of coarse samples (one `docker stats` reading every ~2 s), so a short spike can be missed. Before the trim, the core compose stack also ran Prometheus, Alertmanager and Grafana, and the `stream` profile added Redpanda, TimescaleDB and the scorer; their compose memory was never measured.

## k3d — pre-trim history (not re-measured)

Measured on 2026-10-10 **before the trim** on local k3d (k3s in Docker on the same laptop), steady state after the bootstrap job, with `kubectl top`. The trimmed chart was linted and rendered but not re-deployed, so this table was not re-measured. Rows marked *removed* no longer exist in the chart; the forecast API has changed since (no `prometheus-client`, no metrics endpoint), so its figure is pre-trim.

| Component | Measured (pre-trim) | Limit in the chart | Status |
|---|---|---|---|
| MLflow server (1 worker, job execution off) | 366 Mi | 768 Mi | unchanged component |
| Forecast API | 126 Mi | 512 Mi | changed since (smaller code path) |
| Grafana | 109 Mi | — | *removed* |
| Postgres | 55 Mi | 512 Mi | unchanged component |
| Prometheus (7-day retention) | 28 Mi | — | *removed* |
| Alertmanager | 17 Mi | — | *removed* |
| Training CronJob (weekly) | ≈ 0.45 Gi peak seen (coarse sampling) | 2 Gi | runs then exits |
| Whole k3d node incl. k3s and system pods | 2.1 Gi | | |

Two findings from that run still shape the chart:
- **MLflow 3.17 starts a background job system** (a job runner plus ~8 huey consumers): 1.62 GB measured, and it was OOM-killed at 768 Mi and again at 1 Gi. `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` brings it to 349 MB. This project does not use MLflow jobs.
- **Airflow is not on the node.** The Airflow 3 image is ≈4.7 GB and standalone runs a scheduler, API server, DAG processor and triggerer (1.3–1.7 GiB in compose, above); the node runs the same CLI steps as CronJobs instead (ADR 0005).

Project 3 no longer shares the node or MLflow, so the "headroom for Project 3" figure from the first version no longer applies.
