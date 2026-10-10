# 0003 — Airflow 3 with the pipeline in its own virtualenv

**Status:** Accepted (2026-10-10)

## Context
Airflow 3.3 pins many libraries. The pipeline needs pandas 3, MLflow 3.17, LightGBM and Evidently. Installing both into one environment invites version conflicts that break either Airflow or the model.

## Decision
`Dockerfile.airflow` installs the `powerfc` package into `/opt/powerfc-venv`. Every DAG task is `@task.external_python(python="/opt/powerfc-venv/bin/python")` and only calls one function from `powerfc.pipeline`. Airflow runs as `standalone` with LocalExecutor and Postgres, which is the lightest configuration that still uses a real database.

The venv has no Airflow inside, so Airflow context is not injected; the logical date is passed as a rendered template (`ds="{{ ds }}"`) and return values must be plain JSON (a pandas Timestamp in a return value failed XCom on 2026-10-10 and was fixed).

## Consequences
- ✅ The DAGs are thin; the same functions run from the CLI, CI and Kubernetes CronJobs.
- ⚠️ The Airflow image is large (≈4.7 GB locally).
- ⚠️ In holdout-replay mode, the logical date is not the data date (see `powerfc.pipeline.ingest_for`).
