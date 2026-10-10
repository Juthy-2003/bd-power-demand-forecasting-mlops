"""Airflow DAGs for the Bangladesh power demand pipeline. Two DAGs:

daily_pipeline           ingest → forecast → drift_check → drift_detected? → trigger_retrain
train_evaluate_promote   train → gate → forecast_if_promoted   (weekly, or triggered by drift)

Each ML task calls one function in `powerfc`, executed in a separate
virtualenv (/opt/powerfc-venv) with `@task.external_python`, so the ML
stack's dependencies never collide with Airflow's own. The functions hold
all the logic; the DAGs only order them.

The closed loop: `drift_check` returns `drift_detected` (PSI > 0.25 or error
ratio > 1.5, decided in powerfc.monitoring.drift.decide_retrain). The
`drift_detected` short-circuit task stops the run there when it is false;
when it is true, `trigger_retrain` starts a train_evaluate_promote run. No
human, no external alerting system.

Idempotency:
- ingest lands one day per run and overwrites that day's partition. With a
  live source the day would be the run's date. In holdout-replay mode (no
  live source; see powerfc.data.store) a date outside the dataset maps to
  "the next day not yet landed", so every run (scheduled or manual) replays
  one more day; once the holdout is exhausted, ingest is a no-op.
- forecast upserts on (origin, target hour, model version).
- train_evaluate_promote logs a new MLflow run each time, but promotion is a
  deterministic function of the metrics.
"""

from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task

VENV = "/opt/powerfc-venv/bin/python"
DEFAULT_ARGS = {"retries": 1, "retry_delay": timedelta(minutes=2)}
START = pendulum.datetime(2025, 3, 9, tz="Asia/Dhaka")  # the holdout starts 2025-03-09
RETRAIN_DAG = "train_evaluate_promote"


@dag(
    dag_id="daily_pipeline",
    schedule="@daily",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["powerfc"],
    doc_md="One day of the pipeline, in order: land and validate the day's PGCB readings, write the "
    "champion's 72-hour forecast, check drift, and trigger retraining if drift was detected. "
    "Trigger it by hand to replay the next holdout day.",
)
def daily_pipeline():
    @task.external_python(python=VENV)
    def ingest(day: str) -> dict:
        from powerfc.pipeline import ingest_for

        return ingest_for(day)

    @task.external_python(python=VENV)
    def forecast(_landed: dict) -> dict:
        from powerfc.pipeline import batch_forecast

        return batch_forecast()

    @task.external_python(python=VENV)
    def drift_check(_forecast: dict) -> dict:
        from powerfc.monitoring.drift import run_drift_check

        return run_drift_check()

    @task.short_circuit
    def drift_detected(result: dict) -> bool:
        """True → continue to trigger_retrain; False → skip it."""
        print("drift reasons:", result["reasons"] or "none")
        return bool(result["drift_detected"])

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=RETRAIN_DAG,
        conf={"reason": "drift detected by daily_pipeline run {{ run_id }}"},
    )

    # The venv has no Airflow, so the run's date is passed as a rendered template.
    landed = ingest(day="{{ dag_run.run_after | ds }}")
    drift_detected(drift_check(forecast(landed))) >> trigger_retrain


@dag(
    dag_id=RETRAIN_DAG,
    schedule="@weekly",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["powerfc"],
    doc_md="Train the challenger, backtest it with rolling origins, log to MLflow, register it, then let "
    "the promotion gate decide. Runs weekly, and whenever daily_pipeline detects drift.",
)
def train_evaluate_promote():
    @task.external_python(python=VENV)
    def train() -> dict:
        from powerfc.pipeline import train as train_step

        return train_step()

    @task.external_python(python=VENV)
    def gate(trained: dict) -> dict:
        from powerfc.pipeline import promote

        decision = promote()
        decision["challenger_version"] = trained.get("version")
        return decision

    @task.external_python(python=VENV)
    def forecast_if_promoted(decision: dict) -> dict:
        if not decision["promote"]:
            return {"skipped": "champion unchanged", "reasons": decision["reasons"]}
        from powerfc.pipeline import batch_forecast, reload_api

        return {**batch_forecast(), "api_reload": reload_api()}

    forecast_if_promoted(gate(train()))


daily_pipeline()
train_evaluate_promote()
