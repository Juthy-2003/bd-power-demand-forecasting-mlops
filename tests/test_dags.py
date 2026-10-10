"""DAG structure. Needs Airflow, which is not in the project venv (it lives in
the Airflow image), so this is skipped by `make test` and run inside the
Airflow container. That image has no pytest, so on 2026-10-10 the checks were
called directly with a stub `pytest` module:

    docker cp tests/test_dags.py powerfc-airflow-1:/tmp/
    docker exec -w /tmp powerfc-airflow-1 python -c "
    import sys, types
    sys.modules['pytest'] = types.SimpleNamespace(importorskip=__import__, fixture=lambda **k: (lambda f: f))
    import test_dags as t; bag = t.dagbag()
    for n in dir(t):
        if n.startswith('test_'): getattr(t, n)(bag); print('PASS', n)"
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402  (Airflow 3.3 location)

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
if not DAGS_DIR.exists():  # inside the Airflow container
    DAGS_DIR = Path("/opt/airflow/dags")


@pytest.fixture(scope="module")
def dagbag():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert not bag.import_errors, bag.import_errors
    return bag


def test_only_two_dags(dagbag):
    assert set(dagbag.dag_ids) == {"daily_pipeline", "train_evaluate_promote"}


def test_daily_steps_wait_for_each_other(dagbag):
    dag = dagbag.get_dag("daily_pipeline")
    order = ["ingest", "forecast", "drift_check", "drift_detected", "trigger_retrain"]
    for upstream, downstream in zip(order, order[1:], strict=False):
        assert dag.get_task(upstream).downstream_task_ids == {downstream}


def test_drift_triggers_the_retrain_dag(dagbag):
    trigger = dagbag.get_dag("daily_pipeline").get_task("trigger_retrain")
    assert trigger.trigger_dag_id == "train_evaluate_promote"


def test_retrain_dag_order(dagbag):
    dag = dagbag.get_dag("train_evaluate_promote")
    assert dag.get_task("train").downstream_task_ids == {"gate"}
    assert dag.get_task("gate").downstream_task_ids == {"forecast_if_promoted"}
