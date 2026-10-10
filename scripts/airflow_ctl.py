"""Tiny Airflow 3 REST client for the Makefile and demos.

python scripts/airflow_ctl.py unpause train_evaluate_promote
python scripts/airflow_ctl.py run train_evaluate_promote          # trigger and wait
python scripts/airflow_ctl.py runs train_evaluate_promote
"""

import json
import os
import sys
import time

import requests

URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
USER, PASSWORD = os.environ.get("AIRFLOW_USERNAME", "admin"), os.environ.get("AIRFLOW_PASSWORD", "admin")


def headers() -> dict:
    for _ in range(60):
        try:
            r = requests.post(f"{URL}/auth/token", json={"username": USER, "password": PASSWORD}, timeout=5)
            if r.ok:
                return {"Authorization": f"Bearer {r.json()['access_token']}"}
        except requests.RequestException:
            pass
        time.sleep(3)
    raise SystemExit("Airflow API not reachable")


def api(method: str, path: str, **kw):
    r = requests.request(method, f"{URL}/api/v2{path}", headers=headers(), timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def wait_for_dag(dag_id: str) -> None:
    for _ in range(60):
        try:
            api("GET", f"/dags/{dag_id}")
            return
        except requests.HTTPError:
            time.sleep(3)
    raise SystemExit(f"DAG {dag_id} not found")


def unpause(dag_id: str) -> None:
    wait_for_dag(dag_id)
    api("PATCH", f"/dags/{dag_id}", json={"is_paused": False})
    print(f"unpaused {dag_id}")


def run(dag_id: str, timeout: int = 1800, conf: dict | None = None) -> dict:
    unpause(dag_id)
    started = time.time()
    dr = api("POST", f"/dags/{dag_id}/dagRuns", json={"logical_date": None, "conf": conf or {}})
    run_id = dr["dag_run_id"]
    print(f"triggered {dag_id} {run_id}")
    while time.time() - started < timeout:
        state = api("GET", f"/dags/{dag_id}/dagRuns/{run_id}")["state"]
        if state in ("success", "failed"):
            break
        time.sleep(5)
    tis = api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")["task_instances"]
    out = {
        "dag_id": dag_id,
        "run_id": run_id,
        "state": state,
        "seconds": round(time.time() - started),
        "tasks": {t["task_id"]: t["state"] for t in tis},
    }
    for t in tis:
        try:
            xcom = api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{t['task_id']}/xcomEntries/return_value")
            out.setdefault("results", {})[t["task_id"]] = xcom.get("value")
        except requests.HTTPError:
            pass
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    cmd, dag = sys.argv[1], sys.argv[2]
    if cmd == "unpause":
        unpause(dag)
    elif cmd == "run":
        result = run(dag)
        sys.exit(0 if result["state"] == "success" else 1)
    elif cmd == "runs":
        print(json.dumps(api("GET", f"/dags/{dag}/dagRuns?order_by=-start_date&limit=5"), indent=2))
