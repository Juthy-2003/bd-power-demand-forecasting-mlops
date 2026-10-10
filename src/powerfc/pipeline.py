"""Pipeline steps as plain functions. Airflow tasks, the CLI and CI all call
these, so there is exactly one implementation of each step."""

import json
import logging
import os

import mlflow
import pandas as pd

from powerfc import config
from powerfc.data import store
from powerfc.data.clean import summary
from powerfc.registry import promote as gate
from powerfc.train import evaluate_recipe, gate_window, make_recipe, recipe_hash, train_and_register

log = logging.getLogger(__name__)


def ingest(day: str) -> dict:
    report = store.land_day(day)
    clean, version = store.assemble()
    report.update(data_version=version, **summary(clean))
    return report


def next_replay_day() -> str | None:
    """The day after the last landed one, or None when the holdout is used up."""
    from powerfc.data.load import read_raw

    days = store.landed_days()
    nxt = (pd.Timestamp(days[-1]) if days else store.AS_OF_START.normalize()) + pd.Timedelta(days=1)
    return None if nxt > read_raw()["datetime"].max().normalize() else f"{nxt:%Y-%m-%d}"


def ingest_for(logical_date: str) -> dict:
    """Land the logical date if the dataset has it, else replay the next day."""
    from powerfc.data.load import read_raw

    day = pd.Timestamp(logical_date).normalize()
    raw_end = read_raw()["datetime"].max().normalize()
    if store.AS_OF_START < day <= raw_end:
        return ingest(f"{day:%Y-%m-%d}")
    nxt = next_replay_day()
    if nxt is None:
        return {"skipped": "holdout replay finished; no new data"}
    return {"replayed": True, **ingest(nxt)}


def assemble() -> dict:
    clean, version = store.assemble()
    return {"data_version": version, **summary(clean)}


def train(weeks: int = 26, refit_every: int = 8, params: dict | None = None) -> dict:
    clean, version = store.load_clean()
    result = train_and_register(clean, version, make_recipe(params), weeks=weeks, refit_every=refit_every)
    return {k: v for k, v in result.items() if k != "metrics"} | {
        "smape": {b: round(result["metrics"][f"smape_{b}h"], 3) for b in config.BLOCKS},
        "coverage": {b: round(result["metrics"][f"coverage80_{b}h"], 3) for b in config.BLOCKS},
    }


def _evaluate_champion(recipe: dict, data_version: str, window: str) -> dict:
    """Backtest the champion's recipe on the challenger's data and folds, and
    log it as an evaluation run so the next comparison can reuse it."""
    clean, version = store.load_clean()
    if version != data_version:
        raise RuntimeError(f"local data {version} differs from challenger's {data_version}")
    weeks_part, refit_part = window.split("/")[0], window.split("refit")[-1]
    start, end = weeks_part.split("..")
    weeks = int((pd.Timestamp(end) - pd.Timestamp(start)).days / 7) + 1
    cfg = gate_window(clean, weeks, int(refit_part))
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"eval-champion-{recipe_hash(recipe)}-{data_version}"):
        mlflow.set_tags(
            {"recipe_hash": recipe_hash(recipe), "data_version": data_version, "window": window, "kind": "evaluation"}
        )
        _, metrics = evaluate_recipe(clean, recipe, cfg)
        mlflow.log_metrics(metrics)
    return metrics


def promote() -> dict:
    decision = gate.promote(evaluate_champion=_evaluate_champion)
    return {"promote": decision.promote, "kind": decision.kind, "reasons": decision.reasons}


def batch_forecast() -> dict:
    from powerfc.serve import store as forecast_store

    clean, version = store.load_clean()
    model = mlflow.pyfunc.load_model(f"models:/{config.REGISTERED_MODEL}@champion")
    mv = mlflow.MlflowClient().get_model_version_by_alias(config.REGISTERED_MODEL, "champion")
    fc = model.predict(clean)
    n = forecast_store.write_forecast(fc, model_version=mv.version, data_version=version)
    return {"rows": n, "origin": str(fc["origin"].iloc[0]), "model_version": mv.version, "data_version": version}


def reload_api() -> dict:
    """Ask the forecast API to re-read the champion from MLflow (after a promotion).
    Best effort: a missing API must not fail the training run."""
    import requests

    url = os.environ.get("POWERFC_API_URL", "http://api:8000")
    try:
        return requests.post(f"{url}/admin/reload", timeout=10).json()
    except requests.RequestException as exc:
        return {"reloaded": False, "error": str(exc)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="powerfc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("assemble")
    p = sub.add_parser("ingest")
    p.add_argument("--date", required=True)
    p = sub.add_parser("train")
    p.add_argument("--weeks", type=int, default=26)
    p.add_argument("--refit-every", type=int, default=8)
    p.add_argument("--params", default="{}", help="JSON LightGBM overrides (a new recipe)")
    sub.add_parser("promote")
    sub.add_parser("forecast")
    sub.add_parser("drift")
    sub.add_parser("replay-next", help="land the day after the last landed one")
    sub.add_parser("reload-api", help="tell the forecast API to reload the champion")
    args = ap.parse_args()
    if args.cmd == "fetch":
        from powerfc.data.load import fetch_raw

        out = fetch_raw()
    elif args.cmd == "assemble":
        out = assemble()
    elif args.cmd == "ingest":
        out = ingest(args.date)
    elif args.cmd == "train":
        out = train(args.weeks, args.refit_every, json.loads(args.params))
    elif args.cmd == "promote":
        out = promote()
    elif args.cmd == "forecast":
        out = batch_forecast()
    elif args.cmd == "replay-next":
        nxt = next_replay_day()
        out = ingest(nxt) if nxt else {"skipped": "holdout replay finished"}
    elif args.cmd == "reload-api":
        out = reload_api()
    else:
        from powerfc.monitoring.drift import run_drift_check

        out = run_drift_check()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
