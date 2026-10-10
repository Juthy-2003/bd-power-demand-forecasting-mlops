"""Daily drift check: are today's inputs and errors still like the champion's?

Two signals (answers to v01 §18 Q9, decided 2026-10-10):

1. Data drift — the inputs moved. Evidently's DataDriftPreset runs per-feature
   tests (K-S on these sample sizes) for the report; the *alert* uses PSI,
   because K-S p-values flag tiny, harmless shifts once samples are large.
   PSI > 0.25 on any monitored feature is a significant shift.
2. Concept drift — the input→demand relationship moved. Measured as the
   champion's sMAPE on recent hours that now have actuals, divided by its own
   backtest sMAPE ("error ratio"). A ratio above 1.5 means it is doing half
   again as badly as it promised.

Reference window: the 8 weeks of data before the champion's training cut-off.
Current window: data the champion has NOT seen — the hours after its training
cut-off, at most the latest 14 days. No verdict until there are at least 7
days of such data (MIN_CURRENT_DAYS). Before 2026-10-10 the current window
was simply "the latest 14 days"; right after training that window is mostly
the champion's own training data, and on the first replayed day it already
reported PSI 5.83 and triggered a retrain. Waiting for 7 unseen days also
gives a natural one-week pause after every promotion.

The decision is made here, in plain Python (`decide_retrain`): drift is
detected when PSI > 0.25 on any monitored feature OR the error ratio > 1.5.
The daily Airflow DAG reads `drift_detected` from this function's result and,
if it is true, triggers the train_evaluate_promote DAG. Nothing else decides.
"""

import json

import numpy as np
import pandas as pd

from powerfc import config
from powerfc.data.store import load_clean
from powerfc.evaluate import metrics
from powerfc.features.build import block_features

MONITORED = ["level", "lag_24", "lag_168", "mean_24", "max_24", "min_24", "same_hour_mean_7d", "shed_share_24"]
REFERENCE_WEEKS = 8
CURRENT_DAYS = 14
MIN_CURRENT_DAYS = 7
PSI_ALERT = 0.25
ERROR_RATIO_ALERT = 1.5


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over reference-quantile bins."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < bins or len(cur) < bins:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def decide_retrain(psi_max: float | None, error_ratio: float | None) -> tuple[bool, list[str]]:
    """The retraining rule. Either signal on its own is enough."""
    reasons = []
    if psi_max is not None and psi_max > PSI_ALERT:
        reasons.append(f"data drift: max PSI {psi_max:.2f} > {PSI_ALERT}")
    if error_ratio is not None and error_ratio > ERROR_RATIO_ALERT:
        reasons.append(f"performance drift: error ratio {error_ratio:.2f} > {ERROR_RATIO_ALERT}")
    return bool(reasons), reasons


def windows(clean: pd.DataFrame, reference_end: pd.Timestamp):
    data_end = clean["datetime"].max()
    first_unseen = reference_end + pd.Timedelta(hours=1)
    current = (max(first_unseen, data_end - pd.Timedelta(days=CURRENT_DAYS) + pd.Timedelta(hours=1)), data_end)
    reference = (reference_end - pd.Timedelta(weeks=REFERENCE_WEEKS) + pd.Timedelta(hours=1), reference_end)
    return reference, current


def data_drift(clean: pd.DataFrame, reference_end: pd.Timestamp) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    feats = block_features(clean, 24)[MONITORED]
    (r0, r1), (c0, c1) = windows(clean, reference_end)
    ref, cur = feats.loc[r0:r1].dropna(), feats.loc[c0:c1].dropna()
    snapshot = Report([DataDriftPreset()]).run(cur, ref)
    tests = {}
    for m in snapshot.dict()["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            # p-value tests drift when value < threshold; distance methods
            # (Wasserstein, Jensen-Shannon, PSI) drift when value >= threshold.
            is_p_value = "p_value" in cfg["method"]
            drifted = m["value"] < cfg["threshold"] if is_p_value else m["value"] >= cfg["threshold"]
            tests[cfg["column"]] = {"method": cfg["method"], "value": m["value"], "drifted": bool(drifted)}
    psis = {c: round(psi(ref[c].to_numpy(), cur[c].to_numpy()), 4) for c in MONITORED}
    finite = [v for v in psis.values() if not np.isnan(v)]
    return {
        "reference_start": r0,
        "reference_end": r1,
        "current_start": c0,
        "current_end": c1,
        "drift_share": float(np.mean([t["drifted"] for t in tests.values()])) if tests else 0.0,
        "drifted_features": int(sum(t["drifted"] for t in tests.values())),
        "n_features": len(MONITORED),
        "psi_max": max(finite) if finite else 0.0,
        "details": {"psi": psis, "ks": tests},
        "_snapshot": snapshot,
    }


def performance_drift(clean: pd.DataFrame, forecasts: pd.DataFrame, max_lead: int = 24) -> dict:
    """Champion error on recent hours whose actuals have arrived."""
    if forecasts.empty:
        return {"rolling_smape": None, "rolling_coverage": None, "scored_hours": 0}
    fc = forecasts[forecasts["lead"] <= max_lead].sort_values("origin").drop_duplicates("target_time", keep="last")
    actual = clean.set_index("datetime")["demand_mw"]
    fc = fc.assign(actual=actual.reindex(pd.DatetimeIndex(fc["target_time"])).to_numpy()).dropna(subset=["actual"])
    if fc.empty:
        return {"rolling_smape": None, "rolling_coverage": None, "scored_hours": 0}
    return {
        "rolling_smape": metrics.smape(fc.actual, fc.p50),
        "rolling_coverage": metrics.coverage(fc.actual, fc.p10, fc.p90),
        "scored_hours": int(len(fc)),
    }


def run_drift_check(save_report: bool = True) -> dict:
    import mlflow
    from mlflow import MlflowClient

    from powerfc.serve import store

    clean, _ = load_clean()
    client = MlflowClient()
    champ = client.get_model_version_by_alias(config.REGISTERED_MODEL, "champion")
    run = client.get_run(champ.run_id)
    reference_end = pd.Timestamp(run.data.params["data_end"])
    champion_smape = run.data.metrics.get("smape_24h")

    unseen_hours = int((clean["datetime"].max() - reference_end) / pd.Timedelta(hours=1))
    if unseen_hours < MIN_CURRENT_DAYS * 24:
        return {
            "champion_version": champ.version,
            "unseen_hours": unseen_hours,
            "drift_detected": False,
            "reasons": [f"waiting: {unseen_hours} h after the champion's cut-off, need {MIN_CURRENT_DAYS * 24}"],
        }

    result = data_drift(clean, reference_end)
    snapshot = result.pop("_snapshot")
    (c0, c1) = result["current_start"], result["current_end"]
    # Score only the champion's own forecasts: right after a promotion the old
    # champion's errors must not count against the new one.
    fc = store.forecasts_between(c0, c1 + pd.Timedelta(hours=1))
    if not fc.empty:
        fc = fc[fc["model_version"] == str(champ.version)]
    perf = performance_drift(clean, fc)
    result.update({k: perf[k] for k in ("rolling_smape", "rolling_coverage")})
    ratio = perf["rolling_smape"] / champion_smape if perf["rolling_smape"] and champion_smape else None
    drift_detected, reasons = decide_retrain(result["psi_max"], ratio)

    if save_report:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT + "-monitoring")
        with mlflow.start_run(run_name=f"drift-{c1:%Y-%m-%d}"):
            mlflow.set_tags({"kind": "drift", "champion_version": champ.version, "drift_detected": drift_detected})
            mlflow.log_metrics(
                {
                    "psi_max": result["psi_max"],
                    "drift_share": result["drift_share"],
                    **({"rolling_smape": perf["rolling_smape"]} if perf["rolling_smape"] is not None else {}),
                    **({"error_ratio": ratio} if ratio is not None else {}),
                }
            )
            path = "/tmp/powerfc_drift_report.html"  # noqa: S108 - scratch file, uploaded then discarded
            snapshot.save_html(path)
            mlflow.log_artifact(path, artifact_path="evidently")

    out = {
        **{k: v for k, v in result.items() if k != "details"},
        "psi": result["details"]["psi"],
        "error_ratio": ratio,
        "champion_version": champ.version,
        "champion_backtest_smape_24h": champion_smape,
        "scored_hours": perf["scored_hours"],
        "unseen_hours": unseen_hours,
        "drift_detected": drift_detected,
        "reasons": reasons,
    }
    return json.loads(json.dumps(out, default=str))  # plain JSON for Airflow XCom
