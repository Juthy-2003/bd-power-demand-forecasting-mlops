"""Train a recipe, backtest it, log everything to MLflow, register it.

A *recipe* is everything that defines a model except the data: LightGBM
parameters and the feature-set name. Its hash (`recipe_hash`)
lets the promotion gate tell "same recipe on newer data" (a refresh) from
"a different recipe" (which must beat the champion).
"""

import hashlib
import json
import os
import subprocess  # git commit for run lineage
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from powerfc import config  # noqa: E402
from powerfc.data.clean import summary  # noqa: E402
from powerfc.evaluate.backtest import BacktestConfig, run, summarise  # noqa: E402
from powerfc.features.build import block_features, targets  # noqa: E402
from powerfc.models.lgbm import DEFAULT_PARAMS, fit_block  # noqa: E402
from powerfc.models.pyfunc import PowerForecaster  # noqa: E402

FEATURE_SET = "v1-calendar-holiday-lags-ratio"
PACKAGE_DIR = Path(__file__).resolve().parent


def recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def make_recipe(params: dict | None = None) -> dict:
    return {"params": {**DEFAULT_PARAMS, **(params or {})}, "feature_set": FEATURE_SET}


def gate_window(clean: pd.DataFrame, weeks: int, refit_every: int) -> BacktestConfig:
    """The evaluation window is a pure function of the data end, so two
    recipes evaluated on the same data version see exactly the same folds."""
    end = clean["datetime"].max().normalize() - pd.Timedelta(days=4)
    start = end - pd.Timedelta(weeks=weeks - 1)
    return BacktestConfig(start=f"{start:%Y-%m-%d}", end=f"{end:%Y-%m-%d}", step_days=7, refit_every=refit_every)


def _git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()  # noqa: S603
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def _plot(bt: pd.DataFrame, path: Path) -> None:
    last = bt[bt["origin"] == bt["origin"].max()].sort_values("target_time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(last.target_time, last.p10, last.p90, alpha=0.25, label="P10–P90")
    ax.plot(last.target_time, last.p50, label="P50 forecast")
    ax.plot(last.target_time, last.actual, label="actual", color="black", linewidth=1)
    ax.plot(last.target_time, last.naive, label="seasonal naive", linestyle=":", color="grey")
    ax.set_title(f"Last backtest origin {last.origin.iloc[0]:%Y-%m-%d %H:%M} (Asia/Dhaka)")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_recipe(clean: pd.DataFrame, recipe: dict, window: BacktestConfig, feats=None) -> tuple[pd.DataFrame, dict]:
    window.params = recipe["params"]
    t = time.perf_counter()
    bt = run(clean, window, feats=feats)
    metrics = summarise(bt)
    metrics["backtest_seconds"] = round(time.perf_counter() - t, 1)
    return bt, metrics


def fit_final(clean: pd.DataFrame, recipe: dict, feats=None) -> PowerForecaster:
    feats = feats or {b: block_features(clean, b) for b in config.BLOCKS}
    tgt = targets(clean)
    blocks = {b: fit_block(feats[b], tgt, b, recipe["params"]) for b in config.BLOCKS}
    return PowerForecaster(blocks, recipe)


def train_and_register(
    clean: pd.DataFrame,
    data_version: str,
    recipe: dict | None = None,
    *,
    weeks: int = 26,
    refit_every: int = 8,
    register: bool = True,
    run_name: str | None = None,
) -> dict:
    recipe = recipe or make_recipe()
    rhash = recipe_hash(recipe)
    window = gate_window(clean, weeks, refit_every)
    feats = {b: block_features(clean, b) for b in config.BLOCKS}
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name or f"train-{data_version}-{rhash}") as active:
        mlflow.set_tags(
            {
                "data_version": data_version,
                "recipe_hash": rhash,
                "feature_set": recipe["feature_set"],
                "git_commit": _git_commit(),
                "window": f"{window.start}..{window.end}/{window.step_days}d/refit{window.refit_every}",
                "kind": "train",
            }
        )
        mlflow.log_params({**recipe["params"], "weeks": weeks})
        mlflow.log_param("recipe_json", json.dumps(recipe, sort_keys=True))
        mlflow.log_params({f"data_{k}": v for k, v in summary(clean).items()})

        bt, metrics = evaluate_recipe(clean, recipe, window, feats)
        mlflow.log_metrics(metrics)

        model = fit_final(clean, recipe, feats)
        t = time.perf_counter()
        sample = model.forecast(clean.tail(24 * 30))
        metrics_latency = (time.perf_counter() - t) * 1000
        mlflow.log_metric("predict_latency_ms", round(metrics_latency, 1))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bt.to_parquet(tmp / "backtest.parquet", index=False)
            _plot(bt, tmp / "last_origin.png")
            imp = pd.DataFrame(
                {
                    f"block_{b}h": pd.Series(
                        model.blocks[b].quantile_models[0.5].feature_importances_, index=model.blocks[b].columns
                    )
                    for b in config.BLOCKS
                }
            ).sort_values(f"block_{config.BLOCKS[0]}h", ascending=False)
            imp.to_csv(tmp / "feature_importance.csv")
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            code_paths=[str(PACKAGE_DIR)],
            input_example=clean.tail(24 * 30),
            pip_requirements=["lightgbm>=4.5", "pandas>=2.2", "numpy", "holidays", "pyarrow", "mlflow>=3"],
        )
        result = {
            "run_id": active.info.run_id,
            "model_uri": info.model_uri,
            "data_version": data_version,
            "recipe_hash": rhash,
            "metrics": metrics,
            "sample_rows": len(sample),
        }
        if register:
            mv = mlflow.register_model(info.model_uri, config.REGISTERED_MODEL)
            client = MlflowClient()
            client.set_model_version_tag(config.REGISTERED_MODEL, mv.version, "data_version", data_version)
            client.set_model_version_tag(config.REGISTERED_MODEL, mv.version, "recipe_hash", rhash)
            client.set_registered_model_alias(config.REGISTERED_MODEL, "challenger", mv.version)
            result["version"] = mv.version
    return result
