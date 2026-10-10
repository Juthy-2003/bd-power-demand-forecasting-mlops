"""Forecast API (FastAPI).

Serves the latest batch forecast written by the daily pipeline and the
champion's metadata from the MLflow registry.

The API does not run the model per request: forecasts are produced once per
origin by the pipeline and read from Postgres, so request latency does not
depend on model size. After a promotion the pipeline calls `/admin/reload`
so `/model` shows the new champion.
"""

import logging
import math
from contextlib import asynccontextmanager
from enum import IntEnum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from powerfc import __version__, config
from powerfc.serve import store

log = logging.getLogger("powerfc.api")

_champion: dict = {}
SHOWN_METRICS = ("smape_", "skill_", "coverage80_", "width80_", "mae_")


def load_champion() -> dict:
    """Champion metadata from the registry. Returns {} if MLflow is unreachable."""
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        mv = client.get_model_version_by_alias(config.REGISTERED_MODEL, "champion")
        run = client.get_run(mv.run_id)
        metrics = {
            k: round(v, 4)
            for k, v in run.data.metrics.items()
            if k.startswith(SHOWN_METRICS) and math.isfinite(v)  # NaN is not valid JSON
        }
        return {
            "name": config.REGISTERED_MODEL,
            "version": mv.version,
            "run_id": mv.run_id,
            "data_version": run.data.tags.get("data_version"),
            "recipe_hash": run.data.tags.get("recipe_hash"),
            "backtest_window": run.data.tags.get("window"),
            "gate_decision": mv.tags.get("gate_decision"),
            "gate_reasons": mv.tags.get("gate_reasons"),
            "backtest_metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001 - readiness reports it
        log.warning("champion not loaded: %s", exc)
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _champion
    _champion = load_champion()
    yield


app = FastAPI(
    title="Bangladesh Power Demand Forecast API",
    version=__version__,
    description=(
        "Hourly national electricity demand forecasts for Bangladesh, 1–72 hours ahead, with an 80% "
        "prediction interval (P10–P90). Trained on PGCB data (Mendeley doi:10.17632/vpk8spw2mm.1, "
        "CC BY 4.0). Times are Asia/Dhaka local time."
    ),
    lifespan=lifespan,
)


class ForecastPoint(BaseModel):
    target_time: str
    lead: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    origin: str
    horizon: int
    model_version: str
    data_version: str
    unit: str = "MW"
    timezone: str = config.TZ
    points: list[ForecastPoint]


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/ready")
def ready():
    global _champion
    if not _champion:  # MLflow may have come up after the API; try again
        _champion = load_champion()
    if not _champion:
        raise HTTPException(503, "champion model metadata not loaded")
    return {"ready": True, "champion": _champion["version"]}


class Horizon(IntEnum):
    h24 = 24
    h48 = 48
    h72 = 72


@app.get("/forecast", response_model=ForecastResponse)
def forecast(horizon: Horizon = Query(Horizon.h72, description="Hours ahead: 24, 48 or 72")):
    horizon = int(horizon)
    df = store.latest_forecast(horizon)
    if df.empty:
        raise HTTPException(404, "no forecast has been written yet")
    return ForecastResponse(
        origin=str(df.origin.iloc[0]),
        horizon=horizon,
        model_version=str(df.model_version.iloc[0]),
        data_version=str(df.data_version.iloc[0]),
        points=[
            ForecastPoint(
                target_time=str(r.target_time),
                lead=int(r.lead),
                p10=round(r.p10, 1),
                p50=round(r.p50, 1),
                p90=round(r.p90, 1),
            )
            for r in df.itertuples()
        ],
    )


@app.get("/model")
def model():
    if not _champion:
        raise HTTPException(503, "champion model metadata not loaded")
    return _champion


@app.post("/admin/reload")
def reload_champion():
    """Re-read the champion from MLflow. Called by the pipeline after a promotion
    (Airflow task, or the Kubernetes training CronJob via `powerfc reload-api`)."""
    global _champion
    _champion = load_champion()
    return {"reloaded": bool(_champion), "version": _champion.get("version")}
