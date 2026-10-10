"""Paths, horizons and constants shared by every stage.

All paths can be overridden with environment variables so the same code
runs on a laptop, inside the Airflow container and in CI.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("POWERFC_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("POWERFC_DATA", ROOT / "data"))
RAW_CSV = DATA_DIR / "raw" / "pgcb_V1_raw.csv"
CLEAN_PARQUET = DATA_DIR / "clean" / "hourly.parquet"
LANDING_DIR = DATA_DIR / "landing"  # daily partitions written by the ingest DAG

TZ = "Asia/Dhaka"  # the dataset's timestamps are local time (UTC+6, no DST)
TARGET = "demand_mw"

# Direct multi-horizon: one model per horizon block. A block's model serves
# lead times (previous block, block] and may only use lags >= block hours.
BLOCKS = (24, 48, 72)
QUANTILES = (0.1, 0.5, 0.9)

# Holdout: the last 12 months are touched once, for the final table, and are
# replayed day by day as "new data" by the ingest DAG.
HOLDOUT_DAYS = 365

MLFLOW_EXPERIMENT = os.environ.get("POWERFC_EXPERIMENT", "powerfc-demand")
REGISTERED_MODEL = os.environ.get("POWERFC_MODEL_NAME", "powerfc-demand")
