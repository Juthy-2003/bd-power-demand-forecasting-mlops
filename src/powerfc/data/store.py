"""The pipeline's view of the data: a base snapshot plus daily landings.

The last 12 months of the PGCB file are the holdout. The pipeline starts
"as of" the day before the holdout and receives one new day at a time:
`land_day(day)` copies that day's rows into `landing/date=YYYY-MM-DD/`.
This is the documented fallback for a live PGCB ERP scraper, which is not
built because permission to scrape the portal has not been checked (v01 §18
Q14). The interface is the same: one logical date in, one partition out.

Re-landing a day overwrites its partition, so DAG re-runs and backfills
are idempotent.
"""

import hashlib
import os
from pathlib import Path

import pandas as pd

from powerfc import config
from powerfc.data.clean import clean
from powerfc.data.contract import validate_clean, validate_raw
from powerfc.data.load import read_raw

# Last hour of the initial snapshot: the day before the 365-day holdout.
AS_OF_START = pd.Timestamp(os.environ.get("POWERFC_AS_OF_START", "2025-03-08 23:00"))


def _partition(day: pd.Timestamp) -> Path:
    return config.LANDING_DIR / f"date={day:%Y-%m-%d}" / "part.parquet"


def land_day(day: str | pd.Timestamp, source: pd.DataFrame | None = None) -> dict:
    """Validate and store one day of new readings. Returns a small report."""
    day = pd.Timestamp(day).normalize()
    full = source if source is not None else read_raw()
    rows = full[(full["datetime"] >= day) & (full["datetime"] < day + pd.Timedelta(days=1))]
    rows = validate_raw(rows.reset_index(drop=True))
    path = _partition(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)  # overwrite: idempotent per logical date
    return {"date": f"{day:%Y-%m-%d}", "rows": int(len(rows)), "path": str(path)}


def landed_days() -> list[str]:
    if not config.LANDING_DIR.exists():
        return []
    return sorted(p.name.removeprefix("date=") for p in config.LANDING_DIR.glob("date=*"))


def assemble(as_of_start: pd.Timestamp = AS_OF_START) -> tuple[pd.DataFrame, str]:
    """Base snapshot + every landed day → cleaned, validated hourly table."""
    base = read_raw()
    base = base[base["datetime"] <= as_of_start]
    parts = (
        [pd.read_parquet(p) for p in sorted(config.LANDING_DIR.glob("date=*/part.parquet"))]
        if config.LANDING_DIR.exists()
        else []
    )
    raw = pd.concat([base, *parts], ignore_index=True) if parts else base
    raw = raw.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    cleaned = validate_clean(clean(validate_raw(raw)))
    config.CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(config.CLEAN_PARQUET, index=False)
    return cleaned, data_version(cleaned)


def load_clean() -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    return df, data_version(df)


def data_version(df: pd.DataFrame) -> str:
    """Content hash of the cleaned table: same data → same version, whatever
    the file's bytes. Every MLflow run records it."""
    cols = ["datetime", "demand_mw", "load_shedding", "generation_mw"]
    digest = hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes())
    return digest.hexdigest()[:12]
