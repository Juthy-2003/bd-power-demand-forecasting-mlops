"""Turn raw readings into a clean, complete hourly series.

Rules (the data contract in `contract.py` checks the result):
1. Reindex to a complete hourly grid. Missing hours become NaN, never
   silently dropped.
2. Flag, don't delete, impossible values: the raw file contains typos such
   as 121,000 MW of demand when the national median is about 8,600 MW.
   Flagged values are set to NaN in the cleaned columns and recorded in
   `quality_flag`.
3. Short gaps (<= MAX_FILL_HOURS) in `demand_mw` are interpolated into a
   separate `demand_filled` column used only for building lag features,
   with `imputed = True`. The target `demand_mw` is never imputed.
"""

import numpy as np
import pandas as pd

from powerfc.data.load import NUMERIC

# Bangladesh's recorded national demand has stayed well inside this band
# since 2015 (yearly means 5.9k–12k MW in this dataset). Anything outside is
# a recording error, not a grid event.
DEMAND_RANGE = (1_000.0, 25_000.0)
# A reading this far from the surrounding week's median is treated as a
# typo (e.g. an extra digit). Real events (Eid, blackouts) move demand by
# tens of percent, not by multiples. Known limit: the median is over all
# hours, so a night reading that doubles to a normal peak-hour value is not
# caught; a same-hour-of-day median would be stricter.
MAX_RATIO_TO_MEDIAN = 1.8
MAX_FILL_HOURS = 6
COMPONENT_MAX = 30_000.0  # no single fuel or import line exceeds this


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.set_index("datetime")
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", name="datetime")
    df = df.reindex(grid)
    df["missing"] = df["demand_mw"].isna()

    flags = pd.Series("", index=df.index, dtype="object")

    def flag(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False)
        flags[mask] = np.where(flags[mask] == "", reason, flags[mask] + "," + reason)

    demand = df["demand_mw"]
    out_of_range = (demand < DEMAND_RANGE[0]) | (demand > DEMAND_RANGE[1])
    flag(out_of_range, "demand_range")
    weekly_median = demand.where(~out_of_range).rolling(168, center=True, min_periods=24).median()
    ratio = demand / weekly_median
    spike = (ratio > MAX_RATIO_TO_MEDIAN) | (ratio < 1 / MAX_RATIO_TO_MEDIAN)
    flag(spike & ~out_of_range, "demand_spike")
    df.loc[out_of_range.fillna(False) | spike.fillna(False), "demand_mw"] = np.nan

    for col in NUMERIC:
        if col == "demand_mw":
            continue
        bad = (df[col] < 0) | (df[col] > COMPONENT_MAX * (3 if col == "generation_mw" else 1))
        flag(bad, f"{col}_range")
        df.loc[bad.fillna(False), col] = np.nan

    df["quality_flag"] = flags.replace("", pd.NA).astype("string")
    df["demand_filled"] = fill_short_gaps(df["demand_mw"], MAX_FILL_HOURS)
    df["imputed"] = df["demand_mw"].isna() & df["demand_filled"].notna()
    df["shedding"] = df["load_shedding"].fillna(0) > 0
    return df.reset_index()


def fill_short_gaps(s: pd.Series, max_len: int) -> pd.Series:
    """Interpolate only gaps of at most `max_len` consecutive NaNs.

    (`Series.interpolate(limit=n)` would also fill the first n hours of a
    long gap, inventing data inside an outage.)"""
    is_nan = s.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_len = is_nan.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~is_nan | (run_len <= max_len))


def summary(df: pd.DataFrame) -> dict:
    """Numbers for the data report and the MLflow run."""
    return {
        "hours": int(len(df)),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "missing_hours": int(df["missing"].sum()),
        "flagged_hours": int(df["quality_flag"].notna().sum()),
        "imputed_hours": int(df["imputed"].sum()),
        "target_nan_hours": int(df["demand_mw"].isna().sum()),
        "shedding_share": round(float(df["shedding"].mean()), 4),
    }
