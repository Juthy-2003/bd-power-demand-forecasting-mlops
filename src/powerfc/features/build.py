"""Origin-safe features for direct multi-horizon forecasting.

For horizon block `b` (24, 48 or 72 hours), the model predicting demand at
target hour T may only use observations at or before T − b. Every feature
here is a shift of at least `b` hours, so when the forecast is made at an
origin o with lead time T − o <= b, nothing after o is used. A unit test
mutates future data and checks the features do not change.

Target transform: national demand grows about 5% a year, and tree models
cannot extrapolate a level they never saw. So the model predicts the ratio
of demand to the mean of the 168 hours ending at T − b ("recent level"),
and lag features are expressed as ratios to that same level. Predictions
are multiplied back by the level.
"""

import numpy as np
import pandas as pd

from powerfc.features.calendar import calendar_features

LEVEL_HOURS = 168


def lags_for(block: int) -> list[int]:
    """Lag hours allowed for a block: never shorter than the block."""
    candidates = [block, block + 24, block + 48, 168, 168 + 24, 336, 504]
    return sorted({k for k in candidates if k >= block})


def block_features(clean: pd.DataFrame, block: int) -> pd.DataFrame:
    """One row per hour T: features for predicting demand at T with block `block`.

    `clean` must be the hourly grid from `data.clean.clean`.
    """
    df = clean.set_index("datetime")
    y = df["demand_filled"]  # lags may use short-gap interpolation
    base = y.shift(block)
    level = base.rolling(LEVEL_HOURS, min_periods=LEVEL_HOURS // 2).mean()

    f = calendar_features(df.index)
    f["level"] = level
    for k in lags_for(block):
        f[f"lag_{k}"] = y.shift(k) / level
    f["mean_24"] = base.rolling(24, min_periods=12).mean() / level
    f["max_24"] = base.rolling(24, min_periods=12).max() / level
    f["min_24"] = base.rolling(24, min_periods=12).min() / level
    # Same hour of day over the last 7 days that are at least `block` back.
    first_day = int(np.ceil(block / 24))
    same_hour = pd.concat([y.shift(24 * d) for d in range(first_day, first_day + 7)], axis=1)
    f["same_hour_mean_7d"] = same_hour.mean(axis=1) / level
    # Supply context known at T − b: was there shedding, how much?
    shed = df["load_shedding"].fillna(0)
    f["shed_share_24"] = shed.shift(block).rolling(24, min_periods=12).apply(lambda w: (w > 0).mean(), raw=True)
    f["shed_mean_24"] = shed.shift(block).rolling(24, min_periods=12).mean() / level
    f["block"] = block
    return f


FEATURE_EXCLUDE = {"level", "block"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in FEATURE_EXCLUDE]


def targets(clean: pd.DataFrame) -> pd.Series:
    """The target: observed demand in MW, indexed by hour (never imputed)."""
    return clean.set_index("datetime")["demand_mw"]


def seasonal_naive(clean: pd.DataFrame, lag: int = 168) -> pd.Series:
    """Same hour one week earlier — the baseline every claim is measured against."""
    return clean.set_index("datetime")["demand_filled"].shift(lag)
