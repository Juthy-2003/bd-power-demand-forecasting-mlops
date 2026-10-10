"""LightGBM models: three quantile regressors (P10, P50, P90) per horizon block.

Quantile regression with pinball loss means the alpha = 0.9 model learns the
90th percentile, so P10–P90 is meant to be an 80% prediction interval. The
intervals are NOT calibrated afterwards: the backtest reports how often the
actual really fell inside (coverage) and how wide the interval was, and the
promotion gate checks coverage against a band set from those backtest
numbers (registry/promote.py, ADR 0009).

Independent quantile models can cross (P10 > P50); predictions are sorted
per hour and the crossing rate is reported.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerfc import config
from powerfc.features.build import feature_columns

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": 4,
}


@dataclass
class BlockModel:
    block: int
    columns: list[str]
    quantile_models: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        x = feats[self.columns]
        level = feats["level"].to_numpy()
        r = np.column_stack([self.quantile_models[a].predict(x) for a in config.QUANTILES])
        crossed = ~np.all(np.diff(r, axis=1) >= 0, axis=1)
        r.sort(axis=1)
        q = r * level[:, None]  # ratio to recent level → MW
        out = pd.DataFrame(q, index=feats.index, columns=["p10", "p50", "p90"])
        out["crossed"] = crossed
        out.loc[feats["level"].isna(), ["p10", "p50", "p90"]] = np.nan
        return out


def fit_block(feats: pd.DataFrame, target: pd.Series, block: int, params: dict | None = None) -> BlockModel:
    """Fit one block on rows where the target and the level are known."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    cols = feature_columns(feats)
    rows = feats["level"].notna() & target.notna()
    x, level = feats.loc[rows, cols], feats.loc[rows, "level"]
    y_ratio = target[rows] / level

    model = BlockModel(block=block, columns=cols)
    for alpha in config.QUANTILES:
        reg = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)
        model.quantile_models[alpha] = reg.fit(x, y_ratio)
    return model
