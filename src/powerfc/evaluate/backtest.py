"""Rolling-origin backtest (never random K-fold).

Origins are every `step_days` days. At each origin o the forecast covers
hours o+1 … o+72; the block serving lead time L is the smallest block >= L.
Models are refit every `refit_every` origins on all targets at or before
the refit origin (an expanding window), so a model is never trained on any
hour it is later asked to forecast.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powerfc import config
from powerfc.evaluate import metrics
from powerfc.features.build import block_features, seasonal_naive, targets
from powerfc.models.lgbm import fit_block


@dataclass
class BacktestConfig:
    start: str  # first origin (inclusive), local time
    end: str  # last origin (inclusive)
    step_days: int = 7
    refit_every: int = 4
    origin_hour: int = 23  # forecast issued after the 23:00 reading
    params: dict | None = None


def block_for_lead(lead: int) -> int:
    for b in config.BLOCKS:
        if lead <= b:
            return b
    raise ValueError(f"lead {lead} beyond the longest block")


def origins(cfg: BacktestConfig) -> pd.DatetimeIndex:
    days = pd.date_range(cfg.start, cfg.end, freq=f"{cfg.step_days}D")
    return days.normalize() + pd.Timedelta(hours=cfg.origin_hour)


def run(clean: pd.DataFrame, cfg: BacktestConfig, *, feats: dict | None = None) -> pd.DataFrame:
    """Return one row per (origin, target hour) with actuals, naive and model quantiles."""
    feats = feats or {b: block_features(clean, b) for b in config.BLOCKS}
    tgt = targets(clean)
    naive = seasonal_naive(clean)
    rows = []
    models = None
    for i, o in enumerate(origins(cfg)):
        if i % cfg.refit_every == 0:
            train_until = o  # targets observed at or before the origin
            models = {
                b: fit_block(feats[b].loc[:train_until], tgt.loc[:train_until], b, cfg.params) for b in config.BLOCKS
            }
        hours = pd.date_range(o + pd.Timedelta(hours=1), periods=max(config.BLOCKS), freq="h")
        hours = hours[hours.isin(tgt.index)]
        if len(hours) == 0:
            continue
        leads = ((hours - o) / pd.Timedelta(hours=1)).astype(int)
        for b in config.BLOCKS:
            sel = hours[(leads <= b) & (leads > b - 24)]
            if len(sel) == 0:
                continue
            pred = models[b].predict(feats[b].loc[sel])
            frame = pd.DataFrame(
                {
                    "origin": o,
                    "target_time": sel,
                    "lead": ((sel - o) / pd.Timedelta(hours=1)).astype(int),
                    "block": b,
                    "actual": tgt.loc[sel].to_numpy(),
                    "naive": naive.loc[sel].to_numpy(),
                    "is_holiday": (feats[b].loc[sel, "is_holiday"] == 1).to_numpy()
                    | (feats[b].loc[sel, "days_to_eid_fitr"].abs() <= 3).to_numpy()
                    | (feats[b].loc[sel, "days_to_eid_adha"].abs() <= 3).to_numpy(),
                }
            )
            rows.append(pd.concat([frame.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
    return pd.concat(rows, ignore_index=True)


def summarise(bt: pd.DataFrame) -> dict[str, float]:
    """Per-block metrics (scored only where actual, naive and model all exist)."""
    out: dict[str, float] = {}
    scored = bt.dropna(subset=["actual", "naive", "p50"])
    for b, g in scored.groupby("block"):
        model_smape = metrics.smape(g.actual, g.p50)
        naive_smape = metrics.smape(g.actual, g.naive)
        out.update(
            {
                f"smape_{b}h": model_smape,
                f"naive_smape_{b}h": naive_smape,
                f"skill_smape_{b}h": metrics.skill(model_smape, naive_smape),
                f"mape_{b}h": metrics.mape(g.actual, g.p50),
                f"mae_{b}h": metrics.mae(g.actual, g.p50),
                f"naive_mae_{b}h": metrics.mae(g.actual, g.naive),
                f"pinball_p10_{b}h": metrics.pinball(g.actual, g.p10, 0.1),
                f"pinball_p50_{b}h": metrics.pinball(g.actual, g.p50, 0.5),
                f"pinball_p90_{b}h": metrics.pinball(g.actual, g.p90, 0.9),
                f"coverage80_{b}h": metrics.coverage(g.actual, g.p10, g.p90),
                f"width80_{b}h": metrics.mean_width(g.p10, g.p90),
                f"crossing_rate_{b}h": float(g.crossed.mean()),
                f"smape_holiday_{b}h": metrics.smape(g[g.is_holiday].actual, g[g.is_holiday].p50),
                f"smape_normal_{b}h": metrics.smape(g[~g.is_holiday].actual, g[~g.is_holiday].p50),
                f"n_hours_{b}h": float(len(g)),
            }
        )
    out["smape_mean"] = float(np.mean([out[f"smape_{b}h"] for b in config.BLOCKS]))
    out["origins"] = float(bt["origin"].nunique())
    return out
