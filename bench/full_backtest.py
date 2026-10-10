"""The accuracy numbers in the README.

1. Backtest: weekly origins 2023-03-12 → 2025-03-02 (two years before the
   holdout), refit every 4 weeks, expanding window.
2. Holdout: weekly origins over the final 12 months (2025-03-09 → 2026-03-01),
   evaluated ONCE with the same rolling procedure. Nothing was tuned on it.

Both use the default recipe and the full cleaned PGCB file.

    uv run python bench/full_backtest.py      # ~10–15 minutes on a laptop
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from powerfc import config
from powerfc.data.clean import clean
from powerfc.data.load import fetch_raw, read_raw
from powerfc.evaluate.backtest import BacktestConfig, run, summarise
from powerfc.features.build import block_features
from powerfc.models.lgbm import DEFAULT_PARAMS

OUT = Path("bench/results")
REUSE = "--reuse" in sys.argv
WINDOWS = {
    "backtest_2023_2025": ("2023-03-12", "2025-03-02"),
    "holdout_2025_2026": ("2025-03-09", "2026-03-01"),
}
KEYS = [
    "smape",
    "naive_smape",
    "skill_smape",
    "mae",
    "naive_mae",
    "coverage80",
    "width80",
    "smape_holiday",
    "smape_normal",
    "crossing_rate",
    "n_hours",
]


def main() -> None:
    fetch_raw()
    data = clean(read_raw())
    feats = {b: block_features(data, b) for b in config.BLOCKS}
    report = {"recipe": DEFAULT_PARAMS, "windows": {}}
    for name, (start, end) in WINDOWS.items():
        t = time.time()
        cached = OUT / f"{name}.parquet"
        if REUSE and cached.exists():  # summarise a saved run instead of retraining
            bt = pd.read_parquet(cached)
        else:
            cfg = BacktestConfig(start=start, end=end, step_days=7, refit_every=4, params=DEFAULT_PARAMS)
            bt = run(data, cfg, feats=feats)
        s = summarise(bt)
        report["windows"][name] = {
            "origins": s["origins"],
            "seconds": round(time.time() - t),
            "by_block": {b: {k: round(s[f"{k}_{b}h"], 4) for k in KEYS} for b in config.BLOCKS},
        }
        bt.to_parquet(OUT / f"{name}.parquet", index=False)
        print(name, json.dumps(report["windows"][name]["by_block"][24]))
    (OUT / "forecast_accuracy.json").write_text(json.dumps(report, indent=2))

    lines = []
    for name, w in report["windows"].items():
        lines += [
            f"**{name.replace('_', ' ')}** ({int(w['origins'])} weekly origins)",
            "",
            "| Lead time | sMAPE | Seasonal-naive sMAPE | Skill | MAE (MW) | 80% interval coverage (raw) | Interval width (MW) | Holiday sMAPE |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for b, m in w["by_block"].items():
            lines.append(
                # Block b serves lead times b-23 … b hours (backtest.block_for_lead).
                f"| {b - 23}–{b} h | {m['smape']:.2f}% | {m['naive_smape']:.2f}% | {m['skill_smape']:.0%} | {m['mae']:.0f} | "
                f"{m['coverage80']:.1%} | {m['width80']:.0f} | {m['smape_holiday']:.2f}% |"
            )
        lines.append("")
    (OUT / "forecast_accuracy.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
