"""The deployable model: one MLflow pyfunc holding all three horizon blocks.

`predict(history)` takes the cleaned hourly table up to the forecast origin
(the last row) and returns the next 72 hours of P10/P50/P90 demand, each
from the block that owns that lead time.
The package code travels with the model (`code_paths`), so the registry
entry is self-contained.
"""

import mlflow.pyfunc
import numpy as np
import pandas as pd

from powerfc import config
from powerfc.evaluate.backtest import block_for_lead
from powerfc.features.build import block_features
from powerfc.models.lgbm import BlockModel


class PowerForecaster(mlflow.pyfunc.PythonModel):
    def __init__(self, blocks: dict[int, BlockModel], recipe: dict):
        self.blocks = blocks
        self.recipe = recipe

    def forecast(self, history: pd.DataFrame) -> pd.DataFrame:
        history = history.sort_values("datetime").reset_index(drop=True)
        origin = history["datetime"].iloc[-1]
        horizon = max(config.BLOCKS)
        future_times = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        future = pd.DataFrame({"datetime": future_times})
        for col in history.columns:
            if col != "datetime":
                future[col] = np.nan if history[col].dtype.kind == "f" else history[col].iloc[-1]
        extended = pd.concat([history, future], ignore_index=True)
        out = []
        for b in config.BLOCKS:
            leads = np.arange(1, horizon + 1)
            mine = future_times[[block_for_lead(int(L)) == b for L in leads]]
            feats = block_features(extended, b).loc[mine]
            pred = self.blocks[b].predict(feats)
            pred.insert(0, "block", b)
            out.append(pred)
        result = pd.concat(out).sort_index()
        result.index.name = "target_time"
        result = result.reset_index()
        result.insert(0, "origin", origin)
        result.insert(2, "lead", ((result["target_time"] - origin) / pd.Timedelta(hours=1)).astype(int))
        return result

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.forecast(model_input)
