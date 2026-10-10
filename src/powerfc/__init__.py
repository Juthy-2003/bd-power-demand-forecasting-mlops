"""Bangladesh Power Demand Forecasting MLOps Pipeline."""

__version__ = "0.1.0"

import os as _os

_os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep CLI output clean
