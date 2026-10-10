import numpy as np
import pandas as pd
import pytest

from powerfc.data.clean import clean


def synthetic_raw(days: int = 120, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """A small, realistic-looking raw table for fast tests (not PGCB data)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 24, freq="h")
    hour = t.hour.to_numpy()
    daily = 1 + 0.25 * np.sin((hour - 6) / 24 * 2 * np.pi) + 0.15 * (hour >= 18) * (hour <= 22)
    weekly = np.where(t.dayofweek == 4, 0.9, 1.0)
    trend = 1 + np.arange(len(t)) / len(t) * 0.05
    demand = 9000 * daily * weekly * trend * (1 + rng.normal(0, 0.02, len(t)))
    shed = np.where(rng.random(len(t)) < 0.1, rng.uniform(50, 800, len(t)), 0.0)
    df = pd.DataFrame(
        {
            "datetime": t,
            "generation_mw": demand - shed,
            "demand_mw": demand.round(),
            "load_shedding": shed.round(),
            "gas": demand * 0.5,
            "liquid_fuel": demand * 0.2,
            "coal": demand * 0.2,
            "hydro": 80.0,
            "solar": 0.0,
            "wind": 0.0,
            "india_bheramara_hvdc": 900.0,
            "india_tripura": 150.0,
            "india_adani": 1400.0,
            "nepal": 0.0,
            "remarks": pd.array([None] * len(t), dtype="string"),
        }
    )
    return df


@pytest.fixture
def raw():
    return synthetic_raw()


@pytest.fixture
def cleaned(raw):
    return clean(raw)
