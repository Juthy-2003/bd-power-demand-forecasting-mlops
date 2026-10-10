import numpy as np
import pandas as pd
import pytest

from powerfc.evaluate import metrics
from powerfc.features.calendar import calendar_features


def test_metrics_basics():
    y, p = np.array([100.0, 200.0, np.nan]), np.array([110.0, 180.0, 5.0])
    assert metrics.mae(y, p) == 15
    assert round(metrics.mape(y, p), 6) == 10.0
    assert metrics.coverage([1, 2, 3], [0, 2.5, 2], [2, 3, 4]) == 2 / 3
    assert metrics.skill(8, 10) == pytest.approx(0.2)
    assert metrics.pinball([10], [8], 0.9) == 0.9 * 2
    assert metrics.mean_width([1, 2], [3, 5]) == 2.5


def test_eid_and_ramadan_features_2025():
    """Bangladesh observed Eid-ul-Fitr on 2025-03-31 (holidays 31 Mar–2 Apr)."""
    f = calendar_features(
        pd.DatetimeIndex(["2025-03-20 12:00", "2025-03-31 12:00", "2025-04-02 12:00", "2025-05-15 12:00"])
    )
    assert f["ramadan"].tolist() == [1, 0, 0, 0]
    # 11 days before is outside the ±10-day window, so it reads "far" (11).
    assert f["days_to_eid_fitr"].tolist() == [11, 0, 2, 11]
    assert f["is_holiday"].iloc[1] == 1 and f["is_holiday"].iloc[3] == 0
