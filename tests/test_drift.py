"""The drift rule that decides, inside the daily DAG, whether to retrain."""

import numpy as np
import pandas as pd

from powerfc.monitoring.drift import CURRENT_DAYS, ERROR_RATIO_ALERT, PSI_ALERT, decide_retrain, psi, windows


def test_no_signal_no_retrain():
    assert decide_retrain(0.05, 1.1) == (False, [])
    assert decide_retrain(None, None) == (False, [])  # nothing measurable yet


def test_data_drift_alone_triggers():
    detected, reasons = decide_retrain(PSI_ALERT + 0.01, 1.0)
    assert detected and reasons[0].startswith("data drift")


def test_performance_drift_alone_triggers():
    detected, reasons = decide_retrain(0.1, ERROR_RATIO_ALERT + 0.1)
    assert detected and reasons[0].startswith("performance drift")


def test_thresholds_are_strict_inequalities():
    assert decide_retrain(PSI_ALERT, ERROR_RATIO_ALERT) == (False, [])


def test_both_signals_give_two_reasons():
    assert len(decide_retrain(1.0, 2.0)[1]) == 2


def test_psi_is_small_for_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1.0, 1, 5000)) > PSI_ALERT


def test_current_window_only_holds_data_the_champion_has_not_seen():
    hours = pd.DataFrame({"datetime": pd.date_range("2025-01-01", "2025-03-20 23:00", freq="h")})
    cut = pd.Timestamp("2025-03-15 23:00")  # champion trained up to here
    (r0, r1), (c0, c1) = windows(hours, cut)
    assert r1 == cut and c0 == cut + pd.Timedelta(hours=1) and c1 == hours.datetime.max()
    # Long after the cut-off, the current window is capped at the latest CURRENT_DAYS.
    (_, _), (c0, _) = windows(hours, pd.Timestamp("2025-02-01 23:00"))
    assert c0 == hours.datetime.max() - pd.Timedelta(days=CURRENT_DAYS) + pd.Timedelta(hours=1)
