"""Forecast metrics. All functions ignore positions where any input is NaN."""

import numpy as np


def _mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= ~np.isnan(np.asarray(a, dtype=float))
    return m


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p)
    return float(np.mean(np.abs(y[m] - p[m])))


def mape(y, p) -> float:
    """Undefined when an actual is zero, so sMAPE is the headline metric."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p) & (y != 0)
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def smape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p)
    denom = (np.abs(y[m]) + np.abs(p[m])) / 2
    return float(np.mean(np.abs(y[m] - p[m]) / denom) * 100)


def pinball(y, q_pred, alpha: float) -> float:
    y, q = np.asarray(y, float), np.asarray(q_pred, float)
    m = _mask(y, q)
    diff = y[m] - q[m]
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = _mask(y, lo, hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


def mean_width(lo, hi) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = _mask(lo, hi)
    return float(np.mean(hi[m] - lo[m]))


def skill(model_error: float, baseline_error: float) -> float:
    """1 − model/baseline: 0.2 means 20% lower error than the baseline."""
    return float(1 - model_error / baseline_error) if baseline_error else float("nan")
