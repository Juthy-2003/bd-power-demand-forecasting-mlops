"""Calendar and holiday features — known in advance for any future hour.

Eid-ul-Fitr and Eid-ul-Adha follow the lunar Hijri calendar and move about
11 days earlier every Gregorian year, so month-of-year features cannot learn
them. Dates come from the `holidays` package (Bangladesh), which carries the
observed Islamic holiday dates for past years. Ramadan is approximated as the
30 days before Eid-ul-Fitr (the true start depends on moon sighting, ±1 day).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

WINDOW_DAYS = 10  # how far before/after an Eid the countdown features reach


@lru_cache(maxsize=4)
def _holiday_table(first_year: int, last_year: int) -> pd.DataFrame:
    import holidays

    bd = holidays.BD(years=range(first_year, last_year + 1), language="en_US")
    rows = [{"date": pd.Timestamp(d), "name": name} for d, name in bd.items()]
    return pd.DataFrame(rows).sort_values("date")


def _first_days(table: pd.DataFrame, keyword: str) -> pd.DatetimeIndex:
    days = table.loc[table["name"].str.contains(keyword, case=False), "date"].sort_values()
    # Holidays span several days; keep the first day of each run.
    starts = days[days.diff() != pd.Timedelta(days=1)]
    return pd.DatetimeIndex(starts)


def _signed_days_to_nearest(dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> np.ndarray:
    """Days from each date to the nearest event (negative = before the event),
    clipped to ±WINDOW_DAYS; beyond the window the value is WINDOW_DAYS + 1."""
    d = dates.normalize().values.astype("datetime64[D]").astype(np.int64)
    e = events.values.astype("datetime64[D]").astype(np.int64)
    if len(e) == 0:
        return np.full(len(d), WINDOW_DAYS + 1)
    idx = np.clip(np.searchsorted(e, d), 1, len(e) - 1)
    before, after = e[idx - 1], e[idx]
    nearest = np.where(np.abs(d - before) <= np.abs(after - d), before, after)
    diff = d - nearest
    return np.where(np.abs(diff) <= WINDOW_DAYS, diff, WINDOW_DAYS + 1)


def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    times = pd.DatetimeIndex(times)
    table = _holiday_table(times.min().year - 1, times.max().year + 1)
    holiday_days = set(table["date"])
    fitr = _first_days(table, "Fitr")
    adha = _first_days(table, "Adha")

    out = pd.DataFrame(index=times)
    out["hour"] = times.hour
    out["dow"] = times.dayofweek  # Friday = 4 is the weekly holiday; Saturday = 5 for offices
    out["doy"] = times.dayofyear
    out["month"] = times.month
    out["is_holiday"] = times.normalize().isin(holiday_days).astype("int8")
    out["days_to_eid_fitr"] = _signed_days_to_nearest(times, fitr)
    out["days_to_eid_adha"] = _signed_days_to_nearest(times, adha)

    # Ramadan ≈ the 30 days before the first day of Eid-ul-Fitr.
    d = times.normalize()
    next_fitr_idx = np.searchsorted(fitr.values, d.values)
    next_fitr = pd.DatetimeIndex(
        np.where(next_fitr_idx < len(fitr), fitr.values[np.clip(next_fitr_idx, 0, len(fitr) - 1)], np.datetime64("NaT"))
    )
    days_before = (next_fitr - d).days
    in_ramadan = (days_before >= 1) & (days_before <= 30)
    out["ramadan"] = np.asarray(in_ramadan, dtype="int8")
    out["ramadan_day"] = np.where(in_ramadan, 31 - days_before, 0)
    return out
