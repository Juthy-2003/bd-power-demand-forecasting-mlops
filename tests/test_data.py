import numpy as np
import pandas as pd
import pytest

from powerfc.data.clean import clean, fill_short_gaps, summary
from powerfc.data.contract import SchemaErrors, validate_clean, validate_raw


def test_contract_accepts_clean_data(raw):
    validate_clean(clean(validate_raw(raw)))


def test_missing_hours_become_explicit_nan(raw):
    raw = raw.drop(index=range(100, 110))
    c = clean(raw)
    assert len(c) == len(raw) + 10
    assert c["missing"].sum() == 10


def test_typos_are_flagged_and_removed_from_the_target(raw):
    raw.loc[200, "demand_mw"] = 121_000  # real example from the PGCB file
    night = 24 * 12 + 3  # 03:00, so tripling stays inside the plausible range (about 21.7k MW)
    raw.loc[night, "demand_mw"] = raw.loc[night, "demand_mw"] * 3  # in range, but ~2.4x the weekly median
    c = clean(raw)
    assert np.isnan(c.loc[200, "demand_mw"]) and "demand_range" in c.loc[200, "quality_flag"]
    assert np.isnan(c.loc[night, "demand_mw"]) and c.loc[night, "quality_flag"] == "demand_spike"
    assert summary(c)["flagged_hours"] == 2


def test_only_short_gaps_are_filled():
    s = pd.Series([1.0, np.nan, np.nan, 4.0] + [np.nan] * 8 + [13.0])
    filled = fill_short_gaps(s, max_len=6)
    assert filled.iloc[1:3].tolist() == [2.0, 3.0]
    assert filled.iloc[4:12].isna().all()  # a long gap is never invented


def test_contract_rejects_a_broken_scrape(raw):
    c = clean(raw)
    c.loc[: len(c) // 5, "demand_mw"] = np.nan  # 20% of the target missing
    with pytest.raises(SchemaErrors):
        validate_clean(c)


def test_contract_rejects_negative_shedding(raw):
    raw.loc[5, "load_shedding"] = -10
    with pytest.raises(SchemaErrors):
        validate_raw(raw)
