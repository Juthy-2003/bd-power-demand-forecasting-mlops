"""The no-peeking rule, enforced: features for block b at target hour T
must not change when anything after T − b changes."""

import numpy as np
import pandas as pd
import pytest

from powerfc import config
from powerfc.data.clean import clean
from powerfc.features.build import block_features, lags_for


@pytest.mark.parametrize("block", config.BLOCKS)
def test_lags_never_shorter_than_block(block):
    assert min(lags_for(block)) >= block


@pytest.mark.parametrize("block", config.BLOCKS)
def test_features_ignore_everything_after_the_cutoff(raw, block):
    target = pd.Timestamp("2024-03-20 15:00")
    cutoff = target - pd.Timedelta(hours=block)  # last hour the model may see
    before = block_features(clean(raw), block).loc[target]

    mutated = raw.copy()
    future = mutated["datetime"] > cutoff
    rng = np.random.default_rng(1)
    for col in ["demand_mw", "generation_mw", "load_shedding"]:
        mutated.loc[future, col] = rng.uniform(2000, 20000, future.sum()).round()
    after = block_features(clean(mutated), block).loc[target]

    pd.testing.assert_series_equal(before, after, check_names=False)


def test_mutating_data_inside_the_window_does_change_features(raw):
    """Guard against a vacuous test: data at the cutoff must matter."""
    block, target = 24, pd.Timestamp("2024-03-20 15:00")
    before = block_features(clean(raw), block).loc[target]
    mutated = raw.copy()
    mutated.loc[mutated["datetime"] == target - pd.Timedelta(hours=block), "demand_mw"] *= 1.3
    after = block_features(clean(mutated), block).loc[target]
    assert not before.equals(after)
