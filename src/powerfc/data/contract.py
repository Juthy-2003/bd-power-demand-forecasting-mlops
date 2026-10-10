"""The data contract between ingestion and everything downstream.

If validation fails, the pipeline stops before training: a bad scrape must
never become a bad model. Pandera checks column names, types, ranges,
nullability and timestamp uniqueness.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from powerfc.data.clean import DEMAND_RANGE

MW = Check.in_range(0, 100_000)

RAW = DataFrameSchema(
    {
        "datetime": Column("datetime64[us]", unique=True, nullable=False),
        "generation_mw": Column(float, nullable=True),
        "demand_mw": Column(float, nullable=True),
        "load_shedding": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
    checks=[Check(lambda df: df["datetime"].is_monotonic_increasing, error="timestamps must be sorted")],
)

CLEAN = DataFrameSchema(
    {
        "datetime": Column(
            "datetime64[us]",
            checks=[Check(lambda s: (s.diff().dropna() == s.diff().dropna().iloc[0]).all(), error="hourly grid")],
            unique=True,
        ),
        "demand_mw": Column(float, Check.in_range(*DEMAND_RANGE), nullable=True),
        "demand_filled": Column(float, Check.in_range(*DEMAND_RANGE), nullable=True),
        "load_shedding": Column(float, MW, nullable=True),
        "generation_mw": Column(float, Check.in_range(0, 90_000), nullable=True),
        "missing": Column(bool),
        "imputed": Column(bool),
        "shedding": Column(bool),
    },
    strict=False,
    coerce=True,
    checks=[
        # Completeness guard: if more than 10% of hours have no usable target,
        # something upstream broke.
        Check(lambda df: df["demand_mw"].notna().mean() >= 0.90, error="at least 90% of hours need a demand value"),
    ],
)


def validate_raw(df):
    return RAW.validate(df, lazy=True)


def validate_clean(df):
    return CLEAN.validate(df, lazy=True)


SchemaErrors = pa.errors.SchemaErrors
