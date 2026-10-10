"""The Postgres table the pipeline writes and the API reads.

forecasts  one row per (origin, target hour, model version); upserted, so
           re-running a forecast for the same origin is idempotent.

Drift results are not stored here: each check is logged to MLflow (metrics +
the Evidently report) and returned to Airflow as the task's result.
"""

import os

import pandas as pd
import psycopg

DSN = os.environ.get("POWERFC_DB", "postgresql://powerfc:powerfc@localhost:5433/powerfc")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    origin        timestamp NOT NULL,
    target_time   timestamp NOT NULL,
    lead          int NOT NULL,
    block         int NOT NULL,
    p10           double precision,
    p50           double precision,
    p90           double precision,
    model_version text NOT NULL,
    data_version  text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (origin, target_time, model_version)
);
"""


def connect():
    return psycopg.connect(DSN, autocommit=True)


def ensure_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def write_forecast(fc: pd.DataFrame, model_version: str, data_version: str) -> int:
    ensure_schema()
    rows = [
        (
            r.origin.to_pydatetime(),
            r.target_time.to_pydatetime(),
            int(r.lead),
            int(r.block),
            float(r.p10),
            float(r.p50),
            float(r.p90),
            str(model_version),
            data_version,
        )
        for r in fc.itertuples()
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO forecasts (origin, target_time, lead, block, p10, p50, p90, model_version, data_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (origin, target_time, model_version) DO UPDATE
               SET p10 = EXCLUDED.p10, p50 = EXCLUDED.p50, p90 = EXCLUDED.p90,
                   data_version = EXCLUDED.data_version, created_at = now()""",
            rows,
        )
    return len(rows)


def latest_forecast(max_lead: int) -> pd.DataFrame:
    with connect() as conn:
        cur = conn.execute(
            """SELECT origin, target_time, lead, block, p10, p50, p90, model_version, data_version, created_at
               FROM forecasts
               WHERE (origin, created_at) = (SELECT origin, max(created_at) FROM forecasts
                                             GROUP BY origin ORDER BY origin DESC LIMIT 1)
                 AND lead <= %s
               ORDER BY target_time""",
            (max_lead,),
        )
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def forecasts_between(start, end) -> pd.DataFrame:
    """Every forecast whose target hour falls in [start, end), newest origin first."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM forecasts WHERE target_time >= %s AND target_time < %s ORDER BY origin DESC", (start, end)
        )
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
