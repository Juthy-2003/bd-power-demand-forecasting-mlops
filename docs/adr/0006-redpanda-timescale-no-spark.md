# 0006 — Redpanda + TimescaleDB for the SYNTHETIC stream; no Spark

**Status:** Superseded (2026-10-10, trim) by [0009](0009-trim-for-defensibility.md). The SYNTHETIC feeder stream (simulator, Redpanda, TimescaleDB, scorer, DuckDB micro-batch) was removed. Reason: it was synthetic, unrelated to the forecasting loop, and added a broker, a second database and ~500 lines that a junior must defend. The "no Spark" reasoning still holds for the real data. Kept below as the record of what was built and measured.

## Context
92k hourly rows cannot support a "time-series at scale" claim. A synthetic stream of ~1,000 feeders exercises partitioned messaging, consumer groups, a time-series database and columnar files at realistic rates. **The feeders are synthetic**: national PGCB demand split by fixed random weights plus noise, outages and optional drift.

## Decision
- **Redpanda** (Kafka API, single binary, no JVM/ZooKeeper), topic `feeder.readings`, 6 partitions, key = feeder id so each feeder stays ordered.
- **Scorer**: one consumer in group `feeder-scorer`; batches of up to 5,000; COPY into a temp table then `INSERT … ON CONFLICT (feeder_id, ts) DO NOTHING`; offsets committed only after the write — at-least-once made idempotent.
- **TimescaleDB** hypertable (1-day chunks) for hot queries; hourly **Parquet** partitions for DuckDB/Polars.
- **No Spark**: one node and this data size; Spark's cluster overhead buys nothing.

## Consequences
Throughput and latency are measured on one laptop (README). The batching makes latency *higher* at low rates (the consumer waits up to 1 s to fill a batch) — a tuning knob, not a bug.
