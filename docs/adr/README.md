# Architecture Decision Records

| # | Decision |
|---|---|
| [0001](0001-rolling-origin-backtest.md) | Rolling-origin backtesting, never random K-fold |
| [0002](0002-lightgbm-quantile-conformal.md) | LightGBM quantile models with conformal calibration, not deep learning — *superseded in part by 0009 (no CQR, no shedding classifier)* |
| [0003](0003-airflow-external-python.md) | Airflow 3 (LocalExecutor) with the pipeline in its own virtualenv |
| [0004](0004-promotion-gate.md) | Champion/challenger promotion gate rules — *coverage band superseded by 0009* |
| [0005](0005-k3s-on-oci.md) | k3s on one Oracle ARM VM, not managed Kubernetes; CronJobs instead of Airflow on the node — *amended by 0009* |
| [0006](0006-redpanda-timescale-no-spark.md) | Redpanda + TimescaleDB for the synthetic stream; no Spark — *superseded by 0009 (stream removed)* |
| [0007](0007-mlflow-artifact-proxy.md) | MLflow serves its own artifacts (no MinIO) |
| [0008](0008-psi-alerts-evidently-report.md) | PSI and live error trigger retraining; Evidently produces the report — *trigger path superseded by 0009* |
| [0009](0009-trim-for-defensibility.md) | Trim for defensibility: stream, Prometheus stack, CQR and shedding score removed; one daily DAG decides and triggers retraining |
