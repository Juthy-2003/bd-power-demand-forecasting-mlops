# 0009 — Trim for defensibility

**Status:** Accepted (2026-10-10). Supersedes [0006](0006-redpanda-timescale-no-spark.md); supersedes [0002](0002-lightgbm-quantile-conformal.md) (calibration and shedding classifier), [0004](0004-promotion-gate.md) (coverage band) and [0008](0008-psi-alerts-evidently-report.md) (trigger path) in part; amends [0005](0005-k3s-on-oci.md).

## Context
The first version carried four things a fresh graduate would have to defend in an ML Engineer / MLOps interview without adding much to the job's core (a continuous-training loop on real data):

- a **SYNTHETIC feeder stream** (simulator, Redpanda, TimescaleDB, a scorer, a DuckDB micro-batch, a benchmark) unrelated to the forecasting loop;
- **Prometheus + Alertmanager + Grafana**, used mainly as a four-hop path for the retrain trigger (drift row in Postgres → API `/metrics` → Prometheus rule → Alertmanager → API webhook → Airflow REST);
- **conformal calibration (CQR)** of the quantile intervals;
- a **load-shedding classifier** scored with PR-AUC.

The three daily DAGs (ingest, forecast, drift) also ran independently, so the forecast and the drift check did not wait for the day's data.

## Decision
1. Remove the synthetic stream, the Prometheus stack, CQR and the shedding classifier, with their code, deps, compose services, Helm templates and docs. Kubeflow was considered, not used.
2. **One daily DAG**, `daily_pipeline`: `ingest → forecast → drift_check → drift_detected → trigger_retrain`. `drift_check` returns `drift_detected` (PSI > 0.25 on a monitored feature, or error ratio > 1.5 — the same thresholds as before, in `monitoring/drift.py: decide_retrain`). `drift_detected` is a short-circuit task; when it is true, a `TriggerDagRunOperator` starts `train_evaluate_promote`. Triggering the daily DAG by hand replays the next holdout day, so the separate `replay_day` DAG is gone.
3. **Drift is judged only on data the champion has not seen** (the hours after its training cut-off, at most the latest 14 days), and only once there are at least 7 days of it. Found on 2026-10-10 while re-running the loop: with the old "latest 14 days" window, the first replayed day already reported PSI 5.83 (most of that window was the champion's own training data, in a fast-rising spring) and triggered a retrain on day one; that retrain was refused (refresh regressed sMAPE by more than 5%). The 7-day minimum also gives a one-week pause after each promotion.
4. **Intervals are the raw P10–P90** of the LightGBM quantile models. The backtest and holdout report raw coverage and width as measured.
5. **Coverage band = [0.55, 0.92]**, set from the 2-year backtest (2023-03 → 2025-03) only. Raw coverage there was 68.3% / 65.6% / 61.2% at 1–24 / 25–48 / 49–72 h. The lower edge is about 6 points below the worst block, so the gate stops a model whose intervals have collapsed, not one that behaves like this recipe; the upper edge (unchanged) stops uselessly wide intervals. The band was fixed before the holdout table was read; the holdout (77.6% / 74.6% / 66.0%) was not used to choose it.
6. The API keeps `/health`, `/ready`, `/forecast`, `/model` and `/admin/reload`; no `/metrics`, no webhook. After a promotion the Airflow task, or on Kubernetes the training CronJob (`powerfc reload-api`), calls `/admin/reload`.

## Alternatives
- **Keep CQR:** it fixed coverage (79.5–80.5% in the 2-year backtest, with intervals 32–53% wider), but it is one more method to explain and the gate can be honest about raw coverage instead. If an operator needs calibrated intervals later, CQR can return as a new recipe through the same gate.
- **Keep Alertmanager for the trigger:** standard in production, but four hops for one boolean in a one-node demo. The decision is now one Python function plus one Airflow operator.
- **Airflow Assets (data-aware scheduling) between three DAGs:** works, but one DAG with explicit dependencies is simpler to read and to explain.

## Consequences
- ✅ Python excluding tests (src, dags, bench, scripts) went from 2,742 to 2,075 lines; Helm YAML from 524 to 311 lines plus 369 lines of Grafana JSON gone; compose from 11 containers (7 core + 4 stream) to 4. One retrain path instead of two.
- ✅ sMAPE is essentially unchanged (holdout 3.58 / 5.10 / 5.98% vs 3.57 / 5.20 / 6.00% with CQR); intervals are narrower and under-cover, and the README says so.
- ⚠️ 80% intervals cover only 61–78%: an operator must read P10–P90 as roughly a 65–75% interval.
- ⚠️ PSI on this strongly seasonal series is above 0.25 almost every week in the spring transition, so in practice drift means "weekly retrain while demand is changing fast". A same-weeks-last-year reference window would be a better test; not built.
- ⚠️ If the gate keeps refusing, the daily DAG triggers a retrain every day until one passes; a human alert after N refusals is not built.
- ⚠️ No metrics endpoint or dashboard: monitoring evidence is the MLflow `powerfc-demand-monitoring` experiment (PSI, error ratio, Evidently report per check) and the Airflow run history.
