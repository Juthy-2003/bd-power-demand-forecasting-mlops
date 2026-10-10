# 0008 — PSI and live error trigger retraining; Evidently produces the report

**Status:** Superseded (2026-10-10, trim) by [0009](0009-trim-for-defensibility.md) for the trigger path only. PSI > 0.25 or error ratio > 1.5 is still the rule and Evidently still writes the report. What changed: the decision is now made inside the daily Airflow DAG (`drift_check` → `drift_detected` → `TriggerDagRunOperator`), not by Prometheus rules → Alertmanager → API webhook → Airflow REST. Reason: four hops and three extra services replaced by one task boundary. Kept below as the record.

## Context
Evidently's DataDriftPreset chooses a test per column (Kolmogorov–Smirnov p-values on small samples, distances such as Wasserstein on larger ones). On large samples, p-value tests flag tiny, harmless shifts; in a quick check a column drawn from the *same* distribution was flagged at p = 0.02.

## Decision
- The **alert** uses PSI on 8 monitored features (`powerfc_psi_max > 0.25`) and the champion's **error ratio** — recent sMAPE on hours with actuals ÷ its backtest sMAPE (`> 1.5`).
- Evidently's HTML report is logged to MLflow for every check, for humans.
- Prometheus scrapes the gauges from the API; Alertmanager groups drift alerts by `action` (one retrain for simultaneous alerts) and calls `POST /hooks/alertmanager`, which triggers `train_evaluate_promote` through Airflow's REST API.

## Consequences
- ✅ One measured closed loop on 2026-10-10: PSI 8.27 and error ratio 1.61 after replaying 28 days (Ramadan → Eid → hot season) fired both alerts and Airflow ran the retrain (details in the CHANGELOG).
- ⚠️ `for: 1m` in the rules is for the demo; production should require hours.
