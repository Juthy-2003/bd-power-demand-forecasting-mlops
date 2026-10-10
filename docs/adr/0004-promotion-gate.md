# 0004 — Champion/challenger promotion gate

**Status:** Superseded (2026-10-10, trim) by [0009](0009-trim-for-defensibility.md) for the coverage band only. The gate's structure (beat seasonal-naive, coverage band, latency, compare with the champion on the same folds) is unchanged. The band below, [0.72, 0.92], was for CQR-calibrated intervals; with CQR removed the band was re-set from the raw intervals' backtest coverage (ADR 0009). Kept below as the record, including the honest account of the first revision.

## Decision
`src/powerfc/registry/promote.py` is the only code that moves the `champion` alias in the MLflow registry. Used by the Airflow DAG, the CLI and (via `powerfc.ci_gate`) CI.

Absolute guards: beats seasonal-naive at every block; 80% interval coverage in [0.72, 0.92]; 72-hour predict latency ≤ 3 s.

Against the champion, on the same data version and the same folds:
- **different recipe** (parameters/features): ≥ 1% lower sMAPE at every block;
- **same recipe, newer data** ("refresh"): no more than 5% worse than the champion's recorded sMAPE;
- **same recipe, same data:** nothing new; keep the champion.

A recipe is hashed (`recipe_hash`); the backtest window is a pure function of the data end, so two recipes on one data version see identical folds. If the champion's recipe has not been evaluated on the challenger's data yet, the gate runs that evaluation and logs it.

## The revision, recorded honestly
The first band was [0.72, 0.88]. The first drift-triggered retrains (model versions 3 and 4) had better sMAPE but 88.4% / 90.3% coverage and were refused. The upper edge was then raised to 0.92 because over-coverage only means wider-than-needed intervals, while under-coverage misleads an operator. This was decided *after* seeing a result; the refused versions remain in the registry with their reasons.

## Consequences
- ✅ No manual model copying; every promotion and refusal carries its reason as a model-version tag.
- ⚠️ "Refresh" compares numbers measured on different windows; it guards against regression, it is not an improvement claim.
