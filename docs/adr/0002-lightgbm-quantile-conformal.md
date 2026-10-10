# 0002 — LightGBM quantile models with conformal calibration

**Status:** Superseded (2026-10-10, trim) by [0009](0009-trim-for-defensibility.md). The LightGBM quantile models stay; the conformal (CQR) calibration step and the load-shedding classifier were removed. Intervals are now the raw P10–P90 of the quantile models, and their measured coverage is reported as it is. Reason: two extra methods (conformal prediction, a second model type with PR-AUC) that a junior must defend, for a project whose point is the MLOps loop. Kept below as the record of what was built first.

## Decision
Per horizon block, three LightGBM quantile regressors (α = 0.1, 0.5, 0.9) predicting demand *relative to the mean of the 168 hours ending at the cut-off* (national demand grows about 5% a year and trees cannot extrapolate a level they never saw), plus a LightGBM classifier for P(load shedding > 0).

The raw quantile intervals were too narrow: 80% intervals covered 61–68% in the first 2-year backtest. They are calibrated with conformalized quantile regression (Romano et al., 2019): the last 6 weeks of each training window are held out, the conformity score `max(P10 − y, y − P90)` is measured there, and its 80% quantile widens (or narrows) the interval. Coverage moved to 77–83%.

## Alternatives
- **N-BEATS / LSTM / Temporal Fusion Transformer:** harder to explain, slower on CPU, and unjustifiable before a strong tree baseline exists. Can enter later as a challenger through the same gate.
- **Gaussian intervals around a point forecast:** assume symmetric, constant-width errors; demand errors are neither.

## Consequences
- ✅ Beats seasonal-naive by 33–60% sMAPE depending on horizon (README).
- ⚠️ Calibrated intervals are wider (≈1.5–2.5 GW at P10–P90).
- ⚠️ Independent quantile models can cross; predictions are sorted and the crossing rate is logged (≈0.4–1.2%).
