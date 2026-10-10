# 0001 — Rolling-origin backtesting, never random K-fold

**Status:** Accepted (2026-10-10)

## Context
Random K-fold puts future hours in the training set and past hours in the test set, so the model is graded on a past it has partly seen. Demand is strongly autocorrelated: a neighbour hour in training almost gives away the answer.

## Decision
Every evaluation is a rolling-origin backtest (`src/powerfc/evaluate/backtest.py`). Origins every 7 days at 23:00 Dhaka time; each forecasts the next 72 hours. Models are refit every N origins on all targets at or before the refit origin (an expanding window). Direct multi-horizon: one model per block (24, 48, 72 h), and a block's features are lags of at least that many hours.

A unit test (`tests/test_leakage.py`) mutates every value after the allowed cut-off and asserts the features do not change; a second test proves the mutation is not vacuous.

## Consequences
- ✅ Metrics mean what they would mean in production.
- ⚠️ Slow: a 2-year backtest refits 26 times × 3 blocks × (3 quantiles + 1 classifier). Refitting every 4 weeks instead of every week is the compromise; a model is still never trained on any hour it forecasts.
- The last 12 months (2025-03-09 → 2026-03-08) are a holdout replayed day by day as "new data", not used for any choice.
