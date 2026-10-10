# Everything a reviewer needs, one target each.
.PHONY: setup test lint data backtest up bootstrap replay k3d-up down

setup:        ## install dependencies
	uv sync

test:         ## unit tests: leakage, data contract, gate rules, drift rule, API
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

data:         ## download (sha256-pinned) and build the cleaned table
	uv run powerfc fetch && uv run powerfc assemble

backtest:     ## the full 2-year backtest + 1-year holdout used for the README table (~10–15 min)
	uv run python bench/full_backtest.py

up:           ## local stack: Postgres, MLflow, Airflow, forecast API
	docker compose -p powerfc up -d --build

bootstrap: data  ## first champion via Airflow, then switch the daily DAG on
	uv run python scripts/airflow_ctl.py run train_evaluate_promote
	uv run python scripts/airflow_ctl.py unpause daily_pipeline

replay:       ## one day of the daily DAG: ingest the next holdout day → forecast → drift check → maybe retrain
	uv run python scripts/airflow_ctl.py run daily_pipeline

k3d-up:       ## local Kubernetes with the same chart as the cloud node
	./deploy/k3d/up.sh

down:
	docker compose -p powerfc down
