"""Forecast API with the database and the registry faked out."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from powerfc.serve import app as app_module


@pytest.fixture
def client(monkeypatch):
    fc = pd.DataFrame(
        {
            "origin": pd.Timestamp("2025-03-08 23:00"),
            "target_time": pd.date_range("2025-03-09 00:00", periods=72, freq="h"),
            "lead": range(1, 73),
            "block": [24] * 24 + [48] * 24 + [72] * 24,
            "p10": 9000.0,
            "p50": 10000.0,
            "p90": 11000.0,
            "model_version": "3",
            "data_version": "abc123",
            "created_at": pd.Timestamp.now(tz="UTC"),
        }
    )
    monkeypatch.setattr(app_module.store, "latest_forecast", lambda h: fc[fc.lead <= h])
    monkeypatch.setattr(app_module, "load_champion", lambda: {"version": "3", "backtest_metrics": {"smape_24h": 2.8}})
    with TestClient(app_module.app) as c:
        yield c


def test_forecast_horizons(client):
    for h in (24, 48, 72):
        r = client.get(f"/forecast?horizon={h}")
        assert r.status_code == 200
        body = r.json()
        assert len(body["points"]) == h and body["model_version"] == "3" and body["unit"] == "MW"


def test_invalid_horizon_is_422(client):
    assert client.get("/forecast?horizon=12").status_code == 422


def test_ready_and_model(client):
    assert client.get("/ready").json()["champion"] == "3"
    assert client.get("/model").json()["backtest_metrics"]["smape_24h"] == 2.8


def test_reload_picks_up_a_new_champion(client, monkeypatch):
    monkeypatch.setattr(app_module, "load_champion", lambda: {"version": "4", "backtest_metrics": {}})
    assert client.post("/admin/reload").json() == {"reloaded": True, "version": "4"}
    assert client.get("/ready").json()["champion"] == "4"


def test_removed_endpoints_are_gone(client):
    assert client.get("/metrics").status_code == 404
    assert client.post("/hooks/alertmanager", json={}).status_code == 404
