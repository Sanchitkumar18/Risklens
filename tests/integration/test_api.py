"""End-to-end API tests exercising the full REST surface via TestClient."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.pipelines.synthetic_data import generate_market_data

BASE = "/api/v1"


@pytest.fixture()
def loaded_client(client_db):
    """A TestClient with ~1 year of market data uploaded via the API."""
    df = generate_market_data(start="2022-01-03", end="2022-12-31", seed=7)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    resp = client_db.post(
        f"{BASE}/market-data/upload",
        files={"file": ("data.csv", buf, "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["rows_written"] == len(df)
    return client_db


def _make_portfolio(client) -> int:
    r = client.post(f"{BASE}/portfolios", json={"name": "Tech Growth"})
    assert r.status_code == 201
    pid = r.json()["id"]
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200)]:
        rp = client.post(
            f"{BASE}/portfolios/{pid}/positions",
            json={"ticker": t, "quantity": str(q), "average_price": str(ap)},
        )
        assert rp.status_code == 201
    return pid


@pytest.mark.integration
def test_market_data_query(loaded_client):
    r = loaded_client.get(f"{BASE}/market-data/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] > 0
    # Unknown ticker → 404 envelope.
    r404 = loaded_client.get(f"{BASE}/market-data/ZZZZ")
    assert r404.status_code == 404
    assert r404.json()["error"]["code"] == "MARKET_DATA_NOT_FOUND"


@pytest.mark.integration
def test_portfolio_crud_and_valuation(loaded_client):
    pid = _make_portfolio(loaded_client)

    got = loaded_client.get(f"{BASE}/portfolios/{pid}")
    assert got.status_code == 200
    assert len(got.json()["positions"]) == 3

    val = loaded_client.get(f"{BASE}/portfolios/{pid}/valuation").json()
    assert float(val["total_value"]) > 0
    assert pytest.approx(sum(val["weights"].values()), abs=1e-6) == 1.0


@pytest.mark.integration
def test_duplicate_portfolio_conflict(loaded_client):
    loaded_client.post(f"{BASE}/portfolios", json={"name": "Dup"})
    r = loaded_client.post(f"{BASE}/portfolios", json={"name": "Dup"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "DUPLICATE_RESOURCE"


@pytest.mark.integration
def test_invalid_position_422(loaded_client):
    pid = _make_portfolio(loaded_client)
    r = loaded_client.post(
        f"{BASE}/portfolios/{pid}/positions",
        json={"ticker": "AAPL", "quantity": "0", "average_price": "10"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_POSITION"


@pytest.mark.integration
def test_risk_and_correlation(loaded_client):
    pid = _make_portfolio(loaded_client)
    r = loaded_client.get(f"{BASE}/portfolios/{pid}/risk?confidence=0.99")
    assert r.status_code == 200
    body = r.json()
    assert body["confidence_level"] == 0.99
    assert body["volatility_annualized"] > 0
    assert len(body["risk_contributions"]) == 3

    c = loaded_client.get(f"{BASE}/portfolios/{pid}/correlation").json()
    assert set(c["tickers"]) == {"AAPL", "MSFT", "NVDA"}


@pytest.mark.integration
def test_risk_timeseries(loaded_client):
    pid = _make_portfolio(loaded_client)
    series = loaded_client.get(f"{BASE}/portfolios/{pid}/risk/timeseries").json()
    assert len(series) > 100
    assert {"date", "portfolio_value", "drawdown"} <= set(series[0])
    assert all(p["drawdown"] <= 1e-9 for p in series)  # drawdown is never positive


@pytest.mark.integration
def test_stress_endpoints(loaded_client):
    pid = _make_portfolio(loaded_client)
    scenarios = loaded_client.get(f"{BASE}/portfolios/{pid}/stress-test/scenarios").json()
    assert any(s["name"] == "market_crash" for s in scenarios)

    r = loaded_client.get(f"{BASE}/portfolios/{pid}/stress-test?scenario=market_crash").json()
    assert r["pct_loss"] < 0

    custom = loaded_client.post(
        f"{BASE}/portfolios/{pid}/stress-test/custom",
        json={"name": "x", "ticker_shocks": {"NVDA": -0.5}, "default_shock": 0.0},
    ).json()
    nvda = next(leg for leg in custom["legs"] if leg["ticker"] == "NVDA")
    assert nvda["shock"] == -0.5


@pytest.mark.integration
def test_anomaly_endpoints(loaded_client):
    pid = _make_portfolio(loaded_client)
    scan = loaded_client.post(f"{BASE}/portfolios/{pid}/anomalies/scan?contamination=0.03").json()
    assert scan["anomalies_found"] >= 0
    listing = loaded_client.get(f"{BASE}/portfolios/{pid}/anomalies").json()
    assert len(listing) == scan["anomalies_found"]


@pytest.mark.integration
def test_alert_flow(loaded_client):
    pid = _make_portfolio(loaded_client)
    scan = loaded_client.post(f"{BASE}/portfolios/{pid}/alerts/scan", json={}).json()
    assert scan["breaches"] >= 0

    alerts = loaded_client.get(f"{BASE}/portfolios/{pid}/alerts").json()
    if alerts:
        aid = alerts[0]["id"]
        ack = loaded_client.post(f"{BASE}/portfolios/{pid}/alerts/{aid}/acknowledge").json()
        assert ack["acknowledged"] is True


@pytest.mark.integration
def test_portfolio_not_found_404(loaded_client):
    r = loaded_client.get(f"{BASE}/portfolios/99999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PORTFOLIO_NOT_FOUND"


@pytest.mark.integration
def test_position_update_and_delete(loaded_client):
    pid = _make_portfolio(loaded_client)
    pos = loaded_client.get(f"{BASE}/portfolios/{pid}").json()["positions"][0]

    upd = loaded_client.patch(
        f"{BASE}/portfolios/{pid}/positions/{pos['id']}", json={"quantity": "200"}
    )
    assert upd.status_code == 200
    from decimal import Decimal
    assert Decimal(upd.json()["quantity"]) == Decimal("200")

    dele = loaded_client.delete(f"{BASE}/portfolios/{pid}/positions/{pos['id']}")
    assert dele.status_code == 204

    # Deleting again → 404 with the position-not-found code.
    again = loaded_client.delete(f"{BASE}/portfolios/{pid}/positions/{pos['id']}")
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "POSITION_NOT_FOUND"


@pytest.mark.integration
def test_delete_portfolio(loaded_client):
    pid = _make_portfolio(loaded_client)
    assert loaded_client.delete(f"{BASE}/portfolios/{pid}").status_code == 204
    assert loaded_client.get(f"{BASE}/portfolios/{pid}").status_code == 404
