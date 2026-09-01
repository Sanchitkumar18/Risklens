"""Unit tests for dashboard chart builders and the API client.

Views themselves call Streamlit at runtime; here we test the parts that carry logic
(chart construction and the HTTP client) without a Streamlit runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import plotly.graph_objects as go
import pytest

# The dashboard modules import as top-level (Streamlit adds the script dir to path).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

from api_client import RiskLensAPIError, RiskLensClient  # noqa: E402
from components import charts  # noqa: E402


# ── charts ──────────────────────────────────────────────────
@pytest.mark.unit
def test_correlation_heatmap_builds():
    matrix = {"A": {"A": 1.0, "B": 0.5}, "B": {"A": 0.5, "B": 1.0}}
    fig = charts.correlation_heatmap(matrix)
    assert isinstance(fig, go.Figure)


@pytest.mark.unit
def test_weights_pie_and_contribution_and_var():
    assert isinstance(charts.weights_pie({"A": 0.6, "B": 0.4}), go.Figure)
    contribs = [{"ticker": "A", "weight": 0.6, "percent": 0.7}, {"ticker": "B", "weight": 0.4, "percent": 0.3}]
    assert isinstance(charts.risk_contribution_bar(contribs), go.Figure)
    var = [{"confidence_level": 0.95, "var_value": 1000.0}, {"confidence_level": 0.99, "var_value": 1500.0}]
    assert isinstance(charts.var_bar(var), go.Figure)


@pytest.mark.unit
def test_timeseries_charts():
    points = [
        {"date": "2024-01-02", "portfolio_value": 100.0, "drawdown": 0.0, "rolling_vol": None},
        {"date": "2024-01-03", "portfolio_value": 90.0, "drawdown": -0.1, "rolling_vol": 0.02},
    ]
    assert isinstance(charts.value_line(points), go.Figure)
    assert isinstance(charts.drawdown_area(points), go.Figure)
    assert isinstance(charts.rolling_vol_line(points), go.Figure)


# ── API client (mock transport) ─────────────────────────────
def _client_with(handler) -> RiskLensClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="http://test/api/v1", transport=transport)
    return RiskLensClient("http://test/api/v1", client=http)


@pytest.mark.unit
def test_client_list_portfolios_hits_correct_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=[{"id": 1, "name": "P"}])

    client = _client_with(handler)
    result = client.list_portfolios()
    assert seen["path"] == "/api/v1/portfolios"
    assert result[0]["name"] == "P"


@pytest.mark.unit
def test_client_add_position_serializes_decimals_as_strings():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 5, "ticker": "AAPL"})

    client = _client_with(handler)
    client.add_position(1, "AAPL", 100, 150.5)
    assert seen["body"] == {"ticker": "AAPL", "quantity": "100", "average_price": "150.5"}


@pytest.mark.unit
def test_client_unwraps_error_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "PORTFOLIO_NOT_FOUND", "message": "nope"}})

    client = _client_with(handler)
    with pytest.raises(RiskLensAPIError, match="PORTFOLIO_NOT_FOUND"):
        client.get_portfolio(999)


@pytest.mark.unit
def test_client_handles_204():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    client = _client_with(handler)
    assert client.remove_position(1, 2) is None


@pytest.mark.unit
def test_all_views_import_with_render():
    from views import alerts, anomalies, assistant, overview, portfolio, risk_analytics, stress

    for mod in (overview, portfolio, risk_analytics, stress, anomalies, alerts, assistant):
        assert callable(mod.render)
