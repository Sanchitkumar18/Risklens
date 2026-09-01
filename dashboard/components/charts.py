"""Plotly chart builders.

Each function takes plain Python data (exactly the JSON the API returns) and returns a
``plotly.graph_objects.Figure``. Keeping them free of Streamlit calls makes the
visualization logic unit-testable without a running app.
"""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go

_TRADING_DAYS = 252


def correlation_heatmap(matrix: dict[str, dict[str, float]]) -> go.Figure:
    """Heatmap of an asset return correlation matrix."""
    tickers = list(matrix.keys())
    z = [[matrix[a][b] for b in tickers] for a in tickers]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=tickers, y=tickers, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(title="Asset return correlation", height=420)
    return fig


def weights_pie(weights: dict[str, float]) -> go.Figure:
    """Pie of portfolio weights by gross exposure."""
    labels = list(weights.keys())
    values = [abs(weights[t]) for t in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent"))
    fig.update_layout(title="Portfolio weights", height=380)
    return fig


def risk_contribution_bar(contributions: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar comparing capital weight vs. risk (volatility) contribution."""
    tickers = [c["ticker"] for c in contributions]
    weights = [c["weight"] * 100 for c in contributions]
    risk = [c["percent"] * 100 for c in contributions]
    fig = go.Figure()
    fig.add_bar(name="Weight %", x=tickers, y=weights, marker_color="#7C9CBF")
    fig.add_bar(name="Risk contribution %", x=tickers, y=risk, marker_color="#C05B4D")
    fig.update_layout(
        title="Weight vs. risk contribution", barmode="group",
        yaxis_title="%", height=380,
    )
    return fig


def var_bar(var_historical: list[dict[str, Any]]) -> go.Figure:
    """Bar of historical VaR (dollar loss) across confidence levels."""
    levels = [f"{v['confidence_level']:.0%}" for v in var_historical]
    values = [v["var_value"] for v in var_historical]
    fig = go.Figure(go.Bar(x=levels, y=values, marker_color="#C05B4D", text=[f"${v:,.0f}" for v in values], textposition="auto"))
    fig.update_layout(title="Historical 1-day VaR by confidence", yaxis_title="Loss ($)", height=360)
    return fig


def value_line(points: list[dict[str, Any]]) -> go.Figure:
    """Portfolio value over time."""
    dates = [p["date"] for p in points]
    values = [p["portfolio_value"] for p in points]
    fig = go.Figure(go.Scatter(x=dates, y=values, mode="lines", line=dict(color="#4C78A8")))
    fig.update_layout(title="Portfolio value", yaxis_title="$", height=340)
    return fig


def drawdown_area(points: list[dict[str, Any]]) -> go.Figure:
    """Drawdown (%) over time as a filled area."""
    dates = [p["date"] for p in points]
    dd = [p["drawdown"] * 100 for p in points]
    fig = go.Figure(go.Scatter(x=dates, y=dd, mode="lines", fill="tozeroy", line=dict(color="#C05B4D")))
    fig.update_layout(title="Drawdown", yaxis_title="%", height=340)
    return fig


def rolling_vol_line(points: list[dict[str, Any]]) -> go.Figure:
    """Annualized rolling volatility (%) over time."""
    dates = [p["date"] for p in points if p.get("rolling_vol") is not None]
    vol = [p["rolling_vol"] * math.sqrt(_TRADING_DAYS) * 100 for p in points if p.get("rolling_vol") is not None]
    fig = go.Figure(go.Scatter(x=dates, y=vol, mode="lines", line=dict(color="#54A24B")))
    fig.update_layout(title="Rolling volatility (annualized)", yaxis_title="%", height=340)
    return fig
