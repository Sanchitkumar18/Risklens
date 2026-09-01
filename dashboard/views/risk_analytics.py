"""Risk Analytics page: VaR, contribution, correlation, and time series."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient
from components import charts


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Risk Analytics")
    confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95)
    try:
        risk = client.risk(pid, confidence=confidence)
        corr = client.correlation(pid)
        series = client.risk_timeseries(pid)
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    headline = next((v for v in risk["var_historical"] if v["confidence_level"] == confidence), None)
    if headline:
        c1.metric(f"{confidence:.0%} VaR", f"${headline['var_value']:,.0f}", f"{headline['var_fraction']:.2%}")
    if risk.get("var_parametric"):
        c2.metric("Parametric VaR", f"${risk['var_parametric']['var_value']:,.0f}")
    c3.metric("Annualized volatility", f"{risk['volatility_annualized']:.1%}")

    left, right = st.columns(2)
    left.plotly_chart(charts.var_bar(risk["var_historical"]), use_container_width=True)
    right.plotly_chart(charts.risk_contribution_bar(risk["risk_contributions"]), use_container_width=True)

    st.plotly_chart(charts.value_line(series), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(charts.drawdown_area(series), use_container_width=True)
    b.plotly_chart(charts.rolling_vol_line(series), use_container_width=True)

    st.plotly_chart(charts.correlation_heatmap(corr["matrix"]), use_container_width=True)
    pairs = corr.get("high_correlation_pairs", [])
    if pairs:
        st.subheader("Highly correlated pairs (|ρ| ≥ 0.8)")
        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)
    else:
        st.caption("No asset pairs exceed |ρ| = 0.8.")
