"""Stress Testing page: run built-in and custom scenarios."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient


def _pnl_chart(legs: list[dict]) -> go.Figure:
    legs = sorted(legs, key=lambda leg: leg["pnl"])
    fig = go.Figure(
        go.Bar(
            x=[leg["ticker"] for leg in legs],
            y=[float(leg["pnl"]) for leg in legs],
            marker_color=["#C05B4D" if float(leg["pnl"]) < 0 else "#54A24B" for leg in legs],
            text=[f"${float(leg['pnl']):,.0f}" for leg in legs],
            textposition="auto",
        )
    )
    fig.update_layout(title="P&L by asset", yaxis_title="$", height=380)
    return fig


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Stress Testing")
    try:
        scenarios = client.stress_scenarios(pid)
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    names = [s["name"] for s in scenarios]
    descriptions = {s["name"]: s["description"] for s in scenarios}
    choice = st.selectbox("Scenario", names, format_func=lambda n: f"{n} — {descriptions[n]}")

    if st.button("Run scenario", type="primary"):
        try:
            _show_result(client.stress(pid, choice))
        except RiskLensAPIError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Custom scenario")
    st.caption("Shock a single ticker; everything else moves by the default shock.")
    with st.form("custom"):
        c = st.columns(3)
        ticker = c[0].text_input("Ticker", "NVDA")
        shock = c[1].slider("Ticker shock %", -50, 50, -25) / 100
        default = c[2].slider("Default shock %", -50, 50, -10) / 100
        if st.form_submit_button("Run custom"):
            payload = {"name": "custom", "ticker_shocks": {ticker.upper(): shock}, "default_shock": default}
            try:
                _show_result(client.stress_custom(pid, payload))
            except RiskLensAPIError as exc:
                st.error(str(exc))


def _show_result(res: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total loss", f"${float(res['total_loss']):,.0f}")
    c2.metric("Return", f"{res['pct_loss']:.1%}")
    c3.metric("Worst assets", ", ".join(res["worst_assets"]) or "—")
    st.plotly_chart(_pnl_chart(res["legs"]), use_container_width=True)
    df = pd.DataFrame(res["legs"])[["ticker", "shock", "price_before", "price_after", "pnl", "pct_of_portfolio"]]
    df["shock"] = (df["shock"].astype(float) * 100).round(1)
    df["pct_of_portfolio"] = (df["pct_of_portfolio"].astype(float) * 100).round(2)
    st.dataframe(df, use_container_width=True, hide_index=True)
