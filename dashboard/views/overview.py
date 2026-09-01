"""Overview page: headline risk metrics at a glance."""

from __future__ import annotations

import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Overview")
    try:
        val = client.valuation(pid)
        risk = client.risk(pid, confidence=0.95)
        crash = client.stress(pid, "severe_crash")
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    var95 = next((v for v in risk["var_historical"] if v["confidence_level"] == 0.95), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio value", f"${float(val['total_value']):,.0f}")
    c1.metric("Unrealized P&L", f"${float(val['unrealized_pnl']):,.0f}")
    c2.metric("95% 1-day VaR", f"${var95['var_value']:,.0f}" if var95 else "—")
    c2.metric("Annualized volatility", f"{risk['volatility_annualized']:.1%}")
    c3.metric("Max drawdown", f"{risk['drawdown']['max_drawdown']:.1%}")
    c3.metric("Severe-crash loss", f"${float(crash['total_loss']):,.0f}")

    st.caption(
        f"As of {risk['as_of_date']} · {risk['observations']} observations · "
        f"gross exposure ${risk['gross_exposure']:,.0f}"
    )
    st.info(risk.get("disclaimer", ""), icon="ℹ️")
