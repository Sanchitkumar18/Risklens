"""Portfolio page: holdings, weights, and position management."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient
from components import charts


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Portfolio")
    try:
        val = client.valuation(pid)
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    holdings = val["holdings"]
    if holdings:
        df = pd.DataFrame(holdings)[
            ["ticker", "quantity", "average_price", "last_price", "market_value", "weight", "unrealized_pnl"]
        ]
        df["weight"] = (df["weight"].astype(float) * 100).round(1)
        left, right = st.columns([3, 2])
        left.subheader("Holdings")
        left.dataframe(df, use_container_width=True, hide_index=True)
        right.plotly_chart(charts.weights_pie(val["weights"]), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total value", f"${float(val['total_value']):,.0f}")
        m2.metric("Gross exposure", f"${float(val['gross_exposure']):,.0f}")
        m3.metric("Net exposure", f"${float(val['net_exposure']):,.0f}")
    else:
        st.info("This portfolio has no positions yet. Add one below.")

    st.divider()
    st.subheader("Add position")
    with st.form("add_position"):
        col = st.columns(3)
        ticker = col[0].text_input("Ticker", value="AAPL")
        qty = col[1].number_input("Quantity", value=100.0, step=1.0)
        avg = col[2].number_input("Average price", value=150.0, step=1.0, min_value=0.01)
        if st.form_submit_button("Add / update"):
            try:
                client.add_position(pid, ticker.strip().upper(), qty, avg)
                st.success(f"Position set: {ticker.upper()}")
                st.rerun()
            except RiskLensAPIError as exc:
                st.error(str(exc))

    if holdings:
        st.subheader("Remove position")
        pf = client.get_portfolio(pid)
        for pos in pf["positions"]:
            c = st.columns([4, 1])
            c[0].write(f"**{pos['ticker']}** — {pos['quantity']} @ ${pos['average_price']}")
            if c[1].button("Remove", key=f"rm_{pos['id']}"):
                try:
                    client.remove_position(pid, pos["id"])
                    st.rerun()
                except RiskLensAPIError as exc:
                    st.error(str(exc))
