"""Anomalies page: run detection and review flagged observations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Anomalies")
    contamination = st.slider("Sensitivity (contamination)", 0.01, 0.10, 0.02, 0.01)

    if st.button("Run detection", type="primary"):
        try:
            res = client.scan_anomalies(pid, contamination)
            st.success(f"Analyzed {res['rows_analyzed']} rows · found {res['anomalies_found']} anomalies.")
        except RiskLensAPIError as exc:
            st.error(str(exc))

    try:
        rows = client.anomalies(pid)
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    if not rows:
        st.info("No stored anomalies. Run detection above.")
        return

    df = pd.DataFrame(rows)
    fig = go.Figure(
        go.Scatter(
            x=df["date"], y=df["anomaly_score"], mode="markers",
            marker=dict(size=9, color=df["anomaly_score"], colorscale="OrRd", showscale=True),
            text=df["ticker"] + " · " + df["anomaly_type"],
        )
    )
    fig.update_layout(title="Anomaly scores over time", yaxis_title="score", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomalies")
    st.dataframe(
        df[["date", "ticker", "anomaly_type", "anomaly_score"]].sort_values("anomaly_score", ascending=False),
        use_container_width=True, hide_index=True,
    )
