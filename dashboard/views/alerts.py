"""Alerts page: evaluate thresholds, review and acknowledge alerts."""

from __future__ import annotations

import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient

_SEVERITY_ICON = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}


def render(client: RiskLensClient, pid: int) -> None:
    st.header("Alerts")

    with st.expander("Thresholds", expanded=False):
        c = st.columns(3)
        thresholds = {
            "var_limit": c[0].number_input("VaR limit", value=0.02, step=0.005, format="%.3f"),
            "volatility_limit": c[0].number_input("Volatility limit", value=0.20, step=0.05),
            "drawdown_limit": c[1].number_input("Drawdown limit", value=0.25, step=0.05),
            "max_single_weight": c[1].number_input("Max single weight", value=0.35, step=0.05),
            "anomaly_score_limit": c[2].number_input("Anomaly score limit", value=0.70, step=0.05),
            "stress_loss_limit": c[2].number_input("Stress loss limit", value=0.25, step=0.05),
        }

    if st.button("Evaluate thresholds", type="primary"):
        try:
            res = client.scan_alerts(pid, thresholds)
            st.success(f"{res['breaches']} breach(es) · {res['created']} new · by severity {res['by_severity']}")
        except RiskLensAPIError as exc:
            st.error(str(exc))

    try:
        alerts = client.alerts(pid)
    except RiskLensAPIError as exc:
        st.error(str(exc))
        return

    if not alerts:
        st.info("No alerts. Evaluate thresholds above.")
        return

    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    for a in sorted(alerts, key=lambda x: (x["acknowledged"], order.get(x["severity"], 9))):
        icon = _SEVERITY_ICON.get(a["severity"], "⚪")
        cols = st.columns([6, 1])
        status = "✓ acknowledged" if a["acknowledged"] else ""
        cols[0].markdown(f"{icon} **{a['severity']}** · {a['alert_type']} — {a['message']} {status}")
        if not a["acknowledged"]:
            if cols[1].button("Ack", key=f"ack_{a['id']}"):
                try:
                    client.acknowledge_alert(pid, a["id"])
                    st.rerun()
                except RiskLensAPIError as exc:
                    st.error(str(exc))
