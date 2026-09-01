"""RiskLens Streamlit dashboard.

Run (with the API already up):
    streamlit run dashboard/app.py

Configure the API endpoint via RISKLENS_API_URL (defaults to the local API).
"""

from __future__ import annotations

import os

import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient
from views import alerts, anomalies, assistant, overview, portfolio, risk_analytics, stress

DEFAULT_API = os.getenv("RISKLENS_API_URL", "http://localhost:8000/api/v1")

_DEMO_POSITIONS = [
    ("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200),
    ("AMZN", 60, 130), ("GOOGL", 40, 120),
]

PAGES = {
    "Overview": overview.render,
    "Portfolio": portfolio.render,
    "Risk Analytics": risk_analytics.render,
    "Stress Testing": stress.render,
    "Anomalies": anomalies.render,
    "Alerts": alerts.render,
    "AI Assistant": assistant.render,
}


@st.cache_resource
def get_client(base_url: str) -> RiskLensClient:
    return RiskLensClient(base_url)


def _sidebar(client: RiskLensClient) -> int | None:
    st.sidebar.title("📊 RiskLens")
    st.sidebar.caption("Market Risk Analytics · demo data, not financial advice")

    # Health indicator.
    try:
        client.health()
        st.sidebar.success("API connected", icon="✅")
    except RiskLensAPIError:
        st.sidebar.error("API unreachable. Is it running?", icon="🚫")
        return None

    # Portfolio selection.
    try:
        portfolios = client.list_portfolios()
    except RiskLensAPIError as exc:
        st.sidebar.error(str(exc))
        return None

    if portfolios:
        labels = {p["id"]: p["name"] for p in portfolios}
        pid = st.sidebar.selectbox(
            "Portfolio", options=list(labels), format_func=lambda i: labels[i]
        )
    else:
        pid = None
        st.sidebar.info("No portfolios yet.")

    with st.sidebar.expander("Manage data & portfolios"):
        uploaded = st.file_uploader("Upload market-data CSV", type="csv")
        if uploaded is not None and st.button("Ingest CSV"):
            try:
                res = client.upload_csv(uploaded.name, uploaded.getvalue())
                st.success(f"Ingested {res['rows_written']} rows; {res['rows_rejected']} rejected.")
            except RiskLensAPIError as exc:
                st.error(str(exc))

        if st.button("Create demo 'Tech Growth' portfolio"):
            try:
                pf = client.create_portfolio("Tech Growth", "Demo tech book")
                for t, q, ap in _DEMO_POSITIONS:
                    client.add_position(pf["id"], t, q, ap)
                st.success("Created 'Tech Growth'. Select it above.")
                st.rerun()
            except RiskLensAPIError as exc:
                st.error(str(exc))

    return pid


def main() -> None:
    st.set_page_config(page_title="RiskLens", page_icon="📊", layout="wide")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API)
    client = get_client(api_url)

    pid = _sidebar(client)
    page = st.sidebar.radio("Navigate", list(PAGES))

    if pid is None:
        st.title("Welcome to RiskLens")
        st.write(
            "Connect to the API and create a portfolio (see the sidebar) to begin. "
            "Load the sample dataset with `make load-data`, then click "
            "**Create demo 'Tech Growth' portfolio**."
        )
        return

    PAGES[page](client, pid)


if __name__ == "__main__":
    main()
