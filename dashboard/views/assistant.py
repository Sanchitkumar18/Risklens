"""AI Assistant page (chat shell).

The assistant backend (LangChain tools + LangGraph agent) arrives in Phases 12–13.
This page provides the chat UI now; it calls ``POST /assistant/query`` and degrades
gracefully with a clear message until that endpoint exists.
"""

from __future__ import annotations

import streamlit as st

from api_client import RiskLensAPIError, RiskLensClient

_SUGGESTIONS = [
    "Why is my portfolio risky right now?",
    "Which asset contributes most to portfolio risk?",
    "What are my current risk alerts?",
    "What happens if technology stocks fall 25%?",
    "Explain my 99% VaR in simple terms.",
]


def render(client: RiskLensClient, pid: int) -> None:
    st.header("AI Assistant")
    st.caption(
        "Ask grounded questions about this portfolio's risk. Answers are produced from "
        "the analytical engine — the assistant does not invent numbers."
    )

    key = f"chat_{pid}"
    st.session_state.setdefault(key, [])

    st.write("**Try:** " + " · ".join(f"_{s}_" for s in _SUGGESTIONS))

    for msg in st.session_state[key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about your portfolio risk…")
    if not question:
        return

    st.session_state[key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            resp = client.assistant_query(pid, question)
            answer = resp.get("answer", str(resp))
        except RiskLensAPIError:
            answer = (
                "🛠️ The AI assistant backend is being built (Phases 12–13). "
                "Once available, this chat will answer using real, computed risk metrics "
                "via LangGraph tools. In the meantime, explore the Risk Analytics, Stress "
                "Testing, and Alerts pages."
            )
        st.markdown(answer)
    st.session_state[key].append({"role": "assistant", "content": answer})
