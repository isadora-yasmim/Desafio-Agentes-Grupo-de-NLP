from __future__ import annotations

from typing import Any
import streamlit as st

from agent.agent import build_agent


# =========================
# 🎨 ESTILO (dark minimal)
# =========================
CUSTOM_CSS = """
<style>
:root {
    --bg-main: #1e1e1b;
    --bg-sidebar: #161615;
    --bg-card: #2a2a26;
    --border-soft: #3a3a34;
    --text-main: #e8e3d8;
    --text-muted: #a9a297;
    --accent: #f59e0b;
}

.stApp {
    background: var(--bg-main);
    color: var(--text-main);
}

.main .block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 8rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-soft);
}

/* =========================
   🧠 HEADER
========================= */
.app-title {
    text-align: center;
    margin-top: 4rem;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.app-title::before {
    content: "⚡";
    display: block;
    font-size: 2rem;
    margin-bottom: 0.8rem;
    color: var(--accent);
}

.app-subtitle {
    text-align: center;
    color: var(--text-muted);
    margin-bottom: 2.5rem;
}

/* =========================
   💬 MENSAGENS
========================= */
.message-card {
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    background: var(--bg-card);
}

.user-card {
    background: #32322c;
}

.role-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

/* =========================
   📄 SOURCES
========================= */
.source-card {
    background: #232321;
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 0.8rem;
    margin-top: 0.6rem;
}

.source-title {
    font-weight: 600;
}

.source-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* =========================
   INPUT
========================= */
div[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    border-radius: 20px !important;
    border: 1px solid var(--border-soft) !important;
}
</style>
"""


# =========================
# 🔧 STATE
# =========================
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = build_agent()


# =========================
# ⚙️ SIDEBAR
# =========================
def sidebar():
    with st.sidebar:
        st.title("⚡ RAG Elétrico")

        top_k = st.slider("Documentos", 3, 20, 5)
        rerank = st.toggle("Reranker", False)
        debug = st.toggle("Debug", False)

        if st.button("Limpar"):
            st.session_state.messages = []
            st.rerun()

    return {"top_k": top_k, "rerank": rerank, "debug": debug}


# =========================
# 🧾 SOURCES
# =========================
def render_sources(sources):
    if not sources:
        return

    with st.expander("Fontes"):
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata", {})
            title = meta.get("title") or meta.get("titulo") or "Documento"

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{i}. {title}</div>
                    <div class="source-meta">{meta.get("tipo","")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================
# 💬 CHAT
# =========================
def render_chat():
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="app-title">RAG Domínio Elétrico</div>
            <div class="app-subtitle">
                Consulte documentos regulatórios do setor elétrico com respostas baseadas em evidências.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="message-card user-card">
                    <div class="role-label">Você</div>
                    <div>{msg["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.container():
                st.markdown("**Assistente**")
                st.markdown(msg["content"])
                render_sources(msg.get("sources"))
                
# =========================
# 🚀 MAIN
# =========================
def main():
    st.set_page_config(layout="wide")

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session()
    config = sidebar()

    render_chat()

    question = st.chat_input("Pergunte algo...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("Buscando..."):
            result = st.session_state.agent.invoke(
                {
                    "question": question,
                    "top_k": config["top_k"],
                    "use_reranker": config["rerank"],
                }
            )

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

        st.rerun()


if __name__ == "__main__":
    main()