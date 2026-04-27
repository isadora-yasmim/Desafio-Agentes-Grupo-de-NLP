from __future__ import annotations

import base64
import os
from typing import Any
import streamlit as st

from agent.agent import build_agent


# =========================
# 🎨 ESTILO (responsivo Light/Dark)
# =========================
CUSTOM_CSS = """
<style>
.main .block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 8rem;
}

/* =========================
   🧠 LOGO E HEADER
========================= */
.logo-container {
    display: flex;
    justify-content: center;
    margin-bottom: 0.8rem;
}

.sidebar-logo-container {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1rem;
}

.logo-img {
    width: 80px; /* Ajuste o tamanho da logo principal */
}

.sidebar-logo-img {
    width: 50px; /* Ajuste o tamanho da logo na barra lateral */
}

.app-title {
    text-align: center;
    margin-top: 0.5rem;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.app-subtitle {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 2.5rem;
}

/* =========================
   💬 MENSAGENS
========================= */
.message-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    background: var(--secondary-background-color);
}

.user-card {
    background: rgba(128, 128, 128, 0.1);
}

.role-label {
    font-size: 0.8rem;
    opacity: 0.7;
    margin-bottom: 0.4rem;
}

/* =========================
   📄 SOURCES
========================= */
.source-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 0.8rem;
    margin-top: 0.6rem;
}

.source-title {
    font-weight: 600;
}

.source-meta {
    font-size: 0.8rem;
    opacity: 0.7;
}

/* =========================
   INPUT
========================= */
div[data-testid="stChatInput"] textarea {
    border-radius: 20px !important;
}
</style>
"""


# =========================
# 🖼️ IMAGE HELPER
# =========================
def get_image_base64(file_path):
    """Converte a imagem para base64 para poder injetar no HTML."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

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
        # Repetindo a logo amarela na lateral
        img_base64 = get_image_base64("assets/logo.png")
        if img_base64:
            st.markdown(
                f'<div class="sidebar-logo-container"><img src="data:image/png;base64,{img_base64}" class="sidebar-logo-img"></div>',
                unsafe_allow_html=True
            )
        st.title("LexEletro")

        top_k = st.slider("Documentos", 3, 20, 5)
        rerank = st.toggle("Refinar Resultados", False)
        debug = st.toggle("Detalhes do Processamento", False)

        if st.button("Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    return {"top_k": top_k, "rerank": rerank, "debug": debug}


# =========================
# 🧾 SOURCES
# =========================
def render_sources(sources):
    if not sources:
        return

    with st.expander("Fontes Consultadas"):
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata") or {}

            title = (
                meta.get("titulo")
                or meta.get("title")
                or s.get("titulo")
                or s.get("title")
                or "Documento"
            )

            tipo = (
                meta.get("tipo_ato")
                or meta.get("tipo")
                or s.get("tipo")
                or ""
            )

            score = (
                s.get("final_score")
                or s.get("score")
                or meta.get("score")
            )

            score_text = f"Score: {score:.4f}" if isinstance(score, float) else ""

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{i}. {title}</div>
                    <div class="source-meta">
                        {tipo}{f" • {score_text}" if score_text else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================
# 💬 CHAT
# =========================
def render_chat():
    if not st.session_state.messages:
        img_base64 = get_image_base64("assets/logo.png")
        logo_html = f'<div class="logo-container"><img src="data:image/png;base64,{img_base64}" class="logo-img"></div>' if img_base64 else ""

        st.markdown(
            f"""
            {logo_html}
            <div class="app-title">LexEletro</div>
            <div class="app-subtitle">
                Consulte documentos regulatórios com inteligência.
                <br><br>
                <div style="background-color: var(--secondary-background-color); padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); font-size: 0.95rem; text-align: left; max-width: 650px; margin: 0 auto; line-height: 1.6;">
                    🚀 <strong>Sobre este projeto:</strong><br>
                    Esta aplicação é um sistema de <strong>RAG (Retrieval-Augmented Generation)</strong> desenvolvido exclusivamente para fins educativos. 
                    Seu objetivo é facilitar a consulta e análise de documentos do setor elétrico brasileiro.
                    <br><br>
                    O sistema utiliza busca semântica para encontrar os trechos mais relevantes e uma IA avançada para sintetizar a resposta baseada em evidências. 
                    <em>Lembre-se: as informações são geradas automaticamente para fins didáticos e devem ser validadas nas fontes oficiais.</em>
                </div>
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
    st.set_page_config(
        page_title="LexEletro - RAG",
        page_icon="⚡",
        layout="wide"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session()
    config = sidebar()

    render_chat()

    # Mudança no nome da barra de pesquisa (placeholder)
    question = st.chat_input("Digite sua dúvida sobre a regulação do setor elétrico...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("Analisando documentos..."):
            result = st.session_state.agent.invoke(
                {
                    "question": question,
                    "top_k": config["top_k"],
                    "use_reranker": config["rerank"],
                }
            )

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        st.write("DEBUG SOURCES:", sources)

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