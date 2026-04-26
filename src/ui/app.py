import streamlit as st

from retrieval.qdrant_retriever import QdrantRetriever


st.set_page_config(
    page_title="RAG Domínio Elétrico",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ RAG para Domínio Elétrico")
st.caption("Busca semântica em documentos regulatórios do setor elétrico.")

query = st.text_input(
    "Digite sua pergunta:",
    placeholder="Ex: Quais documentos falam sobre tarifas de energia elétrica?"
)

k = st.slider("Quantidade de chunks recuperados", min_value=3, max_value=10, value=5)

if "retriever" not in st.session_state:
    st.session_state.retriever = QdrantRetriever()

if st.button("Buscar", type="primary"):
    if not query.strip():
        st.warning("Digite uma pergunta antes de buscar.")
    else:
        with st.spinner("Buscando documentos relevantes..."):
            results = st.session_state.retriever.search(query, k=k)

        st.subheader("Resultados recuperados")

        for i, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            content = result.get("content", "")

            with st.expander(f"Chunk {i}", expanded=i == 1):
                st.markdown("### Conteúdo")
                st.write(content)

                st.markdown("### Metadados")
                st.json(metadata)