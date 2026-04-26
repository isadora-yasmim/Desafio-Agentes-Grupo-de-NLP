import streamlit as st

query = st.text_input("Faça sua pergunta")

if query:
    response = agent.run(query)

    st.write(response["answer"])

    st.subheader("📚 Fontes")
    for s in response["sources"]:
        st.write(f"{s['type']} — {s['title']}")

    st.write(f"Confiança: {response['confidence']}")