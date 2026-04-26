from agent import RAGAgent


def main():
    query = "quais documentos falam sobre tarifa social?"

    agent = RAGAgent()

    response = agent.run(query)

    print("\n" + "=" * 80)
    print("RESPOSTA FINAL")
    print("=" * 80)

    print(response["answer"])

    print("\n" + "=" * 80)
    print("METADADOS")
    print("=" * 80)

    print("Query:", response["query"])
    print("Tipo:", response["type"])
    print("Confiança:", response["confidence"])
    print("Usou RAG:", response["used_rag"])
    print("Chunks recuperados:", response["chunks_count"])

    if response.get("sources"):
        print("\nFontes:")
        for source in response["sources"]:
            print(
                f"- {source.get('type')} — {source.get('title')} "
                f"(score: {source.get('score')})"
            )


if __name__ == "__main__":
    main()