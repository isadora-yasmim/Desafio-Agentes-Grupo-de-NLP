from retrieval.qdrant_retriever import QdrantRetriever
from answering import Answerer


def main():
    query = "quais documentos falam sobre tarifa social?"

    retriever = QdrantRetriever()
    answerer = Answerer()

    chunks = retriever.search(query)

    print("\n" + "=" * 80)
    print("CHUNKS RECUPERADOS")
    print("=" * 80)

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        print(f"\n--- Chunk {i} ---")
        print("Título:", metadata.get("title") or metadata.get("titulo"))
        print("Tipo:", metadata.get("tipo_ato") or metadata.get("type"))
        print("Score:", chunk.get("score"))
        print("Conteúdo:")
        print((chunk.get("content") or "")[:1000])

    print("\n" + "=" * 80)
    print("RESPOSTA FINAL")
    print("=" * 80)

    answer = answerer.answer(query=query, chunks=chunks)

    print(answer["answer"])

    print("\n" + "=" * 80)
    print("METADADOS DA RESPOSTA")
    print("=" * 80)
    print("Tipo:", answer["type"])
    print("Confiança:", answer["confidence"])
    print("Usou RAG:", answer["used_rag"])

    if answer.get("sources"):
        print("\nFontes:")
        for source in answer["sources"]:
            print(
                f"- {source.get('type')} — {source.get('title')} "
                f"(score: {source.get('score')})"
            )


if __name__ == "__main__":
    main()