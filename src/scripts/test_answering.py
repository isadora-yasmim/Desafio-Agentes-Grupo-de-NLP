from retrieval.qdrant_retriever import QdrantRetriever
from answering import Answerer


def main():
    query = "quais documentos falam sobre tarifa social?"

    retriever = QdrantRetriever()
    answerer = Answerer()

    # 1. Primeiro busca os chunks
    chunks = retriever.search(query)

    # 2. Depois imprime os chunks recuperados
    print("\n" + "=" * 80)
    print("CHUNKS RECUPERADOS")
    print("=" * 80)

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        print(f"\n--- Chunk {i} ---")
        print("Título:", metadata.get("title") or metadata.get("titulo"))
        print("Tipo:", metadata.get("tipo_ato"))
        print("Score:", chunk.get("score"))
        print("Conteúdo:")
        print((chunk.get("content") or "")[:1000])

    # 3. Depois chama o Answering
    print("\n" + "=" * 80)
    print("RESPOSTA FINAL")
    print("=" * 80)

    answer = answerer.answer(query=query, chunks=chunks)

    print(answer)


if __name__ == "__main__":
    main()