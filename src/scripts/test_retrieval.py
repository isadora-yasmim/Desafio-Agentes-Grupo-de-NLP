from retrieval.semantic import build_semantic_retriever
from retrieval.reranker import get_reranker


def main():
    query = "O que a ANEEL decidiu sobre geração distribuída?"

    retriever = build_semantic_retriever(k=20)
    docs = retriever.invoke(query)

    print(f"\nChunks recuperados: {len(docs)}")

    reranker = get_reranker("bge")
    top_docs = reranker.rerank(query, docs, top_k=5)

    for i, doc in enumerate(top_docs, start=1):
        print("\n" + "=" * 80)
        print(f"RESULTADO {i}")
        print("Score:", doc.metadata.get("rerank_score"))
        print("Título:", doc.metadata.get("titulo"))
        print("Tipo:", doc.metadata.get("tipo_ato"))
        print("Chunk:", doc.metadata.get("chunk_type"))
        print("-" * 80)
        print(doc.page_content[:1000])


if __name__ == "__main__":
    main()