from retrieval.qdrant_retriever import QdrantRetriever


def print_results(title: str, results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("Nenhum resultado encontrado.")
        return

    for result in results:
        metadata = result["metadata"]

        print(f"Vector score: {result['score']:.4f}")

        if "rerank_score" in result:
            print(f"Rerank score: {result['rerank_score']:.4f}")

        print(f"Tipo do ato: {metadata.get('tipo_ato')}")
        print(f"Título: {metadata.get('titulo')}")
        print(f"Temas: {', '.join(metadata.get('themes', []))}")
        print(f"Chunk type: {metadata.get('chunk_type')}")
        print(f"Data publicação: {metadata.get('data_publicacao')}")
        print(f"Query expandida: {result.get('query_expandida')}")
        print(f"Trecho: {result['content'][:250]}")
        print("-" * 80)


def main() -> None:
    query = "revisão tarifária"

    retriever_sem_rerank = QdrantRetriever(use_reranker=False)
    retriever_com_rerank = QdrantRetriever(use_reranker=True)

    results_sem_rerank = retriever_sem_rerank.search(
        query=query,
        k=5,
        fetch_k=30,
    )

    print_results(
        "Busca Qdrant sem reranking",
        results_sem_rerank,
    )

    results_com_rerank = retriever_com_rerank.search(
        query=query,
        k=5,
        fetch_k=30,
    )

    print_results(
        "Busca Qdrant com reranking",
        results_com_rerank,
    )


if __name__ == "__main__":
    main()