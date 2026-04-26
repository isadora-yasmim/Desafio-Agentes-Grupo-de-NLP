from retrieval.qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever()

results = retriever.search("tarifa de energia elétrica")

for result in results:
    metadata = result["metadata"]

    print(f"Score: {result['score']:.4f}")
    print(f"Tipo do ato: {metadata.get('tipo_ato')}")
    print(f"Título: {metadata.get('titulo')}")
    print(f"Temas: {', '.join(metadata.get('themes', []))}")
    print(f"Chunk type: {metadata.get('chunk_type')}")
    print(f"Data publicação: {metadata.get('data_publicacao')}")
    print("-" * 80)