from retrieval.qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever()

results = retriever.search("tarifa de energia elétrica")

for r in results:
    print(r["content"][:200])

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Tipo do ato: {result['metadata'].get('tipo_ato')}")
    print(f"Título: {result['metadata'].get('titulo')}")
    print(f"Temas inferidos: {result['metadata'].get('temas_inferidos')}")
    print("-" * 80)