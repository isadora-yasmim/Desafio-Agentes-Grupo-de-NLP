from retrieval.qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever()

results = retriever.search("tarifa de energia elétrica")

for r in results:
    print(r["content"][:200])