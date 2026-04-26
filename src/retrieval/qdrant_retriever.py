# retrieval/qdrant_retriever.py

from qdrant_client import QdrantClient

from core.config import settings
from ingestion.embedder import get_embeddings
from retrieval.query_expansion import build_expanded_query


class QdrantRetriever:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = get_embeddings()
        self.collection = settings.QDRANT_COLLECTION

    def search(self, query: str, k: int = 5):
        from retrieval.query_expansion import build_expanded_query

        expanded_query = build_expanded_query(query)
        vector = self.embeddings.embed_query(expanded_query)

        # 🔥 busca mais resultados que o necessário
        fetch_k = k * 4

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=fetch_k,
        )

        results = response.points

        # 🔥 deduplicação por doc_id
        seen_docs = set()
        unique_results = []

        for r in results:
            metadata = r.payload.get("metadata", {})
            doc_id = metadata.get("doc_id")

            if doc_id not in seen_docs:
                seen_docs.add(doc_id)

                unique_results.append({
                    "content": r.payload.get("content"),
                    "metadata": metadata,
                    "score": r.score,
                })

            if len(unique_results) == k:
                break

        return unique_results