# retrieval/qdrant_retriever.py

from qdrant_client import QdrantClient
from core.config import settings
from ingestion.embedder import get_embeddings


class QdrantRetriever:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = get_embeddings()
        self.collection = settings.QDRANT_COLLECTION

    def search(self, query: str, k: int = 5):
        vector = self.embeddings.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
        )

        results = response.points

        return [
            {
                "content": r.payload.get("content"),
                "metadata": r.payload.get("metadata", {}),
                "score": r.score,
            }
            for r in results
        ]