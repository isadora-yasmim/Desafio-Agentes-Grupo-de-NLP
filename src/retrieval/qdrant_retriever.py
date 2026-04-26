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

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=k
        )

        return [
            {
                "content": r.payload.get("content"),
                "metadata": r.payload
            }
            for r in results
        ]