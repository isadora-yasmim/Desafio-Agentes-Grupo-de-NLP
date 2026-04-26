from langchain_core.documents import Document

from core.config import settings
from core.database import get_supabase_client
from ingestion.embedder import get_embeddings


class SupabaseSemanticRetriever:
    def __init__(self, k: int = 20):
        self.k = k
        self.client = get_supabase_client()
        self.embeddings = get_embeddings()

    def invoke(self, query: str) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)

        response = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": self.k,
            },
        ).execute()

        rows = response.data or []

        return [
            Document(
                page_content=row.get("content", ""),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"content", "embedding"}
                },
            )
            for row in rows
        ]


def build_semantic_retriever(k: int = 20):
    return SupabaseSemanticRetriever(k=k)