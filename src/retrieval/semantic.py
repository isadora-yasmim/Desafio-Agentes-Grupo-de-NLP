from langchain_community.vectorstores import SupabaseVectorStore

from core.config import settings
from core.database import get_supabase_client
from ingestion.embedder import get_embeddings


def build_semantic_retriever(k: int | None = None):
    k = k or settings.RETRIEVAL_K_SEMANTIC

    vector_store = SupabaseVectorStore(
        client=get_supabase_client(),
        embedding=get_embeddings(),
        table_name=settings.SUPABASE_TABLE,
        query_name="match_documents",
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )