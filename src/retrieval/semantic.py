"""
retrieval/semantic.py
---------------------
Retriever semântico direto ao Supabase via RPC match_documents.

Correções em relação à versão anterior:
  1. Metadata flatten: o Supabase retorna o JSONB como row["metadata"] (dict).
     A versão anterior fazia row.items() e excluía só "content" e "embedding",
     mas os campos úteis (titulo, tipo_ato, etc.) estavam DENTRO de
     row["metadata"] — nunca chegavam ao Document. Resultado: Título: None.

  2. Deduplicação por doc_id: sem isso, o mesmo documento aparecia como
     full_doc + window no mesmo resultado, desperdiçando slots de contexto.

  3. Filtro por chunk_type: por padrão busca apenas "full_doc" para garantir
     contexto completo. Pode ser alterado para buscar em todos os chunks.
"""
from __future__ import annotations

from langchain_core.documents import Document

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class SupabaseSemanticRetriever:
    def __init__(
        self,
        k: int | None = None,
        deduplicate: bool = True,
        chunk_type_filter: str | None = "full_doc",  # None = busca em todos
    ):
        # Lazy imports para não quebrar se as credenciais não estiverem prontas
        from core.database import get_supabase_client
        from ingestion.embedder import get_embeddings

        self.k = k or settings.RETRIEVAL_K_SEMANTIC
        self.deduplicate = deduplicate
        self.chunk_type_filter = chunk_type_filter
        self.client = get_supabase_client()
        self.embeddings = get_embeddings()

    def invoke(self, query: str) -> list[Document]:
        query_embedding = self.embeddings.embed_query(query)

        # Monta o filtro opcional por chunk_type
        rpc_filter: dict = {}
        if self.chunk_type_filter:
            rpc_filter["chunk_type"] = self.chunk_type_filter

        response = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": self.k,
                "filter": rpc_filter,
            },
        ).execute()

        rows = response.data or []
        docs = [self._row_to_document(row) for row in rows]

        if self.deduplicate:
            docs = self._deduplicate(docs)

        logger.debug(f"Semântico: {len(rows)} rows → {len(docs)} docs após dedup")
        return docs

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _row_to_document(self, row: dict) -> Document:
        """
        Constrói o Document com metadata corretamente flattened.

        Estrutura que o Supabase retorna:
          {
            "id": 123,
            "content": "Tipo: DSP | ...",
            "metadata": {               ← JSONB como dict Python
              "titulo": "DSP - ...",
              "tipo_ato": "DSP",
              "doc_id": "2021-03-15__dsp...",
              ...
            },
            "similarity": 0.87
          }

        O metadata do Document deve ser PLANO (chave → valor escalar),
        então fazemos o merge: campos da row + campos do JSONB metadata.
        """
        # Campos do JSONB (onde estão titulo, tipo_ato, etc.)
        jsonb_meta: dict = row.get("metadata") or {}

        # Campos da row que são úteis no metadata do Document
        flat_meta = {
            "similarity":  row.get("similarity"),
            "row_id":      row.get("id"),
        }

        # Merge: JSONB tem prioridade (é a fonte canônica dos metadados)
        merged_meta = {**flat_meta, **jsonb_meta}

        return Document(
            page_content=row.get("content", ""),
            metadata=merged_meta,
        )

    def _deduplicate(self, docs: list[Document]) -> list[Document]:
        """
        Remove documentos repetidos pelo mesmo doc_id,
        mantendo o de maior similarity score.
        Preserva a ordem de ranking.
        """
        seen: set[str] = set()
        result: list[Document] = []

        for doc in docs:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("row_id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(doc)

        return result


def build_semantic_retriever(
    k: int | None = None,
    deduplicate: bool = True,
    chunk_type_filter: str | None = "full_doc",
) -> SupabaseSemanticRetriever:
    return SupabaseSemanticRetriever(
        k=k,
        deduplicate=deduplicate,
        chunk_type_filter=chunk_type_filter,
    )
