"""
embedder.py
-----------
Responsável por gerar embeddings dos chunks e fazer upsert no Supabase.

Suporta dois modelos de embedding (configurável via settings):
  - text-embedding-3-small (OpenAI) — qualidade alta, rápido, barato
  - multilingual-e5-base (HuggingFace, gratuito) — bom para PT-BR

O Supabase usa pgvector para armazenar os vetores.
A tabela esperada (ver setup_supabase.sql) é `documents`.
"""

from __future__ import annotations

import time
from typing import Literal

from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

EmbeddingBackend = Literal["openai", "huggingface"]


# ---------------------------------------------------------------------------
# Factory de embeddings
# ---------------------------------------------------------------------------

def get_embeddings(backend: EmbeddingBackend | None = None):
    """
    Retorna o objeto de embeddings configurado.
    Se backend=None, usa o valor de settings.EMBEDDING_BACKEND.
    """
    backend = backend or settings.EMBEDDING_BACKEND

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Usando embeddings: OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY,
        )

    elif backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Usando embeddings: multilingual-e5-base (HuggingFace)")
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                # O modelo E5 requer prefixo "query:" ou "passage:"
                # Para indexação (passage):
                "prompt": "passage: ",
            },
        )

    else:
        raise ValueError(f"Backend desconhecido: {backend}")


# ---------------------------------------------------------------------------
# Cliente Supabase
# ---------------------------------------------------------------------------

def get_supabase_client():
    """Retorna cliente Supabase autenticado."""
    from supabase import create_client
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Embedder / Upserter
# ---------------------------------------------------------------------------

class AneelEmbedder:
    """
    Gera embeddings e faz upsert dos chunks no Supabase.

    Uso:
        embedder = AneelEmbedder()
        embedder.upsert(chunks)
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        table_name: str = "documents",
        batch_size: int = 100,
    ):
        self.embeddings = get_embeddings(backend)
        self.table_name = table_name
        self.batch_size = batch_size
        self._client = None
        self._vector_store = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    @property
    def vector_store(self) -> SupabaseVectorStore:
        if self._vector_store is None:
            self._vector_store = SupabaseVectorStore(
                client=self.client,
                embedding=self.embeddings,
                table_name=self.table_name,
                query_name="match_documents",  # nome da função RPC no Supabase
            )
        return self._vector_store

    # -----------------------------------------------------------------------
    # Upsert em batches
    # -----------------------------------------------------------------------

    def upsert(self, chunks: list[Document]) -> None:
        """
        Insere ou atualiza chunks no vector store.
        Processa em batches para evitar timeout e rate limit.
        """
        total = len(chunks)
        logger.info(f"Iniciando upsert de {total} chunks em batches de {self.batch_size}...")

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                self.vector_store.add_documents(batch)
                # Pequena pausa para não estourar rate limit da OpenAI
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"  Erro no batch {batch_num}: {e}")
                raise

        logger.info(f"Upsert concluído: {total} chunks inseridos.")

    def upsert_texts(self, texts: list[str], metadatas: list[dict]) -> None:
        """Alternativa: upsert direto por textos + metadatas."""
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    # -----------------------------------------------------------------------
    # Verificação
    # -----------------------------------------------------------------------

    def count_documents(self) -> int:
        """Retorna o número de documentos na tabela."""
        result = self.client.table(self.table_name).select("id", count="exact").execute()
        return result.count or 0

    def clear_table(self) -> None:
        """Remove todos os registros da tabela (use com cuidado!)."""
        logger.warning(f"Limpando tabela '{self.table_name}'...")
        self.client.table(self.table_name).delete().neq("id", 0).execute()
        logger.info("Tabela limpa.")
