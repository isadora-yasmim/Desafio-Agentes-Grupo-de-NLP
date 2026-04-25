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

import time
import requests
from typing import Literal

from langchain_core.documents import Document

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

EmbeddingBackend = Literal["openai", "huggingface"]


# ---------------------------------------------------------------------------
# Embeddings (mantido igual)
# ---------------------------------------------------------------------------

def get_embeddings(backend: EmbeddingBackend | None = None):
    backend = backend or settings.EMBEDDING_BACKEND

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY,
        )

    elif backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "prompt": "passage: ",
            },
        )

    else:
        raise ValueError(f"Backend desconhecido: {backend}")


# ---------------------------------------------------------------------------
# Embedder usando API REST do Supabase
# ---------------------------------------------------------------------------

class AneelEmbedder:

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        table_name: str = "documents",
        batch_size: int = 10,
    ):
        self.embeddings = get_embeddings(backend)
        self.table_name = table_name
        self.batch_size = batch_size

        self.url = f"{settings.SUPABASE_URL}/rest/v1/{self.table_name}"
        self.headers = {
            "apikey": settings.SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"  # upsert
        }

    # -----------------------------------------------------------------------
    # Upsert em batches (via API)
    # -----------------------------------------------------------------------

    def upsert(self, chunks: list[Document]) -> None:
        total = len(chunks)
        logger.info(f"Iniciando upsert de {total} chunks...")

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]

            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            # gerar embeddings
            embeddings = self.embeddings.embed_documents(texts)

            payload = []
            for text, emb, meta in zip(texts, embeddings, metadatas):
                payload.append({
                    "content": text,
                    "embedding": emb,
                    "metadata": meta
                })

            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers
                )

                if response.status_code not in (200, 201):
                    logger.error(f"Erro API: {response.text}")
                    raise Exception(response.text)

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Erro no batch: {e}")
                raise

        logger.info("Upsert concluído.")

    # -----------------------------------------------------------------------
    # Métodos auxiliares
    # -----------------------------------------------------------------------

    def count_documents(self) -> int:
        url = f"{self.url}?select=id"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(response.text)

        return len(response.json())

    def clear_table(self) -> None:
        logger.warning(f"Limpando tabela '{self.table_name}'...")

        response = requests.delete(
            self.url,
            headers=self.headers
        )

        if response.status_code not in (200, 204):
            raise Exception(response.text)

        logger.info("Tabela limpa.")