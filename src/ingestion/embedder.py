"""
embedder.py
-----------
Responsável por gerar embeddings dos chunks e fazer upsert no Qdrant.

O Supabase deixa de ser usado como vector store.
O Qdrant armazena:
  - content
  - embedding
  - metadata
"""

import time
import uuid
from typing import Literal

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

EmbeddingBackend = Literal["openai", "huggingface"]


# ---------------------------------------------------------------------------
# Embeddings
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
# Embedder usando Qdrant
# ---------------------------------------------------------------------------

class AneelEmbedder:
    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        collection_name: str | None = None,
        batch_size: int = 32,
    ):
        self.embeddings = get_embeddings(backend)
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.batch_size = batch_size

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=getattr(settings, "QDRANT_API_KEY", None),
        )

        self.vector_size = self._get_vector_size()

        self._ensure_collection()

    # -----------------------------------------------------------------------
    # Descobre dimensão do embedding
    # -----------------------------------------------------------------------

    def _get_vector_size(self) -> int:
        test_vector = self.embeddings.embed_query("teste")
        return len(test_vector)

    # -----------------------------------------------------------------------
    # Cria collection se não existir
    # -----------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            logger.info(
                f"Criando collection '{self.collection_name}' "
                f"com dimensão {self.vector_size}..."
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info("Collection criada com sucesso.")
        else:
            logger.info(f"Collection '{self.collection_name}' já existe.")

    # -----------------------------------------------------------------------
    # Upsert em batches
    # -----------------------------------------------------------------------

    def upsert(self, chunks: list[Document]) -> None:
        total = len(chunks)
        logger.info(f"Iniciando upsert de {total} chunks no Qdrant...")

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]

            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            vectors = self.embeddings.embed_documents(texts)

            points = []

            for text, vector, metadata in zip(texts, vectors, metadatas):
                point_id = str(uuid.uuid4())

                payload = {
                    "content": text,
                    "metadata": metadata,
                    **metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

                logger.info(
                    f"Batch {i // self.batch_size + 1} enviado "
                    f"({min(i + self.batch_size, total)}/{total})"
                )

                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Erro no batch {i // self.batch_size + 1}: {e}")
                raise

        logger.info("Upsert no Qdrant concluído.")

    # -----------------------------------------------------------------------
    # Métodos auxiliares
    # -----------------------------------------------------------------------

    def count_documents(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def clear_table(self) -> None:
        logger.warning(f"Limpando collection '{self.collection_name}'...")

        self.client.delete_collection(
            collection_name=self.collection_name,
        )

        self._ensure_collection()

        logger.info("Collection limpa e recriada.")