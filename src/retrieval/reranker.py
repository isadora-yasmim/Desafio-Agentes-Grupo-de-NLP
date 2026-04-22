"""
reranker.py
-----------
Reranking dos chunks recuperados usando cross-encoder BGE (gratuito).

Por que reranking?
- O retriever busca os 20 chunks mais prováveis (por similaridade ou BM25).
- Mas "provável" ≠ "relevante". O cross-encoder avalia cada (query, chunk)
  como par, produzindo scores muito mais precisos que o cosine similarity.
- Selecionar os top-5 após reranking melhora significativamente o RAG.

Modelos recomendados (gratuitos, rodando local):
  - BAAI/bge-reranker-base    (~280MB, rápido)
  - BAAI/bge-reranker-large   (~560MB, melhor qualidade)

Alternativa paga:
  - Cohere Rerank API (excelente, mas requer API key)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

RerankerBackend = Literal["bge", "cohere"]


# ---------------------------------------------------------------------------
# BGE Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class BGEReranker:
    """
    Reranker usando cross-encoder da família BGE (HuggingFace, gratuito).

    Uso:
        reranker = BGEReranker()
        top_docs = reranker.rerank(query, candidate_docs, top_k=5)
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy loading do modelo (só carrega quando necessário)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Carregando reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker carregado.")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Reordena documentos por relevância e retorna os top_k.

        Args:
            query: Pergunta do usuário.
            documents: Candidatos do EnsembleRetriever (20 chunks).
            top_k: Quantos retornar após reranking.

        Returns:
            Lista ordenada por score decrescente (mais relevante primeiro).
        """
        top_k = top_k or settings.RETRIEVAL_K_FINAL

        if not documents:
            return []

        # Prepara pares (query, conteúdo_do_chunk)
        pairs = [(query, doc.page_content) for doc in documents]

        # Score de cada par — quanto maior, mais relevante
        scores = self.model.predict(pairs)

        # Associa scores e ordena
        scored = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        # Injeta o score nos metadados para transparência na UI
        results = []
        for rank, (score, doc) in enumerate(scored[:top_k]):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_position"] = rank + 1
            results.append(doc)

        logger.debug(
            f"Reranking: {len(documents)} → {len(results)} chunks | "
            f"top score: {float(scored[0][0]):.4f}"
        )
        return results


# ---------------------------------------------------------------------------
# Cohere Reranker (alternativa)
# ---------------------------------------------------------------------------

class CohereReranker:
    """
    Reranker usando a API do Cohere (pago, mas excelente qualidade).
    Requer COHERE_API_KEY no .env.
    """

    def __init__(self, model: str = "rerank-multilingual-v3.0"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import cohere
            from src.core.config import settings
            self._client = cohere.Client(settings.COHERE_API_KEY)
        return self._client

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        top_k = top_k or settings.RETRIEVAL_K_FINAL

        texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_n=top_k,
        )

        results = []
        for result in response.results:
            doc = documents[result.index]
            doc.metadata["rerank_score"] = result.relevance_score
            doc.metadata["rerank_position"] = result.index + 1
            results.append(doc)

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reranker(backend: RerankerBackend = "bge") -> BGEReranker | CohereReranker:
    if backend == "bge":
        return BGEReranker()
    elif backend == "cohere":
        return CohereReranker()
    raise ValueError(f"Backend desconhecido: {backend}")
