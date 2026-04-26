"""
retrieval/hybrid.py
-------------------
EnsembleRetriever com query expansion integrada.

Fluxo:
  Query do usuário
    ↓
  QueryExpansion → detecta sinônimos do domínio elétrico
    ↓                 ex: "tarifa de energia" → ["TE", "TUSD", ...]
    ├── HyDE: usa a query enriquecida para gerar documento hipotético
    │         com terminologia técnica correta
    │
    ├── Semântico (Supabase): usa o documento hipotético como embedding
    │
    └── BM25 multi-query: roda BM25 para query original + cada sinônimo,
                          funde por RRF
    ↓
  Deduplicação por doc_id
    ↓
  Reranker (cross-encoder BGE) → top-K
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_community.retrievers import BM25Retriever

from core.config import settings
from core.logger import get_logger
from retrieval.semantic import SupabaseSemanticRetriever
from retrieval.query_expansion import (
    build_expanded_query,
    bm25_multi_query_retrieve,
    reciprocal_rank_fusion,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def build_bm25_retriever(docs: list[Document], k: int | None = None) -> BM25Retriever:
    k = k or settings.RETRIEVAL_K_BM25
    retriever = BM25Retriever.from_documents(docs, k=k)
    logger.info(f"BM25Retriever: {len(docs)} documentos indexados.")
    return retriever


# ---------------------------------------------------------------------------
# Retriever híbrido com expansão
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combina semântico + BM25 multi-query com query expansion e deduplicação.
    """

    def __init__(
        self,
        all_chunks: list[Document],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        assert abs(semantic_weight + bm25_weight - 1.0) < 1e-6

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.semantic = SupabaseSemanticRetriever(
            k=settings.RETRIEVAL_K_SEMANTIC,
            deduplicate=False,
            chunk_type_filter="full_doc",
        )
        self.bm25 = build_bm25_retriever(all_chunks, k=settings.RETRIEVAL_K_BM25)

        logger.info(
            f"HybridRetriever: semântico={semantic_weight}, BM25={bm25_weight}"
        )

    def invoke(self, query: str) -> list[Document]:
        bm25_docs = bm25_multi_query_retrieve(
            self.bm25, query, k_per_query=settings.RETRIEVAL_K_BM25
        )
        semantic_docs = self.semantic.invoke(query)

        fused = reciprocal_rank_fusion(
            [semantic_docs] * round(self.semantic_weight * 10)
            + [bm25_docs] * round(self.bm25_weight * 10)
        )

        deduped = self._deduplicate(fused)
        logger.debug(
            f"Híbrido: semântico={len(semantic_docs)}, "
            f"BM25={len(bm25_docs)}, "
            f"após fusão+dedup={len(deduped)}"
        )
        return deduped

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)

    def _deduplicate(self, docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        result: list[Document] = []
        for doc in docs:
            key = doc.metadata.get("doc_id") or doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)
        return result


# ---------------------------------------------------------------------------
# HyDE com query expansion
# ---------------------------------------------------------------------------

HYDE_PROMPT = """Você é um especialista em regulação do setor elétrico brasileiro (ANEEL).
Dado a pergunta abaixo, escreva um trecho de documento regulatório que responderia diretamente a ela.
Escreva como se fosse uma ementa ou despacho real da ANEEL. Seja técnico e conciso (3-5 linhas).
Use a terminologia exata do setor: siglas (TE, TUSD, TUST, ANEEL, ONS), tipos de ato (DSP, REH, REN, PRT).

Termos técnicos relacionados à pergunta: {expanded_terms}

Pergunta: {question}

Trecho hipotético:"""


class HyDERetriever:
    """
    Aplica HyDE com query expansion antes de chamar o retriever base.

    - O documento hipotético usa a terminologia técnica expandida,
      melhorando o embedding para a busca semântica.
    - O BM25 usa multi-query com os sinônimos expandidos diretamente,
      sem passar pelo LLM.
    """

    def __init__(self, base_retriever: HybridRetriever, llm: BaseLanguageModel):
        self.base_retriever = base_retriever
        self.llm = llm

    def invoke(self, query: str) -> list[Document]:
        expanded = build_expanded_query(query)
        hyde_doc = self._generate_hypothetical(query, expanded)
        logger.debug(f"HyDE (expandido): {hyde_doc[:200]}")

        # Semântico usa o doc hipotético; BM25 usa a query original + sinônimos
        semantic_docs = self.base_retriever.semantic.invoke(hyde_doc)
        bm25_docs = bm25_multi_query_retrieve(
            self.base_retriever.bm25, query, k_per_query=settings.RETRIEVAL_K_BM25
        )

        fused = reciprocal_rank_fusion(
            [semantic_docs] * round(self.base_retriever.semantic_weight * 10)
            + [bm25_docs] * round(self.base_retriever.bm25_weight * 10)
        )
        return self.base_retriever._deduplicate(fused)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)

    def _generate_hypothetical(self, question: str, expanded_query: str) -> str:
        expanded_terms = expanded_query.replace(question, "").strip(" []")
        if not expanded_terms:
            expanded_terms = "terminologia técnica do setor elétrico brasileiro"

        prompt = HYDE_PROMPT.format(
            question=question,
            expanded_terms=expanded_terms,
        )
        response = self.llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
