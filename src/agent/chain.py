"""
chain.py
--------
Chain principal do RAG: junta tudo.

Fluxo:
  Pergunta do usuário
      ↓
  HyDE (documento hipotético)
      ↓
  EnsembleRetriever (semântico + BM25) → 20 chunks
      ↓
  BGE Reranker → top 5 chunks
      ↓
  Prompt + LLM (GPT-4o-mini)
      ↓
  Resposta com fontes
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.retrieval.hybrid import HyDERetriever, build_ensemble_retriever
from src.retrieval.reranker import get_reranker
from src.agent.prompts import RAG_PROMPT
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def format_docs(docs: list[Document]) -> str:
    """Formata os chunks para o prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        score = meta.get("rerank_score", "N/A")
        header = (
            f"[Fonte {i}] {meta.get('titulo', 'N/A')} | "
            f"Tipo: {meta.get('tipo_ato')} | "
            f"Publicação: {meta.get('data_publicacao')} | "
            f"Score: {score:.3f}" if isinstance(score, float) else
            f"[Fonte {i}] {meta.get('titulo', 'N/A')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class AneelRAGChain:
    """
    Chain RAG completa para o domínio ANEEL.

    Uso:
        chain = AneelRAGChain(all_chunks=chunks)
        result = chain.invoke("Quais PCHs foram registradas em GO em 2016?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(self, all_chunks: list[Document]):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.reranker = get_reranker("bge")

        # Ensemble retriever (precisa dos chunks p/ BM25)
        ensemble = build_ensemble_retriever(all_chunks)

        # Wrapa com HyDE
        self.retriever = HyDERetriever(
            base_retriever=ensemble,
            llm=self.llm,
        )

    def invoke(self, question: str) -> dict:
        """
        Executa o pipeline RAG completo.

        Returns:
            {
                "answer": str,
                "sources": list[Document],
                "context": str,
            }
        """
        # 1. Retrieval (HyDE + Ensemble)
        candidates = self.retriever.invoke(question)
        logger.info(f"Candidatos recuperados: {len(candidates)}")

        # 2. Reranking
        top_docs = self.reranker.rerank(
            question,
            candidates,
            top_k=settings.RETRIEVAL_K_FINAL,
        )
        logger.info(f"Após reranking: {len(top_docs)} chunks")

        # 3. Geração
        context = format_docs(top_docs)
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": top_docs,
            "context": context,
        }
