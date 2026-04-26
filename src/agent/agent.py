from __future__ import annotations

from answering import Answerer
from retrieval.qdrant_retriever import QdrantRetriever


class RAGAgent:
    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        answerer: Answerer | None = None,
    ):
        self.retriever = retriever or QdrantRetriever()
        self.answerer = answerer or Answerer()

    def run(self, query: str) -> dict:
        if not query or not query.strip():
            return {
                "type": "invalid_query",
                "answer": "Digite uma pergunta válida.",
                "confidence": "baixa",
                "sources": [],
                "used_rag": False,
            }

        normalized_query = query.strip()

        chunks = self.retriever.search(normalized_query)

        response = self.answerer.answer(
            query=normalized_query,
            chunks=chunks,
        )

        return {
            **response,
            "query": normalized_query,
            "chunks_count": len(chunks),
        }