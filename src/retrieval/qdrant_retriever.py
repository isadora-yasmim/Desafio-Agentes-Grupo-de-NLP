from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.config import settings
from core.logger import get_logger
from ingestion.embedder import get_embeddings
from retrieval.query_expansion import build_expanded_query
from retrieval.reranker import CrossEncoderReranker


logger = get_logger(__name__)


TECHNICAL_ACRONYMS = {
    "TE",
    "TUSD",
    "TUST",
    "TEO",
    "RTP",
    "PRORET",
    "CDE",
    "ESS",
    "EER",
    "ONS",
    "CCEE",
    "MME",
    "ANEEL",
    "UFV",
    "UHE",
    "PCH",
    "CGH",
    "UTE",
    "EOL",
}


class QdrantRetriever:
    def __init__(self, use_reranker: bool = True):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = get_embeddings()
        self.collection = settings.QDRANT_COLLECTION
        self.use_reranker = use_reranker
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def _build_filter(
        self,
        tipo_ato: str | None = None,
        chunk_type: str | None = None,
        theme: str | None = None,
    ) -> Filter | None:
        conditions = []

        if tipo_ato:
            conditions.append(
                FieldCondition(
                    key="tipo_ato",
                    match=MatchValue(value=tipo_ato),
                )
            )

        if chunk_type:
            conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type),
                )
            )

        if theme:
            conditions.append(
                FieldCondition(
                    key="themes",
                    match=MatchValue(value=theme),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def _should_use_reranker(self, query: str) -> bool:
        normalized_query = query.strip()
        tokens = normalized_query.split()
        upper_tokens = {token.upper().strip(".,;:()[]{}") for token in tokens}

        if not self.use_reranker:
            return False

        if not self.reranker:
            return False

        if upper_tokens.intersection(TECHNICAL_ACRONYMS):
            return False

        return True

    def _deduplicate_points(
        self,
        points,
        query: str,
        expanded_query: str,
    ) -> list[dict]:
        seen_docs = set()
        unique_results = []

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            doc_id = metadata.get("doc_id") or point.id

            if doc_id in seen_docs:
                continue

            seen_docs.add(doc_id)

            unique_results.append(
                {
                    "content": payload.get("content"),
                    "metadata": metadata,
                    "score": point.score,
                    "query_original": query,
                    "query_expandida": expanded_query,
                }
            )

        return unique_results

    def _apply_reranking_with_fallback(
        self,
        query: str,
        candidates: list[dict],
        k: int,
    ) -> list[dict]:
        if not self._should_use_reranker(query):
            return candidates[:k]

        try:
            reranked = self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=k,
            )
        except Exception as error:
            logger.warning("Falha ao aplicar reranking. Usando ranking original: %s", error)
            return candidates[:k]

        if not reranked:
            return candidates[:k]

        best_rerank_score = reranked[0].get("rerank_score")

        if best_rerank_score is None or best_rerank_score < 0:
            return candidates[:k]

        return reranked[:k]

    def search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int | None = None,
        tipo_ato: str | None = None,
        chunk_type: str | None = None,
        theme: str | None = None,
    ) -> list[dict]:
        expanded_query = build_expanded_query(query)

        logger.debug("Query original: %s", query)
        logger.debug("Query expandida: %s", expanded_query)

        vector = self.embeddings.embed_query(expanded_query)

        fetch_k = fetch_k or k * 10

        query_filter = self._build_filter(
            tipo_ato=tipo_ato,
            chunk_type=chunk_type,
            theme=theme,
        )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True,
        )

        candidates = self._deduplicate_points(
            points=response.points,
            query=query,
            expanded_query=expanded_query,
        )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("content") and len(candidate.get("content")) > 100
        ]

        logger.debug("Total de candidatos após deduplicação e filtro: %s", len(candidates))

        return self._apply_reranking_with_fallback(
            query=query,
            candidates=candidates,
            k=k,
        )