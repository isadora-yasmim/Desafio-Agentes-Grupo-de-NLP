from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ConfidenceDecision:
    level: ConfidenceLevel
    should_answer: bool
    warning: str | None
    final_score: float


HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.45


def get_doc_metadata(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc.get("metadata") or {}

    return getattr(doc, "metadata", {}) or {}


def get_doc_score(doc: Any) -> float:
    """
    Extrai o melhor score disponível do documento/chunk.

    Suporta:
    - dict vindo do retrieval
    - Document do LangChain
    """

    metadata = get_doc_metadata(doc)

    possible_values = []

    if isinstance(doc, dict):
        possible_values.extend(
            [
                doc.get("final_score"),
                doc.get("reranker_score"),
                doc.get("score"),
                doc.get("vector_score"),
                doc.get("semantic_score"),
                doc.get("bm25_score"),
            ]
        )

    possible_values.extend(
        [
            metadata.get("final_score"),
            metadata.get("reranker_score"),
            metadata.get("score"),
            metadata.get("vector_score"),
            metadata.get("semantic_score"),
            metadata.get("bm25_score"),
        ]
    )

    for value in possible_values:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def calculate_final_confidence(docs: list[Any]) -> float:
    if not docs:
        return 0.0

    scores = [get_doc_score(doc) for doc in docs]
    return max(scores, default=0.0)


def decide_confidence(docs: list[Any]) -> ConfidenceDecision:
    final_score = calculate_final_confidence(docs)

    if final_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceDecision(
            level=ConfidenceLevel.HIGH,
            should_answer=True,
            warning=None,
            final_score=final_score,
        )

    if final_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return ConfidenceDecision(
            level=ConfidenceLevel.MEDIUM,
            should_answer=True,
            warning=(
                "⚠️ Encontrei documentos relacionados, mas a evidência não é totalmente forte. "
                "A resposta abaixo deve ser interpretada com cautela."
            ),
            final_score=final_score,
        )

    return ConfidenceDecision(
        level=ConfidenceLevel.LOW,
        should_answer=False,
        warning=(
            "⚠️ Não encontrei evidência documental suficiente para responder com segurança "
            "com base nos documentos disponíveis."
        ),
        final_score=final_score,
    )