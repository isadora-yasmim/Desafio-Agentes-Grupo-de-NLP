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


def get_doc_score(doc: Any) -> float:
    """
    Extrai o melhor score disponível do documento recuperado.

    Ordem esperada:
    1. final_score
    2. reranker_score
    3. score
    4. vector_score
    """

    metadata = getattr(doc, "metadata", {}) or {}

    for key in ("final_score", "reranker_score", "score", "vector_score"):
        value = metadata.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return 0.0


def calculate_final_confidence(docs: list[Any]) -> float:
    """
    Calcula a confiança final da resposta com base nos documentos recuperados.

    Usa o maior score como sinal principal, porque em RAG geralmente o top documento
    mais relevante é o melhor indicador de confiança.
    """

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