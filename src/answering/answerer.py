from __future__ import annotations

from answering.llm import build_llm
from answering.prompt import build_answer_prompt


CONCEPTUAL_KEYWORDS = [
    "como funciona",
    "o que é",
    "explique",
    "definição",
    "conceito",
]


DOCUMENT_LISTING_KEYWORDS = [
    "quais documentos",
    "liste documentos",
    "documentos falam",
    "documentos sobre",
    "quais atos",
    "atos sobre",
]


def is_conceptual_query(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in CONCEPTUAL_KEYWORDS)


def is_document_listing_query(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DOCUMENT_LISTING_KEYWORDS)


def is_valid_rag_response(response: str) -> bool:
    if not response:
        return False

    invalid_signals = [
        "não foi possível responder",
        "não contém informações",
        "não há informação suficiente",
        "não encontrei documentos",
    ]

    response_lower = response.lower()

    return not any(signal in response_lower for signal in invalid_signals)


def extract_sources(chunks: list[dict]) -> list[dict]:
    sources = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}

        sources.append(
            {
                "title": (
                    metadata.get("title")
                    or metadata.get("titulo")
                    or metadata.get("document_title")
                ),
                "type": metadata.get("tipo_ato") or metadata.get("type"),
                "source": metadata.get("source") or metadata.get("fonte"),
                "score": chunk.get("score"),
            }
        )

    return sources


def format_document_listing(chunks: list[dict]) -> str:
    documents = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}

        title = (
            metadata.get("title")
            or metadata.get("titulo")
            or metadata.get("document_title")
            or "Documento sem título"
        )

        doc_type = (
            metadata.get("tipo_ato")
            or metadata.get("type")
            or metadata.get("tipo")
            or "Tipo não informado"
        )

        content = (chunk.get("content") or "").strip()
        score = chunk.get("score")

        score_text = (
            f"{score:.4f}" if isinstance(score, (int, float)) else "não informado"
        )

        evidence = content if content else "Sem evidência textual disponível."

        documents.append(
            f"- **{doc_type} — {title}**\n"
            f"  - Evidência de relação: {evidence}\n"
            f"  - Score de recuperação: {score_text}"
        )

    return "Encontrei os seguintes documentos relacionados:\n\n" + "\n\n".join(documents)


class Answerer:
    def __init__(self):
        self.llm = build_llm()

    def _conceptual_answer(self, query: str) -> str:
        return self.llm.invoke(
            f"""
Explique de forma clara e objetiva:

{query}

Formato:

🔎 Resumo:
...

⚡ Componentes:
- ...
- ...

📌 Observação:
...

Regras:
- Máximo de 6 frases
- Linguagem técnica, mas direta
- Evite parágrafos longos
"""
        ).content

    def _rag_answer(self, query: str, chunks: list[dict]) -> str:
        prompt = build_answer_prompt(query=query, chunks=chunks)
        return self.llm.invoke(prompt).content

    def answer(self, query: str, chunks: list[dict]) -> dict:
        if is_document_listing_query(query):
            if not chunks:
                return {
                    "type": "document_listing",
                    "answer": "Não encontrei documentos relacionados à consulta.",
                    "confidence": "baixa",
                    "sources": [],
                    "used_rag": True,
                }

            return {
                "type": "document_listing",
                "answer": format_document_listing(chunks),
                "confidence": "média",
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        if is_conceptual_query(query):
            explanation = self._conceptual_answer(query)

            if chunks:
                rag_response = self._rag_answer(query, chunks)

                if is_valid_rag_response(rag_response):
                    return {
                        "type": "hybrid",
                        "answer": (
                            "(Baseado em conhecimento geral do modelo, com complementos "
                            "de documentos regulatórios quando disponíveis)\n\n"
                            f"{explanation}\n\n"
                            "---\n\n"
                            "📚 Baseado nos documentos regulatórios:\n\n"
                            f"{rag_response}"
                        ),
                        "confidence": "média",
                        "sources": extract_sources(chunks),
                        "used_rag": True,
                    }

            return {
                "type": "conceptual",
                "answer": (
                    "(Baseado em conhecimento geral do modelo, devido à ausência de "
                    "evidência relevante nos documentos recuperados)\n\n"
                    f"{explanation}"
                ),
                "confidence": "média",
                "sources": [],
                "used_rag": False,
            }

        if not chunks:
            return {
                "type": "factual",
                "answer": "Não encontrei documentos suficientes para responder com segurança.",
                "confidence": "baixa",
                "sources": [],
                "used_rag": False,
            }

        rag_response = self._rag_answer(query, chunks)

        confidence = "alta" if is_valid_rag_response(rag_response) else "baixa"

        return {
            "type": "factual",
            "answer": rag_response,
            "confidence": confidence,
            "sources": extract_sources(chunks),
            "used_rag": True,
        }