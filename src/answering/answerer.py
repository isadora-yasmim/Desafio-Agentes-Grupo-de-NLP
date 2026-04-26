from __future__ import annotations

import re

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


HIGHLIGHT_TERMS = [
    "tarifa social",
    "tarifa social de energia elétrica",
    "tsee",
    "baixa renda",
    "cde",
    "conta de desenvolvimento energético",
    "desconto tarifário",
    "benefício tarifário",
    "subsídio tarifário",
    "subvenção econômica",
    "aneel",
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


def extract_query_terms(query: str) -> list[str]:
    """
    Extrai termos úteis da query para destacar na evidência.
    Remove palavras muito genéricas e mantém expressões importantes.
    """
    query_lower = query.lower()

    stopwords = {
        "quais",
        "documentos",
        "falam",
        "sobre",
        "liste",
        "atos",
        "os",
        "as",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "um",
        "uma",
        "e",
        "a",
        "o",
    }

    terms = []

    for term in HIGHLIGHT_TERMS:
        if term in query_lower:
            terms.append(term)

    for word in re.findall(r"\b[\wÀ-ÿ]{3,}\b", query_lower):
        if word not in stopwords:
            terms.append(word)

    # Remove duplicados preservando ordem
    unique_terms = []
    seen = set()

    for term in terms:
        normalized = term.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_terms.append(term)

    # Termos maiores primeiro evita destacar "tarifa" antes de "tarifa social"
    return sorted(unique_terms, key=len, reverse=True)


def highlight_terms(text: str, query: str) -> str:
    """
    Destaca termos relevantes usando Markdown (**termo**).
    Funciona bem em terminal, Streamlit, Markdown e frontends web.
    """
    if not text:
        return text

    terms = extract_query_terms(query)

    highlighted = text

    for term in terms:
        pattern = re.compile(rf"(?<!\*)\b({re.escape(term)})\b(?!\*)", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)

    return highlighted


def clean_evidence(content: str, doc_type: str, title: str) -> str:
    evidence = content

    for text in [
        f"Tipo do ato: {doc_type}",
        f"Título: {title}",
    ]:
        evidence = evidence.replace(text, "")

    evidence = " ".join(evidence.split()).strip()

    if not evidence:
        return "Sem evidência textual disponível."

    return evidence


def format_document_listing(chunks: list[dict], query: str) -> str:
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

        evidence = clean_evidence(content, doc_type, title)
        evidence = highlight_terms(evidence, query)

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

        # 1. DOCUMENT LISTING
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
                "answer": format_document_listing(chunks, query),
                "confidence": "média",
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        # 2. CONCEITUAL
        if is_conceptual_query(query):
            explanation = self._conceptual_answer(query)

            if chunks:
                rag_response = self._rag_answer(query, chunks)

                if is_valid_rag_response(rag_response):
                    return {
                        "type": "hybrid",
                        "answer": (
                            "(Baseado em conhecimento geral do modelo, com complementos de documentos regulatórios quando disponíveis)\n\n"
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
                    "(Baseado em conhecimento geral do modelo, devido à ausência de evidência relevante nos documentos recuperados)\n\n"
                    f"{explanation}"
                ),
                "confidence": "média",
                "sources": [],
                "used_rag": False,
            }

        # 3. FACTUAL
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