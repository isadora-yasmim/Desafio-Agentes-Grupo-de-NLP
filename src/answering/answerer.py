from __future__ import annotations

import re

from answering.llm import build_llm
from answering.prompt import build_answer_prompt
from retrieval.confidence import ConfidenceLevel, decide_confidence


CONCEPTUAL_KEYWORDS = [
    "como funciona",
    "o que é",
    "explique",
    "definição",
    "conceito",
    "o que significa",
    "qual o papel",
    "quem define",
]


DOCUMENT_LISTING_KEYWORDS = [
    "quais documentos",
    "liste documentos",
    "documentos falam",
    "documentos sobre",
    "quais atos",
    "atos sobre",
    "quais normas",
    "normas tratam",
    "documentos tratam",
    "existe algum documento",
]


SPECIFIC_VALUE_KEYWORDS = [
    "qual o valor",
    "valor exato",
    "quanto custa",
    "hoje",
    "atual",
    "em 2022",
    "em 2021",
    "em 2020",
]


OUT_OF_DOMAIN_TERMS = [
    "tarifa lunar",
]


DOMAIN_TERMS = {
    "tarifa social": "A Tarifa Social de Energia Elétrica é um benefício tarifário voltado a consumidores de baixa renda, associado à TSEE e a regras regulatórias do setor elétrico.",
    "tsee": "A TSEE é a Tarifa Social de Energia Elétrica, benefício destinado a consumidores de baixa renda.",
    "tusd": "A TUSD é a Tarifa de Uso do Sistema de Distribuição, relacionada ao uso da infraestrutura de distribuição de energia elétrica.",
    "tust": "A TUST é a Tarifa de Uso do Sistema de Transmissão, relacionada ao uso da infraestrutura de transmissão de energia elétrica.",
    "te": "TE significa Tarifa de Energia, componente associado à energia elétrica consumida.",
    "cde": "A CDE é a Conta de Desenvolvimento Energético, mecanismo setorial relacionado ao custeio de políticas públicas e encargos do setor elétrico.",
    "conta de desenvolvimento energético": "A Conta de Desenvolvimento Energético é um mecanismo setorial relacionado ao custeio de políticas públicas e encargos do setor elétrico.",
    "bandeiras tarifárias": "Bandeiras tarifárias são mecanismos que sinalizam condições de custo da geração de energia elétrica e podem afetar o valor pago pelo consumidor.",
    "desconto tarifário": "Desconto tarifário é uma redução aplicada sobre tarifas, geralmente vinculada a critérios regulatórios, benefícios ou políticas específicas.",
    "subsídio tarifário": "Subsídio tarifário é um mecanismo de apoio econômico que reduz o custo final pago por determinados consumidores ou agentes.",
    "benefício tarifário": "Benefício tarifário é uma condição regulatória que reduz ou altera o valor tarifário aplicado a determinado grupo ou situação.",
    "aneel": "A ANEEL é a agência reguladora responsável por regular e fiscalizar o setor elétrico brasileiro, incluindo temas tarifários.",
    "regulação tarifária": "Regulação tarifária é o conjunto de regras e procedimentos usados para definir, revisar e homologar tarifas no setor elétrico.",
}


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
    "bandeiras tarifárias",
    "tusd",
    "tust",
    "te",
    "teo",
    "aneel",
]


def normalize_text(text: str) -> str:
    return text.lower().strip()


def contains_any(query: str, terms: list[str]) -> bool:
    query_lower = normalize_text(query)
    return any(term in query_lower for term in terms)


def is_out_of_domain_query(query: str) -> bool:
    return contains_any(query, OUT_OF_DOMAIN_TERMS)


def is_conceptual_query(query: str) -> bool:
    query_lower = normalize_text(query)

    if any(keyword in query_lower for keyword in CONCEPTUAL_KEYWORDS):
        return True

    short_definition_patterns = [
        r"^o que é\s+.+",
        r"^o que significa\s+.+",
        r"^explique\s+.+",
    ]

    return any(re.search(pattern, query_lower) for pattern in short_definition_patterns)


def is_document_listing_query(query: str) -> bool:
    return contains_any(query, DOCUMENT_LISTING_KEYWORDS)


def is_specific_value_query(query: str) -> bool:
    return contains_any(query, SPECIFIC_VALUE_KEYWORDS)


def get_domain_definition(query: str) -> str | None:
    query_lower = normalize_text(query)

    sorted_terms = sorted(DOMAIN_TERMS.keys(), key=len, reverse=True)

    for term in sorted_terms:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, query_lower):
            return DOMAIN_TERMS[term]

    return None


def is_valid_rag_response(response: str) -> bool:
    if not response:
        return False

    invalid_signals = [
        "não foi possível responder",
        "não contém informações",
        "não há informação suficiente",
        "não há informações suficientes",
        "não encontrei informações suficientes",
        "não encontrei evidência suficiente",
        "não há evidência suficiente",
        "nenhum contexto válido foi recuperado",
    ]

    response_lower = normalize_text(response)
    return not any(signal in response_lower for signal in invalid_signals)


def get_chunk_metadata(chunk: dict) -> dict:
    return chunk.get("metadata") or {}


def get_chunk_content(chunk: dict) -> str:
    metadata = get_chunk_metadata(chunk)

    content = (
        chunk.get("content")
        or chunk.get("page_content")
        or chunk.get("text")
        or chunk.get("chunk")
        or metadata.get("content")
        or metadata.get("page_content")
        or metadata.get("text")
        or metadata.get("chunk")
        or metadata.get("ementa")
        or metadata.get("summary")
        or ""
    )

    return str(content).strip()


def get_chunk_title(chunk: dict) -> str:
    metadata = get_chunk_metadata(chunk)

    return (
        metadata.get("title")
        or metadata.get("titulo")
        or metadata.get("document_title")
        or metadata.get("nome")
        or "Documento sem título"
    )


def get_chunk_type(chunk: dict) -> str:
    metadata = get_chunk_metadata(chunk)

    return (
        metadata.get("tipo_ato")
        or metadata.get("type")
        or metadata.get("tipo")
        or metadata.get("document_type")
        or "Tipo não informado"
    )


def get_chunk_score(chunk: dict):
    metadata = get_chunk_metadata(chunk)

    return (
        chunk.get("score")
        or metadata.get("score")
        or metadata.get("vector_score")
        or metadata.get("semantic_score")
        or metadata.get("bm25_score")
    )


def get_chunk_final_score(chunk: dict):
    metadata = get_chunk_metadata(chunk)

    return (
        chunk.get("final_score")
        or metadata.get("final_score")
        or metadata.get("reranker_score")
        or metadata.get("score")
        or chunk.get("score")
        or get_chunk_score(chunk)
    )


def extract_sources(chunks: list[dict]) -> list[dict]:
    sources = []

    for chunk in chunks:
        metadata = get_chunk_metadata(chunk)

        sources.append(
            {
                "title": get_chunk_title(chunk),
                "type": get_chunk_type(chunk),
                "source": metadata.get("source") or metadata.get("fonte"),
                "score": get_chunk_score(chunk),
                "final_score": get_chunk_final_score(chunk),
                "content": get_chunk_content(chunk),
                "metadata": metadata,
            }
        )

    return sources


def extract_query_terms(query: str) -> list[str]:
    query_lower = normalize_text(query)

    stopwords = {
        "quais",
        "documentos",
        "falam",
        "sobre",
        "liste",
        "atos",
        "normas",
        "tratam",
        "existe",
        "algum",
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

    unique_terms = []
    seen = set()

    for term in terms:
        normalized = normalize_text(term)

        if normalized not in seen:
            seen.add(normalized)
            unique_terms.append(term)

    return sorted(unique_terms, key=len, reverse=True)


def highlight_terms(text: str, query: str) -> str:
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

    removable_parts = [
        f"Tipo do ato: {doc_type}",
        f"Título: {title}",
    ]

    for part in removable_parts:
        evidence = evidence.replace(part, "")

    evidence = " ".join(evidence.split()).strip()

    if not evidence:
        return "Sem evidência textual disponível."

    return evidence


def format_document_listing(chunks: list[dict], query: str) -> str:
    documents = []

    for chunk in chunks:
        title = get_chunk_title(chunk)
        doc_type = get_chunk_type(chunk)

        content = get_chunk_content(chunk)
        score = get_chunk_score(chunk)
        final_score = get_chunk_final_score(chunk)

        score_text = (
            f"{score:.4f}" if isinstance(score, (int, float)) else "não informado"
        )

        final_score_text = (
            f"{float(final_score):.4f}" if final_score is not None else "não informado"
        )

        evidence = clean_evidence(content, doc_type, title)
        evidence = highlight_terms(evidence, query)

        documents.append(
            f"- **{doc_type} — {title}**\n"
            f"  - Evidência: {evidence}\n"
            f"  - Score de recuperação: {score_text}\n"
            f"  - Score final de confiança: {final_score_text}"
        )

    return "Encontrei os seguintes documentos relacionados:\n\n" + "\n\n".join(documents)


def normalize_confidence_label(level: ConfidenceLevel) -> str:
    if level == ConfidenceLevel.HIGH:
        return "alta"

    if level == ConfidenceLevel.MEDIUM:
        return "média"

    return "baixa"


def should_use_domain_definition(query: str, chunks: list[dict]) -> bool:
    if not is_conceptual_query(query):
        return False

    if is_specific_value_query(query):
        return False

    if is_out_of_domain_query(query):
        return False

    return get_domain_definition(query) is not None


class Answerer:
    def __init__(self):
        self.llm = build_llm()

    def _conceptual_answer(self, query: str) -> str:
        domain_definition = get_domain_definition(query)

        if domain_definition:
            return domain_definition

        return self.llm.invoke(
            f"""
Responda de forma direta, técnica e objetiva:

{query}

Regras:
- Responda em 1 a 2 frases.
- Não use tópicos.
- Não faça introdução.
- Não cite documentos.
- Não invente números, datas ou normas.
- Se o termo não for reconhecido no domínio elétrico, diga que não há evidência suficiente.
"""
        ).content.strip()

    def _rag_answer(self, query: str, chunks: list[dict]) -> str:
        prompt = build_answer_prompt(query=query, chunks=chunks)
        return self.llm.invoke(prompt).content.strip()

    def answer(self, query: str, chunks: list[dict]) -> dict:
        confidence_decision = decide_confidence(chunks)
        confidence_label = normalize_confidence_label(confidence_decision.level)

        if is_out_of_domain_query(query):
            return {
                "type": "conceptual" if is_conceptual_query(query) else "factual",
                "answer": (
                    "⚠️ Não encontrei evidência documental suficiente para esse termo no domínio regulatório de energia elétrica."
                ),
                "confidence": "baixa",
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks) if chunks else [],
                "used_rag": False,
            }

        if is_document_listing_query(query):
            if not chunks:
                return {
                    "type": "document_listing",
                    "answer": "Não encontrei documentos relacionados à consulta.",
                    "confidence": "baixa",
                    "final_score": 0.0,
                    "sources": [],
                    "used_rag": True,
                }

            answer = format_document_listing(chunks, query)

            if confidence_decision.level == ConfidenceLevel.LOW:
                answer = (
                    f"{confidence_decision.warning}\n\n"
                    "A listagem abaixo pode conter documentos relacionados, mas não deve ser tratada como resposta conclusiva.\n\n"
                    f"{answer}"
                )

            return {
                "type": "document_listing",
                "answer": answer,
                "confidence": confidence_label,
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        if is_conceptual_query(query):
            domain_definition = get_domain_definition(query)

            if chunks:
                rag_response = self._rag_answer(query, chunks)

                if is_valid_rag_response(rag_response):
                    return {
                        "type": "hybrid",
                        "answer": rag_response,
                        "confidence": confidence_label,
                        "final_score": confidence_decision.final_score,
                        "sources": extract_sources(chunks),
                        "used_rag": True,
                    }

                if domain_definition:
                    return {
                        "type": "hybrid",
                        "answer": domain_definition,
                        "confidence": confidence_label,
                        "final_score": confidence_decision.final_score,
                        "sources": extract_sources(chunks),
                        "used_rag": True,
                    }

            explanation = domain_definition or self._conceptual_answer(query)

            return {
                "type": "conceptual",
                "answer": explanation,
                "confidence": confidence_label,
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks) if chunks else [],
                "used_rag": False,
            }

        if not chunks:
            return {
                "type": "factual",
                "answer": "⚠️ Não encontrei documentos suficientes para responder com segurança.",
                "confidence": "baixa",
                "final_score": 0.0,
                "sources": [],
                "used_rag": False,
            }

        if is_specific_value_query(query) and confidence_decision.level != ConfidenceLevel.HIGH:
            return {
                "type": "factual",
                "answer": (
                    "⚠️ Os documentos recuperados não trouxeram informação suficiente para responder com segurança."
                ),
                "confidence": "baixa",
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        if confidence_decision.level == ConfidenceLevel.LOW:
            return {
                "type": "factual",
                "answer": (
                    f"{confidence_decision.warning}\n\n"
                    "Não vou gerar uma resposta afirmativa sem evidência documental suficiente."
                ),
                "confidence": "baixa",
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks),
                "used_rag": False,
            }

        rag_response = self._rag_answer(query, chunks)

        if is_valid_rag_response(rag_response):
            return {
                "type": "factual",
                "answer": rag_response,
                "confidence": confidence_label,
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        domain_definition = get_domain_definition(query)

        if domain_definition and not is_specific_value_query(query):
            return {
                "type": "factual",
                "answer": domain_definition,
                "confidence": confidence_label,
                "final_score": confidence_decision.final_score,
                "sources": extract_sources(chunks),
                "used_rag": True,
            }

        return {
            "type": "factual",
            "answer": (
                "⚠️ Os documentos recuperados não trouxeram informação suficiente para responder com segurança."
            ),
            "confidence": "baixa",
            "final_score": confidence_decision.final_score,
            "sources": extract_sources(chunks),
            "used_rag": True,
        }