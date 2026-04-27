from __future__ import annotations

import re


SYSTEM_PROMPT = """
Você é um assistente especializado em documentos regulatórios do setor elétrico brasileiro.

Sua tarefa é responder à pergunta do usuário com base EXCLUSIVAMENTE no contexto recuperado.

REGRAS:
1. Responda diretamente à pergunta.
2. Use apenas informações presentes no contexto.
3. Não invente, não complete e não use conhecimento externo.
4. Seja objetivo: responda em 1 a 3 frases.
5. Não faça introduções genéricas.
6. Não crie seção de fontes.
7. Não liste documentos, exceto se a pergunta pedir explicitamente documentos.
8. Se o contexto não responder à pergunta, diga:
   "Não foi possível responder com segurança com base nos documentos recuperados."
9. Preserve siglas técnicas como TE, TUSD, TUST, TEO, CDE, ANEEL, PRORET, REH, DSP, PRT e ACP.
"""


def clean_content(text: str) -> str:
    """
    Remove metadados e deixa apenas o conteúdo textual útil.
    """
    lines = text.split("\n")
    filtered = []

    ignored_prefixes = [
        "tipo do ato:",
        "título:",
        "temas inferidos:",
        "termos encontrados:",
        "score:",
        "fonte:",
        "tipo do chunk:",
    ]

    for line in lines:
        line_strip = line.strip()
        line_lower = line_strip.lower()

        if not line_strip:
            continue

        if any(line_lower.startswith(prefix) for prefix in ignored_prefixes):
            continue

        filtered.append(line_strip)

    cleaned = " ".join(filtered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_relevant_snippet(content: str, query: str, max_chars: int = 700) -> str:
    """
    Seleciona sentenças mais relacionadas à pergunta.
    """
    if not content:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", content)

    query_terms = [
        term.lower()
        for term in re.findall(r"\b[\wÀ-ÿ]{3,}\b", query.lower())
    ]

    scored: list[tuple[int, str]] = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)

        if score > 0:
            scored.append((score, sentence.strip()))

    if not scored:
        return content[:max_chars].strip()

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = []
    total_chars = 0

    for _, sentence in scored[:3]:
        if total_chars + len(sentence) > max_chars:
            break

        selected.append(sentence)
        total_chars += len(sentence)

    return " ".join(selected).strip() or content[:max_chars].strip()


def get_chunk_content(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}

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
    metadata = chunk.get("metadata") or {}

    return (
        metadata.get("title")
        or metadata.get("titulo")
        or metadata.get("document_title")
        or metadata.get("nome")
        or "Documento sem título"
    )


def get_chunk_type(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}

    return (
        metadata.get("tipo_ato")
        or metadata.get("type")
        or metadata.get("tipo")
        or metadata.get("document_type")
        or "Tipo não informado"
    )


def format_context(chunks: list[dict], query: str) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        raw_content = get_chunk_content(chunk)

        if not raw_content:
            continue

        cleaned = clean_content(raw_content)

        if not cleaned:
            continue

        snippet = extract_relevant_snippet(cleaned, query)

        if not snippet:
            continue

        title = get_chunk_title(chunk)
        doc_type = get_chunk_type(chunk)

        context_parts.append(
            f"[CHUNK {index}]\n"
            f"Documento: {doc_type} — {title}\n"
            f"Conteúdo: {snippet}"
        )

    if not context_parts:
        return "Nenhum contexto válido foi recuperado."

    return "\n\n".join(context_parts)


def build_answer_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks, query)

    return f"""
{SYSTEM_PROMPT}

CONTEXTO RECUPERADO:
{context}

PERGUNTA:
{query}

RESPOSTA:
"""