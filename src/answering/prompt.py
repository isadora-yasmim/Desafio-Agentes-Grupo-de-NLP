from __future__ import annotations

import re


SYSTEM_PROMPT = """
Você é um assistente especializado em documentos regulatórios do setor elétrico brasileiro.

Sua tarefa é responder à pergunta do usuário com base EXCLUSIVAMENTE no contexto recuperado abaixo.

═══════════════════════════════════════════
REGRAS DE CONTEÚDO
═══════════════════════════════════════════

1. Use apenas informações presentes nos chunks fornecidos. Nunca invente, complete ou suponha.

2. Se o contexto não responder à pergunta, responda exatamente:
   "Não foi possível responder com segurança com base nos documentos recuperados."
   Em seguida, explique brevemente o que os documentos de fato abordam.

3. Preserve termos técnicos e siglas: TE, TUSD, TUST, TEO, RTP, PRORET, CDE, ESS, EER,
   ONS, CCEE, MME, ANEEL, REH, DSP, PRT, ACP etc.

4. Diferencie relações:
   - DIRETA: responde explicitamente
   - INDIRETA: tema relacionado
   - IRRELEVANTE: não inclua

═══════════════════════════════════════════
REGRA CRÍTICA — EVIDÊNCIA VERIFICÁVEL
═══════════════════════════════════════════

- A evidência deve vir do conteúdo textual do documento
- É PROIBIDO usar como evidência:
  - títulos
  - metadados
  - "temas inferidos"
  - "termos encontrados"
- Use um trecho real, específico e com contexto
"""


# 🔥 1. LIMPEZA DE METADADOS
def clean_content(text: str) -> str:
    lines = text.split("\n")

    filtered = []

    for line in lines:
        line_lower = line.lower()

        if any(prefix in line_lower for prefix in [
            "tipo do ato:",
            "título:",
            "temas inferidos:",
            "termos encontrados:",
            "score",
        ]):
            continue

        filtered.append(line)

    return "\n".join(filtered).strip()


# 🔥 2. EXTRAÇÃO DE SENTENÇA RELEVANTE
def extract_relevant_snippet(content: str, query: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', content)

    query_terms = query.lower().split()

    scored = []

    for sentence in sentences:
        score = sum(1 for term in query_terms if term in sentence.lower())
        if score > 0:
            scored.append((score, sentence))

    if not scored:
        return content[:300]

    best_sentence = sorted(scored, reverse=True)[0][1]

    return best_sentence.strip()


def format_context(chunks: list[dict], query: str) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}

        title = (
            metadata.get("title")
            or metadata.get("titulo")
            or metadata.get("document_title")
            or "Documento sem título"
        )

        doc_type = (
            metadata.get("type")
            or metadata.get("tipo")
            or metadata.get("tipo_ato")
            or "Tipo não informado"
        )

        source = (
            metadata.get("source")
            or metadata.get("fonte")
            or metadata.get("url")
            or "Fonte não informada"
        )

        chunk_type = metadata.get("chunk_type") or "Chunk sem tipo"
        score = chunk.get("score")

        raw_content = (chunk.get("content") or "").strip()

        if not raw_content:
            continue

        # 🔥 PASSO 1: limpar metadados
        cleaned = clean_content(raw_content)

        if not cleaned:
            continue

        # 🔥 PASSO 2: extrair sentença relevante
        content = extract_relevant_snippet(cleaned, query)

        # 🔥 PASSO 3: reduzir tamanho
        content = content[:300]

        score_text = (
            f"{score:.4f}" if isinstance(score, (int, float)) else "Score não informado"
        )

        context_parts.append(
            f"""[CHUNK {index}]
Título: {title}
Tipo do ato: {doc_type}
Tipo do chunk: {chunk_type}
Fonte: {source}
Score: {score_text}

Trecho relevante:
{content}
---"""
        )

    if not context_parts:
        return "Nenhum contexto válido foi recuperado."

    return "\n".join(context_parts)


def build_answer_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks, query)

    return f"""
{SYSTEM_PROMPT}

═══════════════════════════════════════════
CONTEXTO RECUPERADO
═══════════════════════════════════════════

{context}

═══════════════════════════════════════════
PERGUNTA DO USUÁRIO
═══════════════════════════════════════════

{query}

═══════════════════════════════════════════
FORMATO OBRIGATÓRIO
═══════════════════════════════════════════

Resposta direta:
...

Documentos relevantes:
1. [Tipo] — [Título]
   - Relação: direta | indireta
   - Por que é relevante: ...
   - Trecho verificável: "..."

Observações:
...
"""