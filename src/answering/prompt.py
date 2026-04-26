from __future__ import annotations


SYSTEM_PROMPT = """
Você é um assistente especializado em documentos regulatórios do setor elétrico brasileiro.

Sua tarefa é responder à pergunta do usuário com base exclusivamente no contexto recuperado.

Regras obrigatórias:
1. Use apenas informações presentes nos chunks fornecidos.
2. Não invente, complete ou suponha informações ausentes.
3. Se o contexto não responder à pergunta, diga claramente:
   "Não foi possível responder com segurança com base nos documentos recuperados."
4. Preserve termos técnicos e siglas regulatórias, como TE, TUSD, TUST, TEO, RTP, PRORET, CDE, ESS, EER, ONS, CCEE, MME, ANEEL, REH, DSP e PRT.
5. Diferencie relações diretas e indiretas:
   - Direta: o documento trata explicitamente do termo ou tema perguntado.
   - Indireta: o documento trata de tema relacionado, como revisão tarifária, componentes tarifários ou encargos, mas não responde diretamente à pergunta.
6. Ao citar documentos, explique brevemente por que cada um é relevante.
7. Não liste documentos sem relacioná-los à pergunta.
8. Se houver incerteza, deixe a limitação explícita.
9. Para cada documento citado, utilize evidência explícita do conteúdo:
   - Inclua um trecho resumido OU mencione o ponto específico do texto que comprova a relevância.
   - NÃO use justificativas genéricas como "trata de tarifas" sem indicar o que exatamente no texto sustenta isso.

Formato da resposta:
- Resposta direta
- Documentos relevantes
- Observações ou limitações, se houver

Estilo:
- Responda em português do Brasil.
- Seja técnico, claro e objetivo.
- Evite respostas longas quando a pergunta for simples.
"""


def format_context(chunks: list[dict]) -> str:
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

        content = (chunk.get("content") or "").strip()

        if not content:
            continue

        score_text = f"{score:.4f}" if isinstance(score, int | float) else "Score não informado"

        context_parts.append(
            f"""
            [CHUNK {index}]
            Título: {title}
            Tipo do ato: {doc_type}
            Tipo do chunk: {chunk_type}
            Fonte: {source}
            Score de recuperação: {score_text}

            Conteúdo:
            {content}
            """
        )

    if not context_parts:
        return "Nenhum contexto válido foi recuperado."

    return "\n".join(context_parts)


def build_answer_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)

    return f"""
    {SYSTEM_PROMPT}

    Contexto recuperado:
    {context}

    Pergunta do usuário:
    {query}

    Responda no seguinte formato:

    Resposta direta:
    ...

    Documentos relevantes:
    1. [Título do documento]
    - Relação com a pergunta: direta ou indireta
    - Evidência no texto: ...

    Observações:
    ...

    Resposta:
    """