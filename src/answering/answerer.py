from __future__ import annotations

from answering.llm import build_llm
from answering.prompt import build_answer_prompt


# 🔥 função fora da classe (helper)
def is_conceptual_query(query: str) -> bool:
    keywords = ["como funciona", "o que é", "explique", "definição"]

    query_lower = query.lower()

    return any(k in query_lower for k in keywords)


class Answerer:
    def __init__(self):
        self.llm = build_llm()

def answer(self, query: str, chunks: list[dict]) -> str:

    # 🔥 1. detectar pergunta conceitual
    if is_conceptual_query(query):

        explanation = self.llm.invoke(f"""
Explique de forma clara e técnica como funciona a tarifa de energia elétrica no Brasil.

Considere:
- TE (Tarifa de Energia)
- TUSD (Tarifa de Uso do Sistema de Distribuição)
- Encargos e componentes tarifárias

Responda de forma didática e estruturada.
""").content

        # 🔥 2. tenta usar RAG também
        if chunks:
            rag_prompt = build_answer_prompt(query=query, chunks=chunks)
            rag_response = self.llm.invoke(rag_prompt).content

            # 🔥 3. só usa RAG se tiver conteúdo útil
            if "Não foi possível responder com segurança" not in rag_response:
                return f"""
{explanation}

---

Baseado nos documentos regulatórios:

{rag_response}
"""

        # 🔥 fallback: só explicação
        return explanation

    # 🔥 fluxo normal RAG
    if not chunks:
        return "Não encontrei documentos relevantes suficientes para responder com segurança."

    prompt = build_answer_prompt(query=query, chunks=chunks)

    response = self.llm.invoke(prompt)

    return response.content