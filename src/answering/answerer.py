from __future__ import annotations

from langchain_core.documents import Document

from answering.llm import build_llm
from answering.prompt import build_answer_prompt


class Answerer:
    def __init__(self):
        self.llm = build_llm()

    def answer(self, query: str, chunks: list[Document]) -> str:
        if not chunks:
            return (
                "Não encontrei documentos relevantes suficientes para responder "
                "à pergunta com segurança."
            )

        prompt = build_answer_prompt(query=query, chunks=chunks)

        response = self.llm.invoke(prompt)

        return response.content