from __future__ import annotations

from langchain_core.documents import Document

from answering.llm import build_llm
from answering.prompt import build_answer_prompt
import re

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

    def extract_relevant_snippet(content: str, query: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', content)

        query_terms = query.lower().split()

        scored = []

        for sentence in sentences:
            score = sum(1 for term in query_terms if term in sentence.lower())
            if score > 0:
                scored.append((score, sentence))

        if not scored:
            return content[:200]

        best_sentence = sorted(scored, reverse=True)[0][1]

        return best_sentence.strip()