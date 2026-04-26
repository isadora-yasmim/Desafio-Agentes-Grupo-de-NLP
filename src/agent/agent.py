from __future__ import annotations

from typing import Any
import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.qdrant_retriever import QdrantRetriever


# 🔥 carrega .env
load_dotenv()


class RegulatoryAgent:
    def __init__(self):
        self.retriever_without_reranker = None
        self.retriever_with_reranker = None

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não encontrada. Configure no arquivo .env"
            )

        self.client = OpenAI(api_key=api_key)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
        top_k = inputs.get("top_k", 5)
        use_reranker = inputs.get("use_reranker", False)

        return self.answer(
            query=question,
            top_k=top_k,
            use_reranker=use_reranker,
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> dict[str, Any]:
        if not query.strip():
            return {
                "answer": "Digite uma pergunta.",
                "sources": [],
            }

        try:
            docs = self._retrieve(
                query=query,
                top_k=top_k,
                use_reranker=use_reranker,
            )
        except Exception as error:
            return {
                "answer": (
                    "Ocorreu um erro ao consultar os documentos.\n\n"
                    f"Erro: `{type(error).__name__}: {error}`"
                ),
                "sources": [],
            }

        if not docs:
            return {
                "answer": "Não encontrei documentos relevantes para essa pergunta.",
                "sources": [],
            }

        try:
            answer = self._generate_answer(
                query=query,
                documents=docs,
            )
        except Exception as error:
            answer = (
                "Encontrei documentos relevantes, mas ocorreu um erro ao gerar a resposta com LLM.\n\n"
                f"Erro: `{type(error).__name__}: {error}`"
            )

        return {
            "answer": answer,
            "sources": self._format_sources(docs),
        }

    def _get_retriever(self, use_reranker: bool) -> QdrantRetriever:
        if use_reranker:
            if self.retriever_with_reranker is None:
                try:
                    self.retriever_with_reranker = QdrantRetriever(use_reranker=True)
                except Exception:
                    self.retriever_with_reranker = QdrantRetriever(use_reranker=False)

            return self.retriever_with_reranker

        if self.retriever_without_reranker is None:
            self.retriever_without_reranker = QdrantRetriever(use_reranker=False)

        return self.retriever_without_reranker

    def _retrieve(
        self,
        query: str,
        top_k: int,
        use_reranker: bool,
    ) -> list[Any]:
        retriever = self._get_retriever(use_reranker)

        method_names = [
            "retrieve",
            "search",
            "query",
            "get_relevant_documents",
            "similarity_search",
        ]

        for method_name in method_names:
            method = getattr(retriever, method_name, None)

            if method is None:
                continue

            try:
                return method(
                    query=query,
                    top_k=top_k,
                    use_reranker=use_reranker,
                )
            except TypeError:
                try:
                    return method(query=query, top_k=top_k)
                except TypeError:
                    try:
                        return method(query, top_k)
                    except TypeError:
                        return method(query)

        raise AttributeError(
            "QdrantRetriever não possui método compatível de busca."
        )

    def _generate_answer(self, query: str, documents: list[Any]) -> str:
        context_parts = []

        for index, doc in enumerate(documents[:5], start=1):
            metadata = self._get_metadata(doc)
            content = self._get_content(doc)

            title = (
                metadata.get("title")
                or metadata.get("titulo")
                or metadata.get("document_title")
                or metadata.get("nome")
                or "Documento sem título"
            )

            doc_type = (
                metadata.get("type")
                or metadata.get("tipo")
                or metadata.get("document_type")
                or "Tipo não informado"
            )

            if content:
                context_parts.append(
                    f"[Documento {index}]\n"
                    f"Título: {title}\n"
                    f"Tipo: {doc_type}\n"
                    f"Conteúdo:\n{content[:2500]}"
                )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
Você é um especialista em regulação do setor elétrico brasileiro.

Responda à pergunta do usuário de forma clara, objetiva e didática.

Regras:
- Responda diretamente à pergunta.
- Use apenas as informações do contexto.
- Não liste os documentos encontrados na resposta principal.
- Não invente informações.
- Se o contexto não for suficiente, diga isso claramente.
- Seja conciso, mas completo.
- As fontes serão exibidas separadamente pela interface, então não precisa criar uma seção de fontes.

Pergunta do usuário:
{query}

Contexto recuperado:
{context}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você responde perguntas sobre documentos regulatórios "
                        "do setor elétrico brasileiro com base em contexto recuperado."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    def _format_sources(self, documents: list[Any]) -> list[dict[str, Any]]:
        sources = []

        for doc in documents:
            metadata = self._get_metadata(doc)
            content = self._get_content(doc)

            sources.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": metadata.get("score"),
                }
            )

        return sources

    def _get_metadata(self, doc: Any) -> dict[str, Any]:
        if isinstance(doc, dict):
            return doc.get("metadata", {}) or {}

        return getattr(doc, "metadata", {}) or {}

    def _get_content(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return (
                doc.get("content")
                or doc.get("page_content")
                or doc.get("text")
                or ""
            )

        return (
            getattr(doc, "page_content", None)
            or getattr(doc, "content", None)
            or getattr(doc, "text", None)
            or ""
        )


def build_agent() -> RegulatoryAgent:
    return RegulatoryAgent()