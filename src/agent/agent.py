from __future__ import annotations
import pickle
from pathlib import Path
from retrieval.hybrid import HybridRetriever
from typing import Any

from openai import OpenAI

from core.config import settings
from retrieval.qdrant_retriever import QdrantRetriever
import re

class RegulatoryAgent:
    def __init__(self):
        self.retriever_without_reranker = None
        self.retriever_with_reranker = None

        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não encontrada. Configure no arquivo .env"
            )

        self.client = OpenAI(api_key=api_key)
        self.hybrid_retriever = self._initialize_hybrid()
    
    def _initialize_hybrid(self) -> HybridRetriever:
        try:
            chunks_path = Path(__file__).resolve().parents[2] / "base" / "chunks_for_bm25.pkl"
            
            if not chunks_path.exists():
                print("Aviso: Arquivo de chunks não encontrado. Usando Qdrant puro.")
                return None
                
            with open(chunks_path, "rb") as f:
                all_chunks = pickle.dump(f) # Erro comum: deve ser pickle.load(f)
                all_chunks = pickle.load(f)
            
            return HybridRetriever(all_chunks=all_chunks)
        except Exception as e:
            print(f"Erro ao inicializar HybridRetriever: {e}")
            return None

    def _extract_filters(self, query: str) -> dict[str, str]:
        """Extrai filtros de metadados a partir da linguagem natural do usuário."""
        filters = {}
        
        # Mapeamento inteligente para as siglas oficiais da ANEEL
        tipo_mapping = {
            r"\bREN\b": "REN",
            r"RESOLU[CÇ][AÃ]O NORMATIVA": "REN",
            r"\bREH\b": "REH",
            r"RESOLU[CÇ][AÃ]O HOMOLOGAT[OÓ]RIA": "REH",
            r"\bDSP\b": "DSP",
            r"DESPACHO": "DSP",
            r"\bPRT\b": "PRT",
            r"PORTARIA": "PRT"
        }
        
        query_upper = query.upper()
        
        for padrao, sigla in tipo_mapping.items():
            if re.search(padrao, query_upper):
                filters["tipo_ato"] = sigla
                break # Pega a primeira que encontrar
                
        return filters

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

    def _retrieve(self, query: str, top_k: int, use_reranker: bool) -> list[Any]:
        filters = self._extract_filters(query)
        tipo_ato = filters.get("tipo_ato")

        # Se o híbrido estiver disponível, usamos ele (BM25 + Semântico)
        if self.hybrid_retriever:
            # O invoke já faz a fusão e retorna Documents do LangChain
            return self.hybrid_retriever.invoke(query)
        
        # Fallback para Qdrant puro caso o arquivo de chunks não exista
        retriever = self._get_retriever(use_reranker)
        return retriever.search(query=query, k=top_k, tipo_ato=tipo_ato)

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
Você é um assistente especialista em regulação do setor elétrico brasileiro (ANEEL).

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