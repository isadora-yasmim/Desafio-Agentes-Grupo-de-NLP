"""
hybrid.py
---------
Retrieval híbrido: semântico (vetores) + BM25 (keyword) + HyDE.

Fluxo completo:
  Query
    ↓
  HyDE (gera documento hipotético para melhorar o embedding da query)
    ↓
  EnsembleRetriever
    ├── SupabaseVectorStore (semântico, cosine similarity)
    └── BM25Retriever (keyword, funciona bem p/ termos como "ANEEL", "SCG")
    ↓
  Reranker (cross-encoder BGE, seleciona top-K)
    ↓
  Chunks finais para o LLM
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import SupabaseVectorStore

from core.config import settings
from core.logger import get_logger
from ingestion.embedder import get_embeddings
from core.database import get_supabase_client

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Vector Store Retriever (semântico)
# ---------------------------------------------------------------------------

def build_semantic_retriever(k: int | None = None) -> SupabaseVectorStore:
    """Retorna o retriever semântico conectado ao Supabase."""
    k = k or settings.RETRIEVAL_K_SEMANTIC
    embeddings = get_embeddings()
    client = get_supabase_client()

    vector_store = SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name=settings.SUPABASE_TABLE,
        query_name="match_documents",
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ---------------------------------------------------------------------------
# 2. BM25 Retriever (keyword)
# ---------------------------------------------------------------------------

def build_bm25_retriever(
    docs: list[Document],
    k: int | None = None,
) -> BM25Retriever:
    """
    Constrói o retriever BM25 a partir dos chunks carregados.

    Nota: O BM25Retriever do LangChain opera in-memory.
    Para produção com muitos documentos, use Elasticsearch ou
    OpenSearch no lugar do BM25Retriever.

    Args:
        docs: Lista de LangChain Documents (todos os chunks).
        k: Número de documentos a retornar.
    """
    k = k or settings.RETRIEVAL_K_BM25
    retriever = BM25Retriever.from_documents(docs, k=k)
    logger.info(f"BM25Retriever construído com {len(docs)} documentos.")
    return retriever


# ---------------------------------------------------------------------------
# 3. Ensemble Retriever
# ---------------------------------------------------------------------------

def build_ensemble_retriever(
    all_chunks: list[Document],
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> EnsembleRetriever:
    """
    Combina busca semântica + BM25 via Reciprocal Rank Fusion (RRF).

    O peso semântico maior é proposital: a qualidade dos vetores é
    melhor para perguntas em linguagem natural. O BM25 complementa
    para termos técnicos exatos (siglas, números de despacho).

    Args:
        all_chunks: Todos os chunks (necessário para construir o BM25).
        semantic_weight: Peso do retriever semântico (0 a 1).
        bm25_weight: Peso do retriever BM25 (0 a 1).
    """
    assert abs(semantic_weight + bm25_weight - 1.0) < 1e-6, \
        "Pesos devem somar 1.0"

    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever(all_chunks)

    ensemble = EnsembleRetriever(
        retrievers=[semantic, bm25],
        weights=[semantic_weight, bm25_weight],
    )
    logger.info(
        f"EnsembleRetriever criado: semântico={semantic_weight}, BM25={bm25_weight}"
    )
    return ensemble


# ---------------------------------------------------------------------------
# 4. HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------

HYDE_PROMPT = """Você é um especialista em regulação do setor elétrico brasileiro (ANEEL).
Dado a pergunta abaixo, escreva um trecho de documento regulatório que responderia diretamente a ela.
Escreva como se fosse uma ementa ou despacho real da ANEEL. Seja técnico e conciso (3-5 linhas).

Pergunta: {question}

Trecho hipotético:"""


class HyDERetriever:
    """
    Wraps um retriever base e aplica HyDE antes da busca.

    HyDE: em vez de embedar a query do usuário diretamente, pede ao LLM
    para gerar um "documento hipotético" que responderia a query, e usa
    esse documento como query de embedding. Isso alinha melhor o espaço
    semântico da busca com o espaço dos documentos reais.
    """

    def __init__(
        self,
        base_retriever,
        llm: BaseLanguageModel,
    ):
        self.base_retriever = base_retriever
        self.llm = llm

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Gera o documento hipotético
        hyde_doc = self._generate_hypothetical(query)
        logger.debug(f"HyDE gerado: {hyde_doc[:200]}...")

        # Usa o documento hipotético como query para o retriever
        return self.base_retriever.get_relevant_documents(hyde_doc)

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        hyde_doc = self._generate_hypothetical(query)
        return await self.base_retriever.aget_relevant_documents(hyde_doc)

    def _generate_hypothetical(self, question: str) -> str:
        prompt = HYDE_PROMPT.format(question=question)
        response = self.llm.invoke(prompt)
        # Extrai o texto da resposta (compatível com ChatOpenAI e LLM base)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
