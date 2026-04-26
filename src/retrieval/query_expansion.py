"""
retrieval/query_expansion.py
-----------------------------
Expansão de query por sinônimos do domínio elétrico (ANEEL).

Usado em duas frentes:
  1. HyDE: enriquece o prompt com termos técnicos antes de gerar o
     documento hipotético, melhorando o embedding da query.
  2. BM25 multi-query: roda o BM25 para cada variação e funde os
     resultados por Reciprocal Rank Fusion (RRF), garantindo que
     termos curtos como "TE" ou "TUSD" sejam ancorados exatamente.

Por que isso importa?
  - Embeddings generalizam: "tarifa de energia" ≈ "custo energético" ≈
    "preço da eletricidade". Bom para linguagem natural, ruim para siglas.
  - BM25 é literal: "TE" só bate em "TE". A expansão garante que a busca
    keyword cubra todas as formas que o documento pode usar o conceito.
"""
from __future__ import annotations

import re
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Dicionário de sinônimos do domínio elétrico
# ---------------------------------------------------------------------------
# Estrutura: "termo do usuário" → [variações que podem aparecer nos docs]
# O termo-chave não precisa ser exato — é comparado em lowercase.

DOMAIN_SYNONYMS: dict[str, list[str]] = {
    # Componentes tarifários
    "tarifa de energia elétrica": ["TE", "Tarifa de Energia", "componente TE", "tarifa de energia"],
    "tarifa de energia":          ["TE", "Tarifa de Energia", "componente TE"],
    "tusd":                       ["TUSD", "Tarifa de Uso do Sistema de Distribuição"],
    "tust":                       ["TUST", "Tarifa de Uso do Sistema de Transmissão"],
    "teo":                        ["TEO", "Tarifa de Energia de Otimização"],
    "encargos setoriais":         ["CDE", "PROINFA", "CCC", "ESS", "EER", "CUST", "encargos de capacidade"],
    "bandeira tarifária":         ["bandeira verde", "bandeira amarela", "bandeira vermelha", "bandeira escassez"],

    # Agentes e contratos
    "distribuidora":              ["concessionária de distribuição", "permissionária", "agente de distribuição"],
    "transmissora":               ["concessionária de transmissão", "agente de transmissão", "ONS"],
    "comercializador":            ["agente comercializador", "livre comércio", "ACL", "ACR"],
    "consumidor livre":           ["ACL", "ambiente de contratação livre", "autoprodução"],
    "gerador":                    ["usina", "empreendimento de geração", "central geradora", "UHE", "PCH", "UFV", "EOL"],

    # Tipos de geração
    "energia solar":              ["fotovoltaica", "UFV", "usina fotovoltaica", "GD solar", "geração distribuída solar"],
    "energia eólica":             ["EOL", "usina eólica", "parque eólico", "aerogerador"],
    "hidrelétrica":               ["UHE", "PCH", "CGH", "usina hidrelétrica", "aproveitamento hidrelétrico"],
    "termelétrica":               ["UTE", "usina termelétrica", "despacho térmico"],

    # Processos regulatórios
    "revisão tarifária":          ["RTP", "revisão tarifária periódica", "reajuste tarifário", "ANEEL tarifa"],
    "outorga":                    ["autorização", "concessão", "permissão", "ato autorizativo"],
    "leilão":                     ["leilão de energia", "leilão de transmissão", "certame", "edital de leilão"],
    "consulta pública":           ["ACP", "audiência pública", "AAP", "tomada de subsídios", "ATS"],
    "infração":                   ["auto de infração", "multa", "penalidade", "advertência", "embargo"],
    "qualidade de energia":       ["DEC", "FEC", "ANSI", "harmônicas", "fator de potência", "continuidade"],

    # Siglas institucionais
    "aneel":                      ["Agência Nacional de Energia Elétrica", "regulador"],
    "ons":                        ["Operador Nacional do Sistema", "operação do sistema"],
    "ccee":                       ["Câmara de Comercialização de Energia Elétrica", "mercado de energia"],
    "mme":                        ["Ministério de Minas e Energia", "ministério"],
    "anp":                        ["Agência Nacional do Petróleo", "combustível regulado"],
}


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def expand_query(query: str) -> list[str]:
    """
    Retorna lista com a query original + todas as expansões encontradas.
    A query original é sempre o primeiro elemento.

    Exemplo:
        expand_query("quero saber sobre tarifa de energia elétrica")
        → [
            "quero saber sobre tarifa de energia elétrica",
            "TE",
            "Tarifa de Energia",
            "componente TE",
            "tarifa de energia",
          ]
    """
    query_lower = query.lower()
    expansions: list[str] = []

    for term, synonyms in DOMAIN_SYNONYMS.items():
        if term in query_lower:
            for syn in synonyms:
                # Evita adicionar duplicatas e o próprio termo já presente
                if syn.lower() not in query_lower and syn not in expansions:
                    expansions.append(syn)

    return [query] + expansions


def build_expanded_query(query: str) -> str:
    """
    Constrói uma string única combinando query original + sinônimos.
    Útil para passar ao embedding do HyDE como contexto enriquecido.

    Exemplo:
        "tarifas de energia elétrica [TE, Tarifa de Energia, componente TE]"
    """
    expansions = expand_query(query)
    if len(expansions) == 1:
        return query  # nenhuma expansão encontrada

    synonyms_part = ", ".join(expansions[1:])
    return f"{query} [{synonyms_part}]"


def get_synonyms_for_query(query: str) -> list[str]:
    """Retorna apenas as expansões (sem a query original)."""
    return expand_query(query)[1:]


# ---------------------------------------------------------------------------
# RRF para fusão de múltiplos rankings
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Funde múltiplos rankings de documentos usando Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Usado para combinar os resultados BM25 de múltiplas queries
    (original + sinônimos) antes de passar ao reranker.

    Args:
        rankings: Lista de listas de Documents (um por query).
        k: Constante de suavização (60 é o padrão da literatura).

    Returns:
        Lista única ordenada por score RRF decrescente.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            # Chave de identidade do documento
            key = doc.metadata.get("doc_id") or doc.page_content[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# BM25 multi-query com expansão
# ---------------------------------------------------------------------------

def bm25_multi_query_retrieve(
    bm25_retriever,
    query: str,
    k_per_query: int = 20,
) -> list[Document]:
    """
    Roda o BM25 para a query original + cada sinônimo encontrado,
    depois funde os resultados por RRF.

    Isso resolve o problema de "TE" não aparecer em buscas por
    "tarifa de energia" no BM25, pois o BM25 é literal.

    Args:
        bm25_retriever: Instância de BM25Retriever já construída.
        query: Query original do usuário.
        k_per_query: Documentos a recuperar por sub-query.

    Returns:
        Lista fundida e ordenada por RRF.
    """
    queries = expand_query(query)
    rankings: list[list[Document]] = []

    # Ajusta k temporariamente para cada sub-query
    original_k = bm25_retriever.k
    bm25_retriever.k = k_per_query

    for q in queries:
        try:
            docs = bm25_retriever.invoke(q)
            rankings.append(docs)
        except Exception:
            pass  # sub-query falhou, ignora

    bm25_retriever.k = original_k  # restaura k original

    if not rankings:
        return []

    return reciprocal_rank_fusion(rankings)