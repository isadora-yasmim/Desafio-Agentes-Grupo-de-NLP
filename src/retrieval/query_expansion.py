from __future__ import annotations


# 🔥 regras de expansão por domínio
DOMAIN_EXPANSIONS = {
    "tarifa social": [
        "tarifa social de energia elétrica",
        "tarifa social de energia",
        "TSEE",
        "baixa renda",
        "consumidor baixa renda",
        "unidade consumidora baixa renda",
        "benefício tarifário",
        "desconto tarifário",
        "desconto na tarifa",
        "subvenção econômica",
        "subsídio tarifário",
        "CDE",
        "Conta de Desenvolvimento Energético",
        "Lei 12.212",
        "Decreto 7.583",
        "ANEEL",
    ],
    "tusd": [
        "tarifa uso sistema distribuição",
        "custo distribuição energia",
    ],
    "tust": [
        "tarifa uso sistema transmissão",
        "custo transmissão energia",
    ],
    "te": [
        "tarifa energia consumo",
        "custo energia gerada",
    ],
}


def normalize(text: str) -> str:
    return text.lower().strip()


def build_expanded_query(query: str) -> str:
    query_norm = normalize(query)

    expanded_terms = [query]

    # 🔥 aplica expansão por domínio
    for key, expansions in DOMAIN_EXPANSIONS.items():
        if key in query_norm:
            expanded_terms.extend(expansions)

    # 🔥 fallback genérico (sempre ajuda o embedding)
    expanded_terms.extend([
        "energia elétrica",
        "regulação ANEEL",
        "tarifas energia",
    ])

    # remove duplicados
    expanded_terms = list(set(expanded_terms))

    return " ".join(expanded_terms)


def reciprocal_rank_fusion(list_of_list_of_docs: list[list], k: int = 60) -> list:
    """Funde várias listas de documentos usando Reciprocal Rank Fusion (RRF)."""
    doc_scores = {}
    doc_map = {}

    for docs in list_of_list_of_docs:
        for rank, doc in enumerate(docs):
            # Tratamento seguro para doc sendo um dicionário ou um objeto Document (LangChain)
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            content = doc.page_content if hasattr(doc, "page_content") else doc.get("content", "")
            
            # Usa doc_id se existir, caso contrário usa um trecho do conteúdo como chave única
            doc_id = metadata.get("doc_id") or content[:80]

            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_map[doc_id] = doc

            doc_scores[doc_id] += 1.0 / (rank + k)

    # Ordena os documentos com base na pontuação RRF
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def bm25_multi_query_retrieve(retriever, query: str, k_per_query: int = 5) -> list:
    """Executa o BM25 para a query original e suas expansões, fundindo os resultados."""
    query_norm = normalize(query)
    queries = [query]

    for key, expansions in DOMAIN_EXPANSIONS.items():
        if key in query_norm:
            queries.extend(expansions)

    queries.extend([
        "energia elétrica",
        "regulação ANEEL",
        "tarifas energia",
    ])

    queries = list(set(queries))
    
    all_results = []
    for q in queries:
        docs = retriever.invoke(q)
        all_results.append(docs[:k_per_query])
        
    return reciprocal_rank_fusion(all_results)