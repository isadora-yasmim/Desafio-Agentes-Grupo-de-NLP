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