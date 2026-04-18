<!-- HEADER CENTRALIZADO -->
<h1 align="center">⚡ RAG para Domínio Elétrico</h1>

<p align="center">
  Sistema de Retrieval-Augmented Generation com foco em alta qualidade, avaliação robusta e transparência.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LLM-GPT--4o--mini-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Embeddings-E5%20%7C%20OpenAI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/VectorDB-Supabase-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-LangChain-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge" />
</p>

---

## Estrutura do Projeto

<p align="center">
  <img src="assets\rag_project_architecture.svg" width="600"/>
</p>

---

## Stack

<div align="center">

| Camada        | Tecnologia                                    |
| ------------- | --------------------------------------------- |
| **LLM**       | GPT-4o-mini                                   |
| **Embedding** | text-embedding-3-small / multilingual-e5-base |
| **Vector DB** | Supabase + pgvector                           |
| **Framework** | LangChain                                     |
| **UI**        | Streamlit                                     |
| **Infra**     | Docker                                        |

</div>

---

## Diferenciais do Projeto

### Retrieval Avançado

- Ensemble Retriever (**semântico + BM25**)
- **HyDE** para melhorar queries
- **Reranking com cross-encoder**

```text
Fluxo:
Query → HyDE → Retrieval híbrido → Reranking → Resposta
```

### Avaliação com RAGAS

Métricas utilizadas:

- ✔ Faithfulness
- ✔ Answer Relevancy
- ✔ Context Precision

📈 Resultados quantitativos para apresentação técnica

### Transparência (UI)

Interface mostra:

- 📄 Chunks recuperados
- 📊 Score de similaridade
- 🔗 Fonte do documento
- 🧠 Nível de confiança

### Organização

```bash
/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml          # dependências
├── README.md
├── .env
├── .gitignore
│
├── src/                    # CÓDIGO PRINCIPAL
│       ├── __init__.py
│       │
│       ├── ingestion/      # entrada de dados
│       │   ├── parser.py
│       │   ├── chunker.py
│       │   └── embedder.py
│       │
│       ├── retrieval/      # busca
│       │   ├── hybrid.py
│       │   └── reranker.py
│       │
│       ├── agent/          # LLM / LangChain
│       │   ├── chain.py
│       │   └── prompts.py
│       │
│       ├── eval/           # métricas (RAGAS)
│       │   └── benchmark.py
│       │
│       ├── ui/             # interface
│       │   └── app.py
│       │
│       ├── core/           # config / utilidades globais
│       │   ├── config.py
│       │   ├── settings.py
│       │   └── logging.py
│       │
│       └── infrastructure/ # integrações externas
│           ├── database.py
│           ├── vector_store.py
│           └── llm_provider.py
│
├── tests/                  # testes
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
├── scripts/                # scripts utilitários
│   ├── ingest_data.py
│   └── run_eval.py
│
├── docs/                   # documentação
│   └── arquitetura.md
│
├── base/                   # documentação
│   ├── _MACOSX/
│   ├── biblioteca_aneel_gov_br_legislacao_2016_metadados.json
│   ├── biblioteca_aneel_gov_br_legislacao_2021_metadados.json
│   └── biblioteca_aneel_gov_br_legislacao_2022_metadados.json 
│
└── assets/                 # imagens, diagramas
```

## Como Executar

(alterar depois de finalizado o projeto)

```bash
# Subir ambiente
docker-compose up --build

# Rodar aplicação
streamlit run ui/app.py
```
