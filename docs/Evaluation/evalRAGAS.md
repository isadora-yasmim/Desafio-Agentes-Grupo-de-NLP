# 📊 Avaliação do Sistema RAG (RAGAS)

Este documento apresenta a avaliação do sistema utilizando **RAGAS**, comparando o desempenho **com e sem reranker**.

## 📈 Comparação de Métricas

| Métrica               | Sem reranker | Com reranker | Variação |
|----------------------|-------------|-------------|----------|
| **Faithfulness**     | 0.74        | **0.83**    | ⬆️ +0.09 |
| **Answer Relevancy** | **0.90**    | 0.88        | ⬇️ -0.02 |
| **Context Precision**| 0.85        | **0.92**    | ⬆️ +0.07 |


## 🧠 Análise dos Resultados

### 🟢 Faithfulness (+0.09)
- Aumentou significativamente
- Indica maior fidelidade das respostas ao contexto
- Redução de alucinações e extrapolações do modelo

**Interpretação:**  
O reranker ajudou o modelo a utilizar melhor os documentos corretos.


### 🟢 Context Precision (+0.07)
- Melhora clara na qualidade do contexto recuperado
- Menos documentos irrelevantes (“ruído”)

**Interpretação:**  
O reranker aumenta a precisão do processo de retrieval.


### 🔴 Answer Relevancy (-0.02)
- Pequena queda (impacto baixo)
- Respostas ficaram mais objetivas e restritas ao contexto

**Interpretação:**  
O sistema se tornou mais preciso, porém ligeiramente menos abrangente nas respostas.


## 🎯 Conclusão

A introdução do reranker trouxe ganhos importantes para o sistema:

- ✅ **Mais fidelidade ao contexto (menos alucinação)**
- ✅ **Melhor qualidade dos documentos recuperados**
- ⚠️ **Leve redução na abrangência das respostas**

> Em resumo: o reranker melhora a **confiabilidade e precisão do sistema**, tornando as respostas mais seguras, mesmo com pequena perda de expressividade.

