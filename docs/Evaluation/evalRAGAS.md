# 📊 Avaliação do Sistema RAG (RAGAS)

Este documento apresenta a avaliação do sistema utilizando **RAGAS**, comparando o desempenho **com e sem reranker**.

---

## 📈 Comparação de Métricas

| Métrica               | Sem reranker | Com reranker | Variação |
| --------------------- | ------------ | ------------ | -------- |
| **Faithfulness**      | 0.79         | **0.85**     | ⬆️ +0.06 |
| **Answer Relevancy**  | 0.84         | **0.89**     | ⬆️ +0.05 |
| **Context Precision** | 0.83         | **0.86**     | ⬆️ +0.03 |

---

## 📊 Visualização

![Comparação RAGAS](comparacao_ragas_reranker.png)

---

## 🧠 Análise dos Resultados

### 🟢 Faithfulness (+0.06)

* Aumentou de forma consistente
* Indica maior fidelidade das respostas ao contexto
* Redução de alucinações

**Interpretação:**
O reranker ajudou o modelo a selecionar documentos mais confiáveis.

---

### 🟢 Answer Relevancy (+0.05)

* Melhora significativa
* Respostas mais alinhadas com a pergunta

**Interpretação:**
O reranker aumentou a relevância das respostas, sem comprometer precisão.

---

### 🟢 Context Precision (+0.03)

* Melhora na qualidade do contexto recuperado
* Menos ruído na recuperação

**Interpretação:**
O pipeline híbrido + reranker melhora a seleção dos documentos.

---

## 🚀 Conclusão

A introdução do reranker trouxe melhorias em **todas as métricas avaliadas**:

* ✅ **Mais fidelidade ao contexto (menos alucinação)**
* ✅ **Respostas mais relevantes**
* ✅ **Melhor qualidade dos documentos recuperados**

> Em resumo: o reranker tornou o sistema **mais confiável, preciso e relevante**, sem trade-offs negativos.

---

## 🏁 Observação importante

Comparado aos experimentos iniciais, o sistema evoluiu significativamente:

* Melhorias no parsing e limpeza dos dados
* Ajustes no retrieval híbrido
* Correções na pipeline de ranking

👉 Resultado: ganho consistente em todas as métricas RAGAS

---
