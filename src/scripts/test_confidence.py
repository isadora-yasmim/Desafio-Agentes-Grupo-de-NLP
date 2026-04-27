from answering.answerer import Answerer

answerer = Answerer()


def run_test(query, chunks):
    print("=" * 80)
    print(f"QUERY: {query}\n")

    result = answerer.answer(query, chunks)

    print("TIPO:", result["type"])
    print("CONFIANÇA:", result["confidence"])
    print("FINAL SCORE:", result.get("final_score"))
    print("\nRESPOSTA:\n")
    print(result["answer"])
    print("\nFONTES:")
    print(result["sources"])
    print("=" * 80)


# -----------------------------
# CASO 1 — ALTA CONFIANÇA
# -----------------------------
chunks_high = [
    {
        "content": "A tarifa social de energia elétrica concede desconto para consumidores de baixa renda.",
        "metadata": {"title": "Lei 12.212", "tipo_ato": "Lei", "final_score": 0.85},
        "score": 0.85,
    }
]

# -----------------------------
# CASO 2 — MÉDIA CONFIANÇA
# -----------------------------
chunks_medium = [
    {
        "content": "Existe desconto tarifário em alguns casos específicos.",
        "metadata": {"title": "Documento genérico", "tipo_ato": "Nota", "final_score": 0.55},
        "score": 0.55,
    }
]

# -----------------------------
# CASO 3 — BAIXA CONFIANÇA
# -----------------------------
chunks_low = [
    {
        "content": "Texto pouco relacionado.",
        "metadata": {"title": "Outro tema", "tipo_ato": "Nota", "final_score": 0.20},
        "score": 0.20,
    }
]


run_test("o que é tarifa social?", chunks_high)
run_test("explique desconto tarifário", chunks_medium)
run_test("qual o valor exato da tarifa social em 2022?", chunks_low)