from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from agent.agent import build_agent


ROOT_DIR = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT_DIR / "eval" / "questions.json"
RESULTS_DIR = ROOT_DIR / "eval" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa avaliação do pipeline RAG usando RAGAS."
    )

    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Ativa o reranker durante a recuperação dos documentos.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Quantidade de documentos recuperados por pergunta.",
    )

    return parser.parse_args()


def load_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de perguntas não encontrado: {QUESTIONS_PATH}"
        )

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("O arquivo questions.json está vazio.")

    return questions


def extract_contexts(result: dict[str, Any]) -> list[str]:
    sources = result.get("sources") or result.get("documents") or result.get("contexts") or []

    contexts: list[str] = []

    for source in sources:
        if isinstance(source, str) and source.strip():
            contexts.append(source.strip())

        elif isinstance(source, dict):
            content = (
                source.get("content")
                or source.get("page_content")
                or source.get("text")
                or source.get("chunk")
                or ""
            )

            if content and content.strip():
                contexts.append(content.strip())

    return contexts


def run_pipeline(
    agent: Any,
    question: str,
    top_k: int,
    use_reranker: bool,
) -> dict[str, Any]:
    result = agent.invoke(
        {
            "question": question,
            "top_k": top_k,
            "use_reranker": use_reranker,
        }
    )

    answer = result.get("answer", "")
    contexts = extract_contexts(result)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


def main() -> None:
    args = parse_args()

    use_reranker = args.reranker
    top_k = args.top_k
    mode = "reranker" if use_reranker else "no_reranker"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    agent = build_agent()

    rows: list[dict[str, Any]] = []

    print("\n🚀 Iniciando avaliação com RAGAS")
    print(f"🔁 Reranker: {'ativado' if use_reranker else 'desativado'}")
    print(f"📌 Top-k: {top_k}")
    print(f"📄 Total de perguntas: {len(questions)}\n")

    for item in questions:
        question = item["question"]
        print(f"🔎 Avaliando: {question}")

        row = run_pipeline(
            agent=agent,
            question=question,
            top_k=top_k,
            use_reranker=use_reranker,
        )

        row["reference"] = item.get("reference", "")
        rows.append(row)

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scores_df = result.to_pandas()
    raw_df = pd.DataFrame(rows)

    scores_path = RESULTS_DIR / f"ragas_scores_{mode}_{timestamp}.csv"
    raw_path = RESULTS_DIR / f"ragas_raw_{mode}_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"ragas_summary_{mode}_{timestamp}.csv"

    scores_df.to_csv(scores_path, index=False, encoding="utf-8")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")

    summary_df = (
        scores_df.mean(numeric_only=True)
        .reset_index()
        .rename(columns={"index": "metric", 0: "score"})
    )

    summary_df.insert(0, "mode", mode)
    summary_df.insert(1, "top_k", top_k)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n✅ Avaliação finalizada.")
    print(f"📊 Scores salvos em: {scores_path}")
    print(f"📄 Respostas salvas em: {raw_path}")
    print(f"📈 Resumo salvo em: {summary_path}")

    print("\nResumo:")
    print(summary_df)


if __name__ == "__main__":
    main()