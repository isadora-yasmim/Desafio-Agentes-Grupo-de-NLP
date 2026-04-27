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
    sources = (
        result.get("sources")
        or result.get("documents")
        or result.get("contexts")
        or []
    )

    contexts: list[str] = []

    for source in sources:
        if isinstance(source, str) and source.strip():
            contexts.append(source.strip())
            continue

        if not isinstance(source, dict):
            continue

        metadata = source.get("metadata") or {}

        content = (
            source.get("content")
            or source.get("page_content")
            or source.get("text")
            or source.get("chunk")
            or metadata.get("content")
            or metadata.get("page_content")
            or metadata.get("text")
            or metadata.get("chunk")
            or metadata.get("ementa")
            or metadata.get("summary")
            or ""
        )

        if content and str(content).strip():
            contexts.append(str(content).strip())

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
        "confidence": result.get("confidence"),
        "final_score": result.get("final_score"),
        "used_rag": result.get("used_rag"),
        "type": result.get("type"),
    }


def build_summary_df(
    scores_df: pd.DataFrame,
    mode: str,
    top_k: int,
    confidence_accuracy: float | None,
    type_accuracy: float | None,
) -> pd.DataFrame:
    summary_df = (
        scores_df.mean(numeric_only=True)
        .reset_index()
        .rename(columns={"index": "metric", 0: "score"})
    )

    summary_df.insert(0, "mode", mode)
    summary_df.insert(1, "top_k", top_k)

    extra_metrics = []

    if confidence_accuracy is not None:
        extra_metrics.append(
            {
                "mode": mode,
                "top_k": top_k,
                "metric": "confidence_accuracy",
                "score": confidence_accuracy,
            }
        )

    if type_accuracy is not None:
        extra_metrics.append(
            {
                "mode": mode,
                "top_k": top_k,
                "metric": "type_accuracy",
                "score": type_accuracy,
            }
        )

    if extra_metrics:
        summary_df = pd.concat(
            [summary_df, pd.DataFrame(extra_metrics)],
            ignore_index=True,
        )

    return summary_df


def build_segmented_summary_df(
    raw_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    mode: str,
    top_k: int,
) -> pd.DataFrame:
    evaluation_df = pd.concat(
        [
            raw_df.reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ],
        axis=1,
    )

    segmented_rows = []

    for column in ["difficulty", "category", "domain"]:
        if column not in evaluation_df.columns:
            continue

        if evaluation_df[column].isna().all():
            continue

        grouped = (
            evaluation_df
            .groupby(column, dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )

        grouped.insert(0, "mode", mode)
        grouped.insert(1, "top_k", top_k)
        grouped.insert(2, "segment_type", column)

        grouped = grouped.rename(columns={column: "segment"})

        segmented_rows.append(grouped)

    if not segmented_rows:
        return pd.DataFrame()

    return pd.concat(segmented_rows, ignore_index=True)


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
        row["expected_confidence"] = item.get("expected_confidence", "")
        row["expected_type"] = item.get("expected_type", "")
        row["difficulty"] = item.get("difficulty", "")
        row["category"] = item.get("category", "")
        row["domain"] = item.get("domain", "")

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

    if "expected_confidence" in raw_df.columns:
        raw_df["confidence_match"] = (
            raw_df["confidence"] == raw_df["expected_confidence"]
        )
        confidence_accuracy = raw_df["confidence_match"].mean()
    else:
        confidence_accuracy = None

    if "expected_type" in raw_df.columns:
        raw_df["type_match"] = raw_df["type"] == raw_df["expected_type"]
        type_accuracy = raw_df["type_match"].mean()
    else:
        type_accuracy = None

    summary_df = build_summary_df(
        scores_df=scores_df,
        mode=mode,
        top_k=top_k,
        confidence_accuracy=confidence_accuracy,
        type_accuracy=type_accuracy,
    )

    segmented_summary_df = build_segmented_summary_df(
        raw_df=raw_df,
        scores_df=scores_df,
        mode=mode,
        top_k=top_k,
    )

    scores_path = RESULTS_DIR / f"ragas_scores_{mode}_{timestamp}.csv"
    raw_path = RESULTS_DIR / f"ragas_raw_{mode}_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"ragas_summary_{mode}_{timestamp}.csv"
    segmented_summary_path = (
        RESULTS_DIR / f"ragas_segmented_summary_{mode}_{timestamp}.csv"
    )

    scores_df.to_csv(scores_path, index=False, encoding="utf-8")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    if not segmented_summary_df.empty:
        segmented_summary_df.to_csv(
            segmented_summary_path,
            index=False,
            encoding="utf-8",
        )

    print("\n✅ Avaliação finalizada.")
    print(f"📊 Scores salvos em: {scores_path}")
    print(f"📄 Respostas salvas em: {raw_path}")
    print(f"📈 Resumo salvo em: {summary_path}")

    if not segmented_summary_df.empty:
        print(f"📚 Resumo segmentado salvo em: {segmented_summary_path}")

    if confidence_accuracy is not None:
        print(f"\n🎯 Accuracy da confiança: {confidence_accuracy:.2%}")

    if type_accuracy is not None:
        print(f"🏷️ Accuracy do tipo de resposta: {type_accuracy:.2%}")

    print("\nResumo:")
    print(summary_df)

    if not segmented_summary_df.empty:
        print("\nResumo segmentado:")
        print(segmented_summary_df)


if __name__ == "__main__":
    main()