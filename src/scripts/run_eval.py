from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from agent.agent import build_agent

import argparse

ROOT_DIR = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT_DIR /"eval" / "questions.json"
RESULTS_DIR = ROOT_DIR / "eval" / "results"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_contexts(result: dict) -> list[str]:
    sources = result.get("sources") or result.get("documents") or result.get("contexts") or []

    contexts = []

    for source in sources:
        if isinstance(source, str):
            contexts.append(source)
        elif isinstance(source, dict):
            content = (
                source.get("content")
                or source.get("page_content")
                or source.get("text")
                or source.get("chunk")
                or ""
            )
            if content:
                contexts.append(content)

    return contexts


def run_pipeline(agent, question: str) -> dict:
    result = agent.invoke(
        {
            "question": question,
            "top_k": 5,
            "use_reranker": True,
        }
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    contexts = [
        source.get("content", "")
        for source in sources
        if source.get("content")
    ]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    questions = load_questions()
    agent = build_agent()

    for item in questions:
        question = item["question"]
        print(f"🔎 Avaliando: {question}")

        row = run_pipeline(agent, question)
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

    scores_path = RESULTS_DIR / f"ragas_scores_{timestamp}.csv"
    raw_path = RESULTS_DIR / f"ragas_raw_{timestamp}.csv"

    scores_df.to_csv(scores_path, index=False, encoding="utf-8")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")

    print("\n✅ Avaliação finalizada.")
    print(f"📊 Scores salvos em: {scores_path}")
    print(f"📄 Respostas salvas em: {raw_path}")
    print("\nResumo:")
    print(scores_df.mean(numeric_only=True))


if __name__ == "__main__":
    main()