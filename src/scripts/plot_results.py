from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("eval/results")

OUTPUT_DIR = Path(
    r"D:\Grupo NLP\Desafio Agente RAG\Desafio-Agentes-Grupo-de-NLP\docs\Evaluation\Gráficos"
)

COLOR_MAP = {
    "answer_relevancy": "#2563EB",
    "faithfulness": "#16A34A",
    "context_precision": "#9333EA",
}

METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
}

SEGMENT_LABELS = {
    "difficulty": "Dificuldade",
    "category": "Categoria",
    "domain": "Domínio",
}


def load_latest_segmented() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("ragas_segmented_summary_*.csv"))

    if not files:
        raise FileNotFoundError("Nenhum arquivo segmentado encontrado.")

    latest_file = files[-1]
    print(f"📂 Usando arquivo: {latest_file}")

    return pd.read_csv(latest_file)


def format_filename(metric: str, segment_type: str) -> str:
    return f"{metric}_por_{segment_type}.png"


def plot_metric(df: pd.DataFrame, segment_type: str, metric: str) -> None:
    filtered = df[df["segment_type"] == segment_type].copy()

    if filtered.empty:
        print(f"⚠️ Nenhum dado para {segment_type}")
        return

    filtered = filtered.sort_values(by=metric, ascending=False)

    metric_label = METRIC_LABELS.get(metric, metric)
    segment_label = SEGMENT_LABELS.get(segment_type, segment_type)
    color = COLOR_MAP.get(metric, "#374151")

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        filtered["segment"].astype(str),
        filtered[metric],
        color=color,
        alpha=0.88,
        edgecolor="#111827",
        linewidth=0.7,
    )

    plt.title(
        f"{metric_label} por {segment_label}",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    plt.xlabel(segment_label, fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12, labelpad=10)

    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    filepath = OUTPUT_DIR / format_filename(metric, segment_type)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 Gráfico salvo em: {filepath}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_latest_segmented()

    metrics = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
    ]

    segment_types = [
        "difficulty",
        "category",
        "domain",
    ]

    for segment_type in segment_types:
        for metric in metrics:
            plot_metric(df, segment_type, metric)


if __name__ == "__main__":
    main()