from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("docs/Evaluation/Graficos")

METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
}


def load_latest_summaries() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("ragas_summary_*.csv"))

    if not files:
        raise FileNotFoundError("Nenhum arquivo summary encontrado.")

    rows = []

    for file in files:
        df = pd.read_csv(file)

        label = (
            "Com reranker"
            if "reranker" in file.name and "no_reranker" not in file.name
            else "Sem reranker"
        )

        pivot = df.pivot_table(
            index=[],
            columns="metric",
            values="score",
            aggfunc="mean",
        ).reset_index(drop=True)

        row = pivot.iloc[0].to_dict()
        row["config"] = label
        row["file"] = file.name
        rows.append(row)

        print(f"📂 Usando arquivo: {file}")

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]

    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        raise ValueError(f"Nenhuma métrica esperada encontrada. Colunas disponíveis: {list(df.columns)}")

    plot_df = df[["config"] + available_metrics].set_index("config")

    # 🎨 CORES DO SLIDE
    colors = [
        "#5B3A8E",  # roxo
        "#F97316",  # laranja
        "#FBBF24",  # dourado
    ]

    plt.figure(figsize=(10, 6))
    ax = plot_df.plot(
        kind="bar",
        color=colors,
        width=0.7,
    )

    # 🎨 Fundo clean
    plt.gca().set_facecolor("#F9FAFB")
    plt.gcf().patch.set_facecolor("#F9FAFB")

    # 🎯 Título
    plt.title(
        "Comparação RAGAS — Sem reranker vs Com reranker",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        pad=16,
    )

    plt.xlabel("")
    plt.ylabel("Score", fontsize=12)

    plt.ylim(0, 1.05)
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=10)

    # Grid suave
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    # Legenda
    plt.legend(
        [METRIC_LABELS.get(m, m) for m in available_metrics],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    # Labels nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")

    plt.tight_layout()

    output_file = OUTPUT_DIR / "comparacao_ragas_reranker.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 Gráfico salvo em: {output_file}")


def main() -> None:
    df = load_latest_summaries()
    plot_comparison(df)


if __name__ == "__main__":
    main()