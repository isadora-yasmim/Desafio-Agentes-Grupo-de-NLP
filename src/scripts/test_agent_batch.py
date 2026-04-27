from agent.agent import RegulatoryAgent


QUERIES = [
    "quais documentos falam sobre tarifa social",
    "o que é tarifa social",
    "qual o valor exato da tarifa social em 2022",
    "quais documentos falam sobre um tema inexistente",
    "o que é TUSD",
]


def main():
    agent = RegulatoryAgent()

    for index, query in enumerate(QUERIES, start=1):
        print("=" * 100)
        print(f"TESTE {index}/{len(QUERIES)}")
        print(f"QUERY: {query}")
        print("=" * 100)

        result = agent.answer(
            query=query,
            top_k=5,
            use_reranker=True,
        )

        print("\nRESPOSTA:\n")
        print(result.get("answer"))

        print("\nFONTES:")
        for source in result.get("sources", []):
            metadata = source.get("metadata", {})
            print(
                "-",
                metadata.get("title")
                or metadata.get("titulo")
                or metadata.get("document_title")
                or "Documento sem título",
            )
            print("  Score:", source.get("score"))

        print("\n\n")


if __name__ == "__main__":
    main()