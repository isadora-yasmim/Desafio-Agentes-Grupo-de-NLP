from retrieval.qdrant_retriever import QdrantRetriever
from answering import Answerer


def main():
    query = "como funciona a tarifa de energia"

    retriever = QdrantRetriever()
    answerer = Answerer()

    chunks = retriever.search(query)

    answer = answerer.answer(query=query, chunks=chunks)

    print(answer)


if __name__ == "__main__":
    main()