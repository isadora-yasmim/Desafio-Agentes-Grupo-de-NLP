"""
chunker.py
----------
Transforma AneelDocuments em LangChain Documents (chunks) prontos
para embedding.

Estratégia hierárquica em dois níveis:

  Nível 1 – "Document chunk" (chunk principal)
  ─────────────────────────────────────────────
  Um chunk por documento ANEEL. Contém título + autor + assunto +
  ementa completa. Ideal para perguntas de alto nível ("O que diz
  o despacho X?").

  Nível 2 – "Sentence / window chunk" (chunks finos)
  ───────────────────────────────────────────────────
  Quebra a ementa em janelas de sentenças com overlap. Ideal para
  perguntas factuais precisas ("Quais municípios foram citados?").

Ambos os níveis carregam os mesmos metadados + um campo extra
`chunk_type` e `chunk_index` para rastreabilidade.

Por que não usar RecursiveCharacterTextSplitter direto?
- Os documentos são curtos (ementa ≤ ~500 tokens).
- A fronteira natural são as sentenças da ementa.
- O chunk de nível 1 garante que o contexto completo sempre seja
  recuperável, mesmo quando o chunk fino acerta no reranker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from ingestion.parser import AneelDocument
from core.logger import get_logger

logger = get_logger(__name__)

from collections import Counter


# ---------------------------------------------------------------------------
# Config otimizada
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    window_size: int = 2
    window_overlap: int = 1
    min_ementa_sentences: int = 2
    include_full_doc_chunk: bool = True
    include_window_chunks: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_DELIMITERS = re.compile(r"(?<=[.;!?])\s+")


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_DELIMITERS.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _sliding_windows(sentences: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(sentences) <= size:
        return [sentences]

    step = max(1, size - overlap)
    windows = []
    i = 0

    while i < len(sentences):
        windows.append(sentences[i : i + size])
        i += step

    return windows


# ---------------------------------------------------------------------------
# Chunker principal
# ---------------------------------------------------------------------------

class AneelChunker:

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def chunk_documents(self, docs: list[AneelDocument]) -> list[Document]:
        all_chunks: list[Document] = []

        

        for doc in docs:
            all_chunks.extend(self._chunk_one(doc))

        counter = Counter(
        len(_split_sentences(doc.ementa))
        for doc in docs
        if doc.ementa
        )
        logger.info(f"Distribuição de sentenças nas ementas: {dict(counter)}")

        logger.info(f"Chunking concluído: {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks

    # -----------------------------------------------------------------------

    def _chunk_one(self, doc: AneelDocument) -> list[Document]:
        
        chunks: list[Document] = []
        base_meta = {
            **doc.to_metadata(),
            "tipo": doc.tipo_ato,
            "titulo": doc.titulo,
        }

        # ── Nível 1: full_doc (otimizado) ────────────────────────────────
        if self.config.include_full_doc_chunk:
            full_text = self._build_full_text(doc)

            chunks.append(
                Document(
                    page_content=full_text,
                    metadata={
                        **base_meta,
                        "chunk_type": "full_doc",
                        "chunk_index": 0,
                        "chunk_total": 1,
                    },
                )
            )

        # ── Nível 2: window chunks (sem prefixo redundante) ──────────────
        if self.config.include_window_chunks and doc.ementa:
            sentences = _split_sentences(doc.ementa)

            if len(sentences) >= self.config.min_ementa_sentences:
                windows = _sliding_windows(
                    sentences,
                    self.config.window_size,
                    self.config.window_overlap,
                )

                for idx, window in enumerate(windows):
                    window_text = (
                        f"Título: {doc.titulo}\n"
                        f"{' '.join(window)}"
                    )

                    chunks.append(
                        Document(
                            page_content=window_text,
                            metadata={
                                **base_meta,
                                "chunk_type": "window",
                                "chunk_index": idx,
                                "chunk_total": len(windows),
                            },
                        )
                    )

        return chunks

    # -----------------------------------------------------------------------

    def _build_full_text(self, doc: AneelDocument) -> str:
        """
        Versão enxuta do chunk completo:
        - foca no que realmente importa semanticamente
        - reduz ruído no embedding
        """

        parts = [
            f"Tipo: {doc.tipo_ato}",
            f"Título: {doc.titulo}",
        ]

        if doc.ementa:
            parts.append(f"Ementa: {doc.ementa}")
        else:
            parts.append("(Ementa não disponível)")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Splitter para PDFs (mantido, já estava bom)
# ---------------------------------------------------------------------------

def build_pdf_text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\nArt.",
            "\n\n§",
            "\n\nI -",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )