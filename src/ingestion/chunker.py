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
from typing import Literal

from langchain_core.documents import Document

from src.ingestion.parser import AneelDocument
from src.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    # Nível 2: janela de sentenças
    window_size: int = 3          # sentenças por chunk fino
    window_overlap: int = 1       # sobreposição de sentenças
    min_ementa_sentences: int = 2 # só quebra se ementa tiver mais que N sentenças
    # Inclui chunk de nível 1 (documento inteiro)?
    include_full_doc_chunk: bool = True
    # Inclui chunks de janela?
    include_window_chunks: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_DELIMITERS = re.compile(r"(?<=[.;!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Divide texto em sentenças usando pontuação como delimitador."""
    raw = _SENTENCE_DELIMITERS.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _sliding_windows(sentences: list[str], size: int, overlap: int) -> list[list[str]]:
    """Gera janelas deslizantes de sentenças."""
    if len(sentences) <= size:
        return [sentences]
    step = max(1, size - overlap)
    windows = []
    i = 0
    while i < len(sentences):
        windows.append(sentences[i : i + size])
        i += step
    return windows


def _build_prefix(doc: AneelDocument) -> str:
    """Prefixo contextual adicionado a TODOS os chunks (parent doc context)."""
    return (
        f"Tipo: {doc.tipo_ato} | "
        f"Número: {doc.numero_ato} | "
        f"Autor: {doc.autor} | "
        f"Assunto: {doc.assunto} | "
        f"Publicação: {doc.data_publicacao}"
    )


# ---------------------------------------------------------------------------
# Chunker principal
# ---------------------------------------------------------------------------

class AneelChunker:
    """
    Converte uma lista de AneelDocuments em LangChain Documents.

    Uso:
        chunker = AneelChunker()
        lc_docs = chunker.chunk_documents(aneel_docs)
    """

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def chunk_documents(self, docs: list[AneelDocument]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            all_chunks.extend(self._chunk_one(doc))
        logger.info(f"Chunking concluído: {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _chunk_one(self, doc: AneelDocument) -> list[Document]:
        chunks: list[Document] = []
        base_meta = doc.to_metadata()
        prefix = _build_prefix(doc)

        # ── Nível 1: chunk do documento completo ───────────────────────────
        if self.config.include_full_doc_chunk:
            full_text = self._build_full_text(doc, prefix)
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

        # ── Nível 2: chunks por janela de sentenças da ementa ──────────────
        if self.config.include_window_chunks and doc.ementa:
            sentences = _split_sentences(doc.ementa)

            # Só quebra em janelas se tiver sentenças suficientes
            if len(sentences) >= self.config.min_ementa_sentences:
                windows = _sliding_windows(
                    sentences,
                    self.config.window_size,
                    self.config.window_overlap,
                )
                for idx, window in enumerate(windows):
                    window_text = (
                        f"{prefix}\n"
                        f"Título: {doc.titulo}\n"
                        f"Trecho da ementa: {' '.join(window)}"
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

    def _build_full_text(self, doc: AneelDocument, prefix: str) -> str:
        """Monta o texto completo do chunk de nível 1."""
        parts = [
            prefix,
            f"Título: {doc.titulo}",
            f"Situação: {doc.situacao}",
        ]
        if doc.ementa:
            parts.append(f"Ementa: {doc.ementa}")
        else:
            # Documentos sem ementa: usa apenas os metadados disponíveis
            parts.append("(Ementa não disponível no metadado)")

        # Lista os arquivos PDF relacionados para facilitar busca por arquivo
        if doc.pdfs:
            arquivos = ", ".join(p.arquivo for p in doc.pdfs)
            parts.append(f"Arquivos PDF: {arquivos}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Estratégia de chunking para textos longos (PDFs extraídos)
# ---------------------------------------------------------------------------
# Quando você extrair o texto dos PDFs, use o splitter abaixo.
# Ele usa RecursiveCharacterTextSplitter com separadores jurídicos.

def build_pdf_text_splitter():
    """
    Splitter para textos completos de PDFs extraídos.
    Usa separadores relevantes para documentos jurídicos/regulatórios.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\nArt.",    # artigos de normas
            "\n\n§",       # parágrafos
            "\n\nI -",     # incisos
            "\n\n",        # quebra de parágrafo
            "\n",
            ". ",
            " ",
        ],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )


# ---------------------------------------------------------------------------
# CLI para testar
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.ingestion.parser import AneelJsonParser

    path = sys.argv[1] if len(sys.argv) > 1 else "base"
    parser = AneelJsonParser()
    docs = list(parser.parse_directory(path))

    chunker = AneelChunker()
    chunks = chunker.chunk_documents(docs[:5])  # testa com 5 docs

    print(f"\nTotal chunks gerados: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n── Chunk {i} [{c.metadata['chunk_type']}] ──")
        print(c.page_content[:300])
        print(f"  metadata: doc_id={c.metadata['doc_id']}")
