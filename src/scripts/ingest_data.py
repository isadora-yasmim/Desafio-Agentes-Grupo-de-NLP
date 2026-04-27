"""
ingest_data.py
--------------
Script principal de ingestão. Orquestra todo o pipeline:

  JSON files → Parser → Chunker → Leitura de PDFs → Embedder → Qdrant

Uso:
    python scripts/ingest_data.py                    # usa settings padrão
    python scripts/ingest_data.py --dry-run          # sem inserir no Qdrant
    python scripts/ingest_data.py --backend openai   # força backend
    python scripts/ingest_data.py --clear            # limpa tabela antes
    python scripts/ingest_data.py --year 2016        # apenas um ano
"""

from __future__ import annotations

import argparse
import sys
import time

import logging

from pathlib import Path
import PyPDF2 
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Adiciona o root do projeto ao path
#sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import AneelJsonParser
from ingestion.chunker import AneelChunker, ChunkConfig, build_pdf_text_splitter
from ingestion.embedder import AneelEmbedder
from core.logger import get_logger


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_ingestion(
    data_dir: str = "base",
    backend: str = "openai",
    dry_run: bool = False,
    clear: bool = False,
    year_filter: str | None = None,
    batch_size: int = 20,
) -> dict:
    """
    Executa o pipeline completo de ingestão.

    Returns:
        dict com estatísticas: n_docs, n_chunks, elapsed_seconds
    """
    start = time.time()
    stats = {}

    # ── 1. Parser ────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("ETAPA 1: Parsing dos JSONs")
    logger.info("═" * 60)

    parser = AneelJsonParser()

    data_path = Path(__file__).resolve().parents[2] / data_dir

    # Filtra por ano se especificado
    if year_filter:
        json_files = list(data_path.glob(f"*{year_filter}*.json"))
        logger.info(f"Filtrando por ano {year_filter}: {len(json_files)} arquivo(s)")
        docs = []
        for f in json_files:
            docs.extend(parser.parse_file(f))
    else:
        docs = list(parser.parse_directory(data_path))

    stats["n_docs"] = len(docs)
    logger.info(f"Total de documentos parseados: {len(docs)}")

    if not docs:
        logger.warning("Nenhum documento encontrado. Verifique o diretório.")
        return stats

    # Estatísticas dos dados
    sem_ementa = sum(1 for d in docs if not d.ementa)
    logger.info(f"  → Com ementa:   {len(docs) - sem_ementa}")
    logger.info(f"  → Sem ementa:   {sem_ementa}")

    tipos = {}
    for d in docs:
        tipos[d.tipo_ato] = tipos.get(d.tipo_ato, 0) + 1
    logger.info(f"  → Por tipo: {dict(sorted(tipos.items(), key=lambda x: -x[1])[:5])}")

    # ── 2. Chunker ───────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("ETAPA 2: Chunking hierárquico (Ementas e Metadados)")
    logger.info("═" * 60)

    chunk_config = ChunkConfig(
    window_size=3,
    window_overlap=1,
    min_ementa_sentences=1,
    include_summary_chunk=True,
    include_window_chunks=True,
    include_keyword_chunk=True,
    )

    chunker = AneelChunker(config=chunk_config)
    chunks = chunker.chunk_documents(docs)

    logger.info("\n" + "═" * 60)
    logger.info("ETAPA 3: Extração e Chunking do texto completo dos PDFs")
    logger.info("═" * 60)

    pdf_splitter = build_pdf_text_splitter()
    pdf_chunks = []
    pdfs_dir = data_path / "pdfs"

    for doc in docs:
        if not doc.pdfs:
            continue

        for pdf_ref in doc.pdfs:
            if not pdf_ref.arquivo:
                continue

            # Tenta encontrar o PDF
            pdf_path = pdfs_dir / pdf_ref.arquivo
            if not pdf_path.exists():
                # Tenta na raiz do data_dir caso não exista a pasta 'pdfs'
                pdf_path = data_path / pdf_ref.arquivo
                if not pdf_path.exists():
                    continue

            try:
                # 1. Lê o PDF completo
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texto_completo = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texto_completo += page_text + "\n"

                if not texto_completo.strip():
                    continue

                # 2. Quebra o texto usando as regras jurídicas
                raw_chunks = pdf_splitter.split_text(texto_completo)

                # 3. Aplica o contexto no topo de cada chunk
                for i, trecho in enumerate(raw_chunks):
                    texto_final = (
                        f"Tipo do ato: {doc.tipo_ato}\n"
                        f"Título: {doc.titulo}\n"
                        f"---\n"
                        f"{trecho.strip()}"
                    )

                    pdf_chunks.append(Document(
                        page_content=texto_final,
                        metadata={
                            "doc_id": doc.doc_id,
                            "titulo": doc.titulo,
                            "tipo_ato": doc.tipo_ato,
                            "chunk_type": "full_pdf_text",
                            "chunk_index": i,
                            "arquivo_origem": pdf_ref.arquivo
                        }
                    ))
            except Exception as e:
                logger.error(f"Erro ao processar PDF {pdf_ref.arquivo}: {e}")

    logger.info(f"Gerados {len(pdf_chunks)} chunks a partir dos PDFs.")
    chunks.extend(pdf_chunks)
    stats["n_chunks"] = len(chunks)

    if dry_run:
        logger.info("\n[DRY RUN] Pulando inserção no Qdrant.")
        return stats

    # ── 3. Embedding + Upsert ────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info(f"ETAPA 3: Embedding ({backend}) + inserção no Qdrant")
    logger.info("═" * 60)

    embedder = AneelEmbedder(backend=backend, batch_size=batch_size)

    if clear:
        logger.warning("Limpando tabela antes da inserção...")
        embedder.clear_table()

    embedder.upsert(chunks)

    final_count = embedder.count_documents()
    stats["final_count"] = final_count
    logger.info(f"Total de documentos na tabela: {final_count}")

    # ── Resumo ───────────────────────────────────────────────────────────────
    stats["elapsed"] = time.time() - start
    logger.info("\n" + "═" * 60)
    logger.info("RESUMO DA INGESTÃO")
    logger.info("═" * 60)
    logger.info(f"  Documentos parseados:  {stats['n_docs']}")
    logger.info(f"  Chunks gerados:        {stats['n_chunks']}")
    logger.info(f"  Chunks no Qdrant:    {stats.get('final_count', 'N/A')}")
    logger.info(f"  Tempo total:           {stats['elapsed']:.1f}s")

    import pickle
    chunks_path = data_path / "chunks_for_bm25.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Chunks salvos para busca híbrida em: {chunks_path}")
    
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de ingestão ANEEL → Qdrant"
    )
    parser.add_argument(
        "--data-dir", default="base",
        help="Diretório com os JSONs (padrão: base/)"
    )
    parser.add_argument(
        "--backend", choices=["openai", "huggingface"], default="openai",
        help="Backend de embeddings"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Executa sem inserir no Qdrant"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Limpa a tabela antes de inserir"
    )
    parser.add_argument(
        "--year", default=None,
        help="Filtra apenas JSONs de um ano (ex: 2016)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Chunks por batch de upsert"
    )

    args = parser.parse_args()

    run_ingestion(
        data_dir=args.data_dir,
        backend=args.backend,
        dry_run=args.dry_run,
        clear=args.clear,
        year_filter=args.year,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()