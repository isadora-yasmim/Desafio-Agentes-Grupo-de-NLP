"""
chunker.py
----------
Transforma AneelDocuments em LangChain Documents prontos para embedding.

Estratégia hierárquica:

1. document_summary
   Chunk de visão geral do documento:
   tipo do ato + título + temas inferidos + ementa completa.

2. semantic_window
   Chunk mais fino:
   tipo do ato + título + temas inferidos + trecho da ementa.

3. keyword_context
   Chunk de enriquecimento semântico:
   tipo do ato + título + temas inferidos + termos relacionados do domínio.

Objetivo:
- Melhorar a recuperação semântica.
- Reduzir ambiguidade entre termos parecidos como TE, TEO, TUST, TUSD e TSEE.
- Ajudar queries do usuário que usam linguagem mais natural.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from core.logger import get_logger
from ingestion.parser import AneelDocument

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    window_size: int = 3
    window_overlap: int = 1
    min_ementa_sentences: int = 1

    include_summary_chunk: bool = True
    include_window_chunks: bool = True
    include_keyword_chunk: bool = True

    # Evita indexar chunks do tipo:
    # "Tipo do ato: DSP | Título: DSP - DESPACHO XXXX | Ementa: não disponível"
    skip_summary_without_ementa: bool = True


# ---------------------------------------------------------------------------
# Termos do domínio
# ---------------------------------------------------------------------------

DOMAIN_TERMS = {
    "bandeiras tarifárias": [
        "bandeira tarifária",
        "bandeiras tarifárias",
        "vermelha patamar",
        "vermelha patamar 1",
        "vermelha patamar 2",
        "amarela",
        "verde",
        "PRORET",
        "submódulo 6.8",
    ],
    "tarifa social / baixa renda": [
        "tarifa social",
        "Tarifa Social de Energia Elétrica",
        "TSEE",
        "baixa renda",
        "residencial baixa renda",
        "subclasse residencial baixa renda",
        "Conta de Desenvolvimento Energético",
        "CDE",
    ],
    "geração distribuída": [
        "geração distribuída",
        "microgeração distribuída",
        "minigeração distribuída",
        "MMGD",
        "sistema de compensação",
        "sistema de compensação de energia elétrica",
        "compensação de energia elétrica",
    ],
    "tarifas de energia": [
    "Tarifas de Energia",
    "Tarifa de Energia",
    "tarifa de energia elétrica",
    "tarifas de energia elétrica",
    "__TE_TARIFA_ENERGIA__",
    "TUSD",
    "TUST",
    "reajuste tarifário",
    "revisão tarifária",
    "tarifa de uso",
    "tarifas de uso",
    ],
    "transmissão": [
        "transmissão",
        "TUST",
        "Sistema Interligado Nacional",
        "SIN",
        "Rede Básica",
    ],
    "distribuição": [
        "distribuição",
        "TUSD",
        "serviço público de distribuição",
        "distribuidora",
        "distribuidoras",
    ],
    "regulação ANEEL": [
    "Resolução Normativa",
    "ato normativo",
    "consulta pública",
    "audiência pública",
    "Análise de Impacto Regulatório",
    "AIR",
    "regulação",
    "regulatório",
    ]
}


# ---------------------------------------------------------------------------
# Regex e normalização
# ---------------------------------------------------------------------------

_PROTECTED_PATTERNS = [
    r"S\.A\.",
    r"Ltda\.",
    r"SPE\s*S\.A\.",
    r"art\.",
    r"arts\.",
    r"inc\.",
    r"Res\.",
    r"RN\.",
    r"n[º°]\s*\d+",
    r"\d{1,2}\.\d{1,2}\.\d{4}",
    r"R\$\s*[\d\.,]+",
]

_SENTENCE_DELIMITERS = re.compile(r"(?<=[.;!?])\s+")


def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _split_sentences(text: str | None) -> list[str]:
    text = _normalize_text(text)

    if not text:
        return []

    placeholder_map: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        key = f"__PROTECTED_{len(placeholder_map)}__"
        placeholder_map[key] = match.group(0)
        return key

    protected = text

    for pattern in _PROTECTED_PATTERNS:
        protected = re.sub(pattern, protect, protected, flags=re.IGNORECASE)

    raw_sentences = _SENTENCE_DELIMITERS.split(protected)
    sentences: list[str] = []

    for sentence in raw_sentences:
        for key, value in placeholder_map.items():
            sentence = sentence.replace(key, value)

        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence)

    return sentences


def _sliding_windows(
    sentences: list[str],
    size: int,
    overlap: int,
) -> list[list[str]]:
    if not sentences:
        return []

    if size <= 0:
        raise ValueError("window_size deve ser maior que 0.")

    if overlap < 0:
        raise ValueError("window_overlap não pode ser negativo.")

    if overlap >= size:
        raise ValueError("window_overlap deve ser menor que window_size.")

    if len(sentences) <= size:
        return [sentences]

    step = max(1, size - overlap)
    windows: list[list[str]] = []

    for i in range(0, len(sentences), step):
        window = sentences[i : i + size]

        if window:
            windows.append(window)

    return windows


def _term_matches(text: str, term: str) -> bool:
    """
    Faz match seguro de termos do domínio.

    - Siglas curtas como TUST, TUSD, CDE e AIR precisam aparecer isoladas.
    - TE é tratada separadamente porque é muito ambígua.
    - Termos longos usam match por substring normalizado.
    """
    normalized_text = _normalize_text(text)
    normalized_term = _normalize_text(term)

    if not normalized_text or not normalized_term:
        return False

    # Regra especial para TE:
    # Só considera TE como Tarifa de Energia quando aparece perto de "Tarifa(s) de Energia".
    te_patterns = [
        r"Tarifas?\s+de\s+Energia\s*[-–—]?\s*TE\b",
        r"\bTE\b\s*[-–—]?\s*Tarifas?\s+de\s+Energia",
    ]

    if normalized_term == "__TE_TARIFA_ENERGIA__":
        return any(
            re.search(pattern, normalized_text, flags=re.IGNORECASE)
            for pattern in te_patterns
        )

    # Siglas curtas em maiúsculas: match por fronteira de palavra.
    # Evita TUST/TUSD/CDE baterem dentro de outras palavras.
    if normalized_term.isupper() and len(normalized_term) <= 5:
        pattern = rf"(?<![A-Za-zÀ-ÿ0-9]){re.escape(normalized_term)}(?![A-Za-zÀ-ÿ0-9])"
        return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None

    return normalized_term.lower() in normalized_text.lower()

def _infer_themes(text: str | None) -> list[str]:
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return []

    themes: list[str] = []

    for theme, terms in DOMAIN_TERMS.items():
        if any(_term_matches(normalized_text, term) for term in terms):
            themes.append(theme)

    return themes

def _matched_terms(text: str | None, themes: list[str]) -> list[str]:
    """
    Retorna apenas os termos do domínio que realmente aparecem no texto.

    Isso evita que o keyword_context adicione termos relacionados demais
    que não estão presentes no documento original.
    """
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return []

    matched: list[str] = []

    for theme in themes:
        for term in DOMAIN_TERMS.get(theme, []):
            if term.startswith("__"):
                continue

            if _term_matches(normalized_text, term):
                matched.append(term)

    return sorted(set(matched))

def _build_context_prefix(doc: AneelDocument, themes: list[str]) -> str:
    parts = [
        f"Tipo do ato: {doc.tipo_ato}",
        f"Título: {doc.titulo}",
    ]

    if themes:
        parts.append(f"Temas inferidos: {', '.join(themes)}")

    return "\n".join(parts)


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

        sentence_counter = Counter(
            len(_split_sentences(doc.ementa))
            for doc in docs
            if doc.ementa
        )

        chunk_type_counter = Counter(
            chunk.metadata.get("chunk_type", "unknown")
            for chunk in all_chunks
        )

        theme_counter = Counter(
            theme
            for chunk in all_chunks
            for theme in chunk.metadata.get("themes", [])
        )

        no_ementa_chunks = sum(
            1
            for chunk in all_chunks
            if chunk.metadata.get("has_ementa") is False
        )

        char_counts = [len(chunk.page_content) for chunk in all_chunks]

        logger.info(f"Distribuição de sentenças nas ementas: {dict(sentence_counter)}")
        logger.info(f"Distribuição de chunks por tipo: {dict(chunk_type_counter)}")
        logger.info(f"Top temas inferidos: {theme_counter.most_common(20)}")
        logger.info(f"Chunks sem ementa: {no_ementa_chunks}")

        if char_counts:
            avg_size = sum(char_counts) // len(char_counts)
            logger.info(
                "Tamanho dos chunks: "
                f"médio={avg_size} chars | "
                f"mínimo={min(char_counts)} chars | "
                f"máximo={max(char_counts)} chars"
            )

        logger.info(f"Chunking concluído: {len(docs)} docs → {len(all_chunks)} chunks")

        return all_chunks

    def _chunk_one(self, doc: AneelDocument) -> list[Document]:
        chunks: list[Document] = []

        ementa = _normalize_text(doc.ementa)
        sentences = _split_sentences(ementa)

        full_text_for_theme = " ".join(
            [
                str(doc.tipo_ato or ""),
                str(doc.titulo or ""),
                ementa,
            ]
        )

        themes = _infer_themes(full_text_for_theme)
        document_id = _hash(f"{doc.tipo_ato}|{doc.titulo}|{ementa}")

        base_meta = {
            **doc.to_metadata(),
            "tipo": doc.tipo_ato,
            "titulo": doc.titulo,
            "document_id": document_id,
            "themes": themes,
            "has_ementa": bool(ementa),
            "ementa_sentence_count": len(sentences),
        }

        # ------------------------------------------------------------------
        # 1. Chunk de resumo do documento
        # ------------------------------------------------------------------
        should_create_summary = (
            self.config.include_summary_chunk
            and (
                bool(ementa)
                or not self.config.skip_summary_without_ementa
            )
        )

        if should_create_summary:
            summary_text = self._build_summary_chunk(doc, ementa, themes)

            chunks.append(
                self._make_chunk(
                    page_content=summary_text,
                    base_meta=base_meta,
                    chunk_type="document_summary",
                    chunk_index=0,
                    chunk_total=1,
                )
            )

        # ------------------------------------------------------------------
        # 2. Chunks por janela semântica da ementa
        # ------------------------------------------------------------------
        if self.config.include_window_chunks and len(sentences) >= self.config.min_ementa_sentences:
            windows = _sliding_windows(
                sentences=sentences,
                size=self.config.window_size,
                overlap=self.config.window_overlap,
            )

            for idx, window in enumerate(windows):
                window_text = self._build_window_chunk(doc, window, themes)

                chunks.append(
                    self._make_chunk(
                        page_content=window_text,
                        base_meta=base_meta,
                        chunk_type="semantic_window",
                        chunk_index=idx,
                        chunk_total=len(windows),
                    )
                )

        # ------------------------------------------------------------------
        # 3. Chunk de termos relacionados do domínio
        # ------------------------------------------------------------------
        if self.config.include_keyword_chunk and themes and ementa:
            matched_terms = _matched_terms(full_text_for_theme, themes)

            if matched_terms:
                keyword_text = self._build_keyword_chunk(
                    doc=doc,
                    themes=themes,
                    matched_terms=matched_terms,
                )

                chunks.append(
                    self._make_chunk(
                        page_content=keyword_text,
                        base_meta={
                            **base_meta,
                            "matched_terms": matched_terms,
                        },
                        chunk_type="keyword_context",
                        chunk_index=len(chunks),
                        chunk_total=1,
                    )
                )

        return chunks

    def _make_chunk(
        self,
        page_content: str,
        base_meta: dict,
        chunk_type: str,
        chunk_index: int,
        chunk_total: int,
    ) -> Document:
        clean_content = _normalize_text(page_content)

        chunk_id = _hash(
            f"{base_meta.get('document_id')}|"
            f"{chunk_type}|"
            f"{chunk_index}|"
            f"{clean_content}"
        )

        return Document(
            page_content=clean_content,
            metadata={
                **base_meta,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
                "char_count": len(clean_content),
            },
        )

    def _build_summary_chunk(
        self,
        doc: AneelDocument,
        ementa: str,
        themes: list[str],
    ) -> str:
        parts = [_build_context_prefix(doc, themes)]

        if ementa:
            parts.append(f"Ementa: {ementa}")
        else:
            parts.append("Ementa: não disponível.")

        return "\n".join(parts)

    def _build_window_chunk(
        self,
        doc: AneelDocument,
        window: list[str],
        themes: list[str],
    ) -> str:
        prefix = _build_context_prefix(doc, themes)
        trecho = " ".join(window)

        return "\n".join(
            [
                prefix,
                f"Trecho da ementa: {trecho}",
            ]
        )

    def _build_keyword_chunk(
        self,
        doc: AneelDocument,
        themes: list[str],
        matched_terms: list[str],
    ) -> str:
        """
        Chunk conservador de apoio à recuperação.

        Ele não inventa todos os sinônimos do domínio.
        Usa apenas:
        - temas inferidos;
        - termos realmente encontrados no documento;
        - metadados básicos úteis.
        """
        parts = [
            f"Tipo do ato: {doc.tipo_ato}",
            f"Título: {doc.titulo}",
        ]

        if themes:
            parts.append(f"Temas inferidos: {', '.join(themes)}")

        if matched_terms:
            parts.append(f"Termos encontrados no documento: {', '.join(matched_terms)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Splitter para PDFs
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