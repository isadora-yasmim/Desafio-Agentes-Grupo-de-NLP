"""
ingestion/parser.py
-------------------
Lê os JSONs da ANEEL e normaliza cada registro em AneelDocument.

Correções em relação à versão anterior:
  - _safe_str(): trata campos que existem no JSON mas têm valor null (None).
    dict.get("campo", "") só usa o fallback quando a chave NÃO existe;
    quando a chave existe com valor None, retorna None mesmo.
    Isso causava os erros "expected string, got NoneType" e
    "'NoneType' object has no attribute 'strip'".
  - _normalize_date(): agora aceita None sem quebrar.
  - _parse_registro(): usa _safe_str() em todos os campos de texto.
  - Documentos sem título real (None/vazio) recebem título gerado
    a partir do tipo do ato + data, em vez de serem descartados.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Modelo de domínio
# ---------------------------------------------------------------------------

@dataclass
class PdfRef:
    tipo: str
    url: str
    arquivo: str
    baixado: bool


@dataclass
class AneelDocument:
    titulo: str
    autor: str
    material: str
    data_publicacao: str
    data_assinatura: str
    esfera: str
    situacao: str
    assunto: str
    ementa: str | None
    pdfs: list[PdfRef] = field(default_factory=list)
    numeracao_item: str = ""
    data_chave: str = ""

    @property
    def doc_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]", "_", self.titulo.lower())
        return f"{self.data_publicacao}__{slug}"

    @property
    def tipo_ato(self) -> str:
        match = re.match(r"^([A-Z]+)\s*-", self.titulo)
        return match.group(1) if match else "OUTROS"

    @property
    def numero_ato(self) -> str:
        match = re.search(r"(\d{3,5}/\d{4})", self.titulo)
        return match.group(1) if match else ""

    @property
    def text_content(self) -> str:
        parts = [
            self.titulo,
            f"Autor: {self.autor}",
            f"Assunto: {self.assunto}",
            f"Situação: {self.situacao}",
        ]
        if self.ementa:
            parts.append(f"Ementa: {self.ementa}")
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        return {
            "doc_id":           self.doc_id,
            "titulo":           self.titulo,
            "autor":            self.autor,
            "material":         self.material,
            "tipo_ato":         self.tipo_ato,
            "numero_ato":       self.numero_ato,
            "data_publicacao":  self.data_publicacao,
            "data_assinatura":  self.data_assinatura,
            "esfera":           self.esfera,
            "situacao":         self.situacao,
            "assunto":          self.assunto,
            "tem_ementa":       self.ementa is not None,
            "qtd_pdfs":         len(self.pdfs),
            "arquivos_pdf":     [p.arquivo for p in self.pdfs],
            "data_chave":       self.data_chave,
        }


# ---------------------------------------------------------------------------
# Helpers None-safe
# ---------------------------------------------------------------------------

def _safe_str(value, fallback: str = "") -> str:
    """
    Converte qualquer valor para str de forma segura.

    O problema: dict.get("campo", "fallback") SÓ usa o fallback quando
    a chave não existe no dicionário. Quando a chave existe mas vale None
    (JSON: null), get() retorna None — e chamadas como .strip() ou re.sub()
    explodem com TypeError/AttributeError.

    Esta função resolve os três casos:
      - chave ausente    → get() retorna ""  → _safe_str("") → ""
      - chave com None   → get() retorna None → _safe_str(None) → fallback
      - chave com string → get() retorna str  → _safe_str(str) → str limpo
    """
    if value is None:
        return fallback
    return str(value).strip()


def _clean_field(value, prefix: str = "") -> str:
    """Remove prefixo e artefatos textuais. None-safe."""
    text = _safe_str(value)
    if not text:
        return ""
    cleaned = text.replace(prefix, "").strip()
    cleaned = re.sub(r"\s*Imprimir\s*$", "", cleaned).strip()
    return cleaned


def _normalize_date(raw) -> str:
    """
    Converte 'DD/MM/YYYY' ou 'Prefixo:DD/MM/YYYY' → 'YYYY-MM-DD'.
    Aceita None sem quebrar.
    """
    text = _safe_str(raw)
    if not text:
        return ""
    # Remove prefixo como "Assinatura:", "Publicação:", etc.
    text = re.sub(r"^[^:]+:", "", text).strip()
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        d, m, y = match.groups()
        return f"{y}-{m}-{d}"
    return text


def _parse_pdfs(raw_pdfs) -> list[PdfRef]:
    """Parseia lista de PDFs com tolerância a None e campos ausentes."""
    if not raw_pdfs or not isinstance(raw_pdfs, list):
        return []
    result = []
    for p in raw_pdfs:
        if not isinstance(p, dict):
            continue
        result.append(PdfRef(
            tipo=_safe_str(p.get("tipo")),
            url=_safe_str(p.get("url")),
            arquivo=_safe_str(p.get("arquivo")),
            baixado=bool(p.get("baixado", False)),
        ))
    return result


def _build_fallback_titulo(reg: dict, data_chave: str) -> str:
    """
    Gera um título descritivo quando o campo titulo é None/vazio.
    Ex: "DSP SEM TÍTULO - 2021-03-15"
    Evita descartar o documento só porque falta o título.
    """
    numeracao = _safe_str(reg.get("numeracaoItem"), "?")
    material  = _safe_str(reg.get("material"), "DOCUMENTO")
    return f"{material} SEM TÍTULO (item {numeracao}) - {data_chave}"


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class AneelJsonParser:

    def parse_file(self, path: str | Path) -> Iterator[AneelDocument]:
        path = Path(path)
        logger.info(f"Parseando arquivo: {path.name}")

        with open(path, encoding="utf-8") as f:
            data: dict = json.load(f)

        total = 0
        skipped = 0
        recovered = 0

        for data_chave, dia_data in data.items():
            registros = dia_data.get("registros", [])
            if not registros:
                continue

            for reg in registros:
                if not isinstance(reg, dict):
                    skipped += 1
                    continue
                try:
                    doc, was_recovered = self._parse_registro(reg, data_chave)
                    total += 1
                    if was_recovered:
                        recovered += 1
                    yield doc
                except Exception as e:
                    skipped += 1
                    titulo_raw = reg.get("titulo", "?")
                    logger.warning(
                        f"Skipping '{titulo_raw}' [{data_chave}]: {type(e).__name__}: {e}"
                    )

        suffix = f", {recovered} recuperados com título gerado" if recovered else ""
        logger.info(
            f"{path.name}: {total} docs parseados{suffix}, {skipped} ignorados."
        )

    def parse_directory(self, directory: str | Path) -> Iterator[AneelDocument]:
        directory = Path(directory)
        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            logger.warning(f"Nenhum JSON encontrado em: {directory}")
            return
        for json_file in json_files:
            yield from self.parse_file(json_file)

    # -----------------------------------------------------------------------

    def _parse_registro(self, reg: dict, data_chave: str) -> tuple[AneelDocument, bool]:
        """
        Retorna (AneelDocument, was_recovered).
        was_recovered=True quando o título foi gerado por fallback.
        """
        titulo_raw = reg.get("titulo")
        titulo = _safe_str(titulo_raw)
        was_recovered = False

        # Título ausente ou nulo: gera fallback em vez de descartar
        if not titulo:
            titulo = _build_fallback_titulo(reg, data_chave)
            was_recovered = True

        doc = AneelDocument(
            titulo=titulo,
            autor=_safe_str(reg.get("autor")),
            material=_safe_str(reg.get("material")),
            data_publicacao=_normalize_date(reg.get("publicacao")),
            data_assinatura=_normalize_date(reg.get("assinatura")),
            esfera=_clean_field(reg.get("esfera"),   "Esfera:"),
            situacao=_clean_field(reg.get("situacao"), "Situação:"),
            assunto=_clean_field(reg.get("assunto"),  "Assunto:"),
            ementa=_clean_field(reg.get("ementa")) or None,
            pdfs=_parse_pdfs(reg.get("pdfs")),
            numeracao_item=_safe_str(reg.get("numeracaoItem")),
            data_chave=data_chave,
        )
        return doc, was_recovered
