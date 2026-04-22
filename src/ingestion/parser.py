"""
parser.py
---------
Responsável por ler os JSONs da ANEEL e normalizar cada registro
em um objeto estruturado (AneelDocument) pronto para o chunker.

Estrutura do JSON de entrada:
{
  "2016-12-30": {
    "status": "23 registro(s).",
    "registros": [
      {
        "titulo": "DSP - DESPACHO 3284/2016",
        "autor": "ANEEL",
        "esfera": "Esfera:Outros",
        "situacao": "Situação:NÃO CONSTA REVOGAÇÃO EXPRESSA",
        "assinatura": "Assinatura:15/12/2016",
        "publicacao": "Publicação:30/12/2016",
        "assunto": "Assunto:Acatamento",
        "ementa": "...",
        "pdfs": [{"tipo": "...", "url": "...", "arquivo": "...", "baixado": true}]
      }
    ]
  }
}
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
    """Representa um único ato normativo/despacho da ANEEL."""

    # Identificação
    titulo: str
    autor: str
    material: str

    # Datas (normalizadas para YYYY-MM-DD)
    data_publicacao: str
    data_assinatura: str

    # Classificação
    esfera: str
    situacao: str
    assunto: str

    # Conteúdo principal
    ementa: str | None

    # Referências aos PDFs
    pdfs: list[PdfRef] = field(default_factory=list)

    # Metadados extras
    numeracao_item: str = ""
    data_chave: str = ""          # data original do dict raiz (ex: "2016-12-30")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def doc_id(self) -> str:
        """ID único: data_publicacao + titulo normalizado."""
        slug = re.sub(r"[^a-z0-9]", "_", self.titulo.lower())
        return f"{self.data_publicacao}__{slug}"

    @property
    def tipo_ato(self) -> str:
        """Extrai o tipo de ato do título: DSP, RES, NOR, etc."""
        match = re.match(r"^([A-Z]+)\s*-", self.titulo)
        return match.group(1) if match else "OUTROS"

    @property
    def numero_ato(self) -> str:
        """Extrai o número do ato (ex: '3284/2016')."""
        match = re.search(r"(\d{3,5}/\d{4})", self.titulo)
        return match.group(1) if match else ""

    @property
    def text_content(self) -> str:
        """Texto consolidado do documento para embedding/BM25."""
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
        """Dicionário de metadados para armazenar no vector store."""
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
# Funções de limpeza
# ---------------------------------------------------------------------------

def _clean_field(value: str | None, prefix: str = "") -> str:
    """Remove prefixos como 'Situação:', 'Esfera:', etc. e faz strip."""
    if not value:
        return ""
    cleaned = value.replace(prefix, "").strip()
    # Remove o "Imprimir" que aparece no final de algumas ementas
    cleaned = re.sub(r"\s*Imprimir\s*$", "", cleaned).strip()
    return cleaned


def _normalize_date(raw: str) -> str:
    """
    Converte datas do formato 'DD/MM/YYYY' ou 'Assinatura:DD/MM/YYYY'
    para 'YYYY-MM-DD'. Retorna string vazia se não conseguir parsear.
    """
    # Remove prefixo
    raw = re.sub(r"^[^:]+:", "", raw).strip()
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", raw)
    if match:
        d, m, y = match.groups()
        return f"{y}-{m}-{d}"
    return raw


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class AneelJsonParser:
    """
    Parseia um ou vários arquivos JSON da ANEEL e emite AneelDocuments.

    Uso:
        parser = AneelJsonParser()
        for doc in parser.parse_file("base/legislacao_2016.json"):
            print(doc.titulo)
    """

    def parse_file(self, path: str | Path) -> Iterator[AneelDocument]:
        path = Path(path)
        logger.info(f"Parseando arquivo: {path.name}")

        with open(path, encoding="utf-8") as f:
            data: dict = json.load(f)

        total = 0
        skipped = 0

        for data_chave, dia_data in data.items():
            registros = dia_data.get("registros", [])
            if not registros:
                continue

            for reg in registros:
                try:
                    doc = self._parse_registro(reg, data_chave)
                    total += 1
                    yield doc
                except Exception as e:
                    skipped += 1
                    logger.warning(
                        f"Skipping registro '{reg.get('titulo', '?')}': {e}"
                    )

        logger.info(
            f"{path.name}: {total} documentos parseados, {skipped} ignorados."
        )

    def parse_directory(self, directory: str | Path) -> Iterator[AneelDocument]:
        """Parseia todos os JSONs de um diretório."""
        directory = Path(directory)
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            logger.warning(f"Nenhum JSON encontrado em: {directory}")
            return

        for json_file in json_files:
            yield from self.parse_file(json_file)

    # -----------------------------------------------------------------------
    # Internos
    # -----------------------------------------------------------------------

    def _parse_registro(self, reg: dict, data_chave: str) -> AneelDocument:
        pdfs = [
            PdfRef(
                tipo=p.get("tipo", ""),
                url=p.get("url", ""),
                arquivo=p.get("arquivo", ""),
                baixado=p.get("baixado", False),
            )
            for p in reg.get("pdfs", [])
        ]

        return AneelDocument(
            titulo=reg.get("titulo", "").strip(),
            autor=reg.get("autor", "").strip(),
            material=reg.get("material", "").strip(),
            data_publicacao=_normalize_date(reg.get("publicacao", "")),
            data_assinatura=_normalize_date(reg.get("assinatura", "")),
            esfera=_clean_field(reg.get("esfera", ""), "Esfera:"),
            situacao=_clean_field(reg.get("situacao", ""), "Situação:"),
            assunto=_clean_field(reg.get("assunto", ""), "Assunto:"),
            ementa=_clean_field(reg.get("ementa")) or None,
            pdfs=pdfs,
            numeracao_item=reg.get("numeracaoItem", ""),
            data_chave=data_chave,
        )


# ---------------------------------------------------------------------------
# CLI rápida para testar
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "base"
    parser = AneelJsonParser()

    docs = list(parser.parse_directory(path))
    print(f"\nTotal de documentos: {len(docs)}")
    print("\nExemplo:")
    if docs:
        d = docs[0]
        print(f"  titulo:          {d.titulo}")
        print(f"  tipo_ato:        {d.tipo_ato}")
        print(f"  numero_ato:      {d.numero_ato}")
        print(f"  data_publicacao: {d.data_publicacao}")
        print(f"  assunto:         {d.assunto}")
        print(f"  tem_ementa:      {d.ementa is not None}")
        print(f"  doc_id:          {d.doc_id}")
        print(f"\n  text_content:\n{d.text_content}")
