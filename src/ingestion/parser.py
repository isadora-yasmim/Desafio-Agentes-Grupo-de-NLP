import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ParsedDocument:
    title: str
    text: str
    doc_type: str          # "norma", "relatorio", "tarifa", "ata", "outro"
    source_file: str
    metadata: dict = field(default_factory=dict)


# ── Entrada principal ────────────────────────────────────────────

def parse_file(path: str | Path) -> list[ParsedDocument]:
    """Ponto de entrada: recebe um .json e devolve documentos limpos."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Suporta tanto um objeto único quanto uma lista
    items = raw if isinstance(raw, list) else [raw]

    docs = []
    for item in items:
        doc = _parse_item(item, source_file=path.name)
        if doc:
            docs.append(doc)
    return docs


def parse_directory(folder: str | Path) -> Iterator[ParsedDocument]:
    """Processa todos os JSONs de uma pasta."""
    for path in Path(folder).glob("**/*.json"):
        yield from parse_file(path)


# ── Roteador por tipo ────────────────────────────────────────────

def _parse_item(item: dict, source_file: str) -> ParsedDocument | None:
    doc_type = _detect_type(item)

    parsers = {
        "norma":    _parse_norma,
        "relatorio": _parse_relatorio,
        "tarifa":   _parse_tarifa,
        "ata":      _parse_ata,
    }

    parser_fn = parsers.get(doc_type, _parse_generic)
    return parser_fn(item, source_file, doc_type)


# ── Detecção de tipo ─────────────────────────────────────────────

_NORMA_KEYWORDS    = {"resolução", "portaria", "instrução normativa",
                      "decreto", "lei", "ren", "aneel", "artigo", "art."}
_RELATORIO_KEYWORDS = {"relatório", "resultado", "desempenho", "indicador",
                       "dec", "fec", "drc"}
_TARIFA_KEYWORDS   = {"tarifa", "tusd", "tust", "te ", "bandeira",
                      "subgrupo", "modalidade tarifária"}
_ATA_KEYWORDS      = {"ata", "reunião", "deliberação", "voto"}

def _detect_type(item: dict) -> str:
    # Junta todos os valores string do item para inspecionar
    haystack = " ".join(
        str(v).lower()
        for v in item.values()
        if isinstance(v, str)
    )

    scores = {
        "norma":    sum(1 for k in _NORMA_KEYWORDS    if k in haystack),
        "relatorio": sum(1 for k in _RELATORIO_KEYWORDS if k in haystack),
        "tarifa":   sum(1 for k in _TARIFA_KEYWORDS   if k in haystack),
        "ata":      sum(1 for k in _ATA_KEYWORDS      if k in haystack),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "outro"


# ── Parsers específicos ──────────────────────────────────────────

def _parse_norma(item: dict, source: str, doc_type: str) -> ParsedDocument:
    title = (
        item.get("titulo")
        or item.get("title")
        or item.get("numero")
        or "Norma sem título"
    )

    # Monta texto preservando a hierarquia dos artigos
    sections = []

    if ementa := item.get("ementa"):
        sections.append(f"Ementa: {_clean(ementa)}")

    # Suporta tanto lista de artigos quanto texto corrido
    if artigos := item.get("artigos") or item.get("articles"):
        for art in artigos:
            if isinstance(art, dict):
                num  = art.get("numero") or art.get("number") or ""
                body = art.get("texto") or art.get("text") or ""
                sections.append(f"Art. {num} {_clean(body)}")
            else:
                sections.append(_clean(str(art)))
    elif corpo := item.get("texto") or item.get("content") or item.get("body"):
        sections.append(_clean(corpo))

    return ParsedDocument(
        title       = _clean(title),
        text        = "\n\n".join(sections),
        doc_type    = doc_type,
        source_file = source,
        metadata    = {
            "numero":    item.get("numero") or item.get("number"),
            "data":      item.get("data")   or item.get("date"),
            "orgao":     item.get("orgao")  or item.get("organ"),
            "vigencia":  item.get("vigencia"),
        },
    )


def _parse_relatorio(item: dict, source: str, doc_type: str) -> ParsedDocument:
    title = (
        item.get("titulo") or item.get("title")
        or f"Relatório — {item.get('periodo') or item.get('ano') or source}"
    )

    sections = []

    if resumo := item.get("resumo") or item.get("summary"):
        sections.append(f"Resumo: {_clean(resumo)}")

    # Indicadores numéricos — preserva unidade junto ao valor
    if indicadores := item.get("indicadores") or item.get("indicators"):
        lines = []
        if isinstance(indicadores, dict):
            for k, v in indicadores.items():
                lines.append(f"{k}: {v}")
        elif isinstance(indicadores, list):
            for ind in indicadores:
                if isinstance(ind, dict):
                    nome  = ind.get("nome") or ind.get("name") or ""
                    valor = ind.get("valor") or ind.get("value") or ""
                    unidade = ind.get("unidade") or ind.get("unit") or ""
                    lines.append(f"{nome}: {valor} {unidade}".strip())
        if lines:
            sections.append("Indicadores:\n" + "\n".join(lines))

    if corpo := item.get("texto") or item.get("content") or item.get("body"):
        sections.append(_clean(corpo))

    return ParsedDocument(
        title       = _clean(title),
        text        = "\n\n".join(sections),
        doc_type    = doc_type,
        source_file = source,
        metadata    = {
            "periodo": item.get("periodo") or item.get("periodo_referencia"),
            "empresa": item.get("empresa") or item.get("distribuidora"),
            "ano":     item.get("ano"),
        },
    )


def _parse_tarifa(item: dict, source: str, doc_type: str) -> ParsedDocument:
    title = (
        item.get("titulo") or item.get("title")
        or f"Tarifa — {item.get('subgrupo') or item.get('modalidade') or source}"
    )

    sections = []

    # Tabelas de tarifa: serializa preservando contexto de cada linha
    tabela = (
        item.get("tabela") or item.get("table")
        or item.get("valores") or item.get("values")
    )
    if tabela:
        sections.append(_serialize_table(tabela))

    if vigencia := item.get("vigencia") or item.get("periodo_vigencia"):
        sections.append(f"Vigência: {vigencia}")

    if corpo := item.get("texto") or item.get("descricao") or item.get("content"):
        sections.append(_clean(corpo))

    return ParsedDocument(
        title       = _clean(title),
        text        = "\n\n".join(sections),
        doc_type    = doc_type,
        source_file = source,
        metadata    = {
            "subgrupo":  item.get("subgrupo"),
            "modalidade": item.get("modalidade"),
            "vigencia":  item.get("vigencia"),
            "distribuidora": item.get("distribuidora") or item.get("empresa"),
        },
    )


def _parse_ata(item: dict, source: str, doc_type: str) -> ParsedDocument:
    title = (
        item.get("titulo") or item.get("title")
        or f"Ata — {item.get('data') or source}"
    )

    sections = []

    if pauta := item.get("pauta") or item.get("agenda"):
        if isinstance(pauta, list):
            sections.append("Pauta:\n" + "\n".join(f"- {p}" for p in pauta))
        else:
            sections.append(f"Pauta: {_clean(str(pauta))}")

    if deliberacoes := item.get("deliberacoes") or item.get("decisoes"):
        if isinstance(deliberacoes, list):
            lines = []
            for d in deliberacoes:
                if isinstance(d, dict):
                    lines.append(
                        f"- {d.get('numero') or ''}: "
                        f"{_clean(d.get('texto') or d.get('descricao') or '')}"
                    )
                else:
                    lines.append(f"- {_clean(str(d))}")
            sections.append("Deliberações:\n" + "\n".join(lines))

    if corpo := item.get("texto") or item.get("content"):
        sections.append(_clean(corpo))

    return ParsedDocument(
        title       = _clean(title),
        text        = "\n\n".join(sections),
        doc_type    = doc_type,
        source_file = source,
        metadata    = {
            "data":    item.get("data"),
            "orgao":   item.get("orgao"),
            "numero":  item.get("numero"),
        },
    )


def _parse_generic(item: dict, source: str, doc_type: str) -> ParsedDocument:
    """Fallback: extrai todos os campos string recursivamente."""
    title = (
        item.get("titulo") or item.get("title")
        or item.get("nome") or item.get("name")
        or source
    )
    text = _extract_all_text(item)

    return ParsedDocument(
        title       = _clean(str(title)),
        text        = text,
        doc_type    = "outro",
        source_file = source,
        metadata    = {"raw_keys": list(item.keys())},
    )


# ── Utilitários ──────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove espaços extras, quebras múltiplas e lixo de encoding."""
    text = re.sub(r"\s+", " ", text)          # espaços múltiplos
    text = re.sub(r"\n{3,}", "\n\n", text)    # quebras excessivas
    text = text.replace("\xa0", " ")          # non-breaking space
    text = text.replace("\x00", "")           # null bytes
    return text.strip()


def _serialize_table(table) -> str:
    """Serializa lista/dict de tabela em texto legível pelo LLM."""
    if isinstance(table, list) and table:
        if isinstance(table[0], dict):
            headers = list(table[0].keys())
            rows = [headers] + [[str(row.get(h, "")) for h in headers] for row in table]
            return "\n".join(" | ".join(row) for row in rows)
        return "\n".join(str(row) for row in table)

    if isinstance(table, dict):
        return "\n".join(f"{k}: {v}" for k, v in table.items())

    return str(table)


def _extract_all_text(obj, depth: int = 0) -> str:
    """Extrai recursivamente todo texto de um objeto JSON."""
    if depth > 5:
        return ""
    if isinstance(obj, str):
        return _clean(obj)
    if isinstance(obj, dict):
        return "\n".join(
            _extract_all_text(v, depth + 1)
            for v in obj.values()
            if v
        )
    if isinstance(obj, list):
        return "\n".join(
            _extract_all_text(i, depth + 1)
            for i in obj
            if i
        )
    return str(obj)