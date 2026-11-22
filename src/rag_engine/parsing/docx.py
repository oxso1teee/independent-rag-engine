"""DOCX parser (optional extra ``docx``): headings, paragraphs, lists, quotes, code and tables.

The archive is checked before anything is read (entry count, uncompressed size, compression
ratio), so a zip bomb is refused cheaply. Text is taken from the document body in order: styles
decide headings (``Title``, ``Heading 1``..), list items, quotes and code; tables become rows of
cells joined by `` | ``. Tracked insertions are included, deletions, text boxes and the
alternative rendering of drawings are not. Headers, footers, footnotes and comments are not read.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import UTC, datetime
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from rag_engine.domain import BlockKind, ParsedDocument, ParseReason, RagError, parse_failed
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import CleanerSet, normalizer_version
from rag_engine.parsing.common import check_input_size, clean_metadata
from rag_engine.ports import ParseHints, ParseLimits

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # how Word stores a password-protected file

_T, _TAB, _BR, _CR, _HYPHEN = (qn(n) for n in ("w:t", "w:tab", "w:br", "w:cr", "w:noBreakHyphen"))
_P, _TBL, _TR, _TC = (qn(n) for n in ("w:p", "w:tbl", "w:tr", "w:tc"))
_SKIPPED = frozenset(
    {
        qn("w:del"),
        qn("w:txbxContent"),
        "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback",
    }
)
_HEADING = re.compile(r"^heading\s*(\d)$")
_TRAILING_NUMBER = re.compile(r"(\d)$")


def check_archive(data: bytes, limits: ParseLimits) -> zipfile.ZipFile:
    """The opened archive, or ``E_PARSE_FAILED`` for anything damaged, protected or oversized."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        if data.startswith(_OLE_MAGIC):
            raise parse_failed(
                ParseReason.ENCRYPTED, "the document is protected by a password"
            ) from None
        raise parse_failed(ParseReason.CORRUPT, "the file is not a valid DOCX archive") from None
    infos = archive.infolist()
    total = sum(info.file_size for info in infos)
    if len(infos) > limits.max_archive_entries:
        raise parse_failed(
            ParseReason.ZIP_BOMB,
            "the archive has too many entries",
            limit=limits.max_archive_entries,
        )
    if total > limits.max_archive_bytes:
        raise parse_failed(
            ParseReason.ZIP_BOMB,
            "the archive expands beyond the allowed size",
            limit=limits.max_archive_bytes,
        )
    if total > limits.max_archive_ratio * max(len(data), 1):
        raise parse_failed(
            ParseReason.ZIP_BOMB,
            "the archive is compressed far beyond what a document is",
            limit=limits.max_archive_ratio,
        )
    return archive


def _text(element: Any, parts: list[str]) -> None:
    for child in element:
        tag = child.tag
        if tag in _SKIPPED:
            continue
        if tag == _T:
            parts.append(child.text or "")
        elif tag == _TAB:
            parts.append("\t")
        elif tag in (_BR, _CR):
            parts.append("\n")
        elif tag == _HYPHEN:
            parts.append("-")
        elif isinstance(tag, str):
            _text(child, parts)


def paragraph_text(element: Any) -> str:
    parts: list[str] = []
    _text(element, parts)
    return "".join(parts)


def _table_rows(table: Any, depth: int, limits: ParseLimits) -> list[str]:
    if depth > limits.max_nesting:
        raise parse_failed(ParseReason.TOO_DEEP, "tables are nested too deeply")
    rows: list[str] = []
    for row in table.iterchildren(_TR):
        cells = []
        for cell in row.iterchildren(_TC):
            parts: list[str] = []
            for item in cell:
                if item.tag == _P:
                    parts.append(paragraph_text(item))
                elif item.tag == _TBL:
                    parts.extend(_table_rows(item, depth + 1, limits))
            cells.append(" ".join(" ".join(parts).split()))
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _created(document: Any) -> datetime | None:
    try:
        moment = document.core_properties.created
    except Exception:
        return None
    if not isinstance(moment, datetime):
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _classify(paragraph: Paragraph, under_title: bool) -> tuple[BlockKind, int | None]:
    try:
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
    except Exception:
        style = ""
    name = style.strip().lower()
    if name == "title":
        return BlockKind.HEADING, 1
    heading = _HEADING.match(name)
    if heading:
        number = int(heading.group(1)) + (1 if under_title else 0)
        return BlockKind.HEADING, max(1, min(number, 6))
    properties = paragraph._p.pPr
    numbering: Any = properties.numPr if properties is not None else None
    if numbering is not None or name.startswith("list"):
        depth = (
            numbering.ilvl.val + 1 if numbering is not None and numbering.ilvl is not None else 1
        )
        trailing = _TRAILING_NUMBER.search(name)
        return BlockKind.LIST_ITEM, int(trailing.group(1)) if trailing else max(1, depth)
    if "quote" in name:
        return BlockKind.QUOTE, None
    if "code" in name or "preformatted" in name or "source" in name:
        return BlockKind.CODE, None
    return BlockKind.PARAGRAPH, None


def _is_title(paragraph: Paragraph) -> bool:
    try:
        style = paragraph.style
        return style is not None and (style.name or "").strip().lower() == "title"
    except Exception:
        return False


class DocxParser:
    parser_id = "docx"
    version = "1"

    def __init__(self, cleaners: CleanerSet | None = None) -> None:
        self._cleaners = cleaners or CleanerSet()

    @property
    def normalizer_version(self) -> str:
        return normalizer_version(self._cleaners)

    def supports(self, mime: str | None, extension: str | None) -> bool:
        if mime is not None:
            return mime == DOCX_MIME
        return (extension or "").lower() == "docx"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        try:
            return self._parse(data, hints.limits)
        except RagError:
            raise
        except RecursionError:
            raise parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply") from None
        except MemoryError:
            raise parse_failed(ParseReason.MEMORY, "the document needs too much memory") from None
        except Exception:
            raise parse_failed(ParseReason.CORRUPT, "the document could not be read") from None

    def _parse(self, data: bytes, limits: ParseLimits) -> ParsedDocument:
        archive = check_archive(data, limits)
        if "word/document.xml" not in archive.namelist():
            raise parse_failed(ParseReason.CORRUPT, "the archive is not a Word document")
        document = Document(io.BytesIO(data))
        assembler = TextAssembler(limits, self._cleaners)
        pending: tuple[BlockKind, list[str]] | None = None
        under_title = False  # a Title paragraph puts the Heading styles one level below it

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                kind, lines = pending
                assembler.add(kind, "\n".join(lines))
                pending = None

        for element in document.element.body.iterchildren():
            if element.tag == _TBL:
                flush()
                rows = _table_rows(element, 1, limits)
                if rows:
                    columns = max(row.count(" | ") + 1 for row in rows)
                    attrs = {"rows": len(rows), "columns": columns}
                    assembler.add(BlockKind.TABLE, "\n".join(rows), attrs=attrs)
            elif element.tag == _P:
                text = paragraph_text(element)
                kind, level = _classify(Paragraph(element, document), under_title)
                if kind is BlockKind.HEADING and level == 1 and not under_title:
                    under_title = _is_title(Paragraph(element, document))
                if kind in (BlockKind.QUOTE, BlockKind.CODE):
                    if pending is not None and pending[0] is not kind:
                        flush()
                    if pending is None:
                        pending = (kind, [])
                    pending[1].append(text)
                    continue
                flush()
                assembler.add(kind, text, level=level)
        flush()
        properties = document.core_properties
        blocks = assembler.build().blocks
        first_heading = next((b.text for b in blocks if b.kind is BlockKind.HEADING), None)
        return assembler.parsed(
            title=clean_metadata(properties.title) or first_heading,
            author=clean_metadata(properties.author),
            created=_created(document),
            warnings=() if blocks else ("empty_document",),
        )
