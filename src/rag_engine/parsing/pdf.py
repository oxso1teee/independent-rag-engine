"""PDF parser (optional extra ``pdf``): text layer to page-numbered paragraphs.

Only the text layer is read; a PDF without one (a scan) is refused with ``no_text_layer``.
Lines of a page are joined into paragraphs by layout heuristics (line length, closing
punctuation, list markers), words broken by a hyphen at the end of a line are re-joined,
typographic ligatures are expanded, and lines that repeat at the top or bottom of most pages
(running titles, page numbers) are dropped. Every block carries its page number.
"""

from __future__ import annotations

import io
import itertools
import re
import time
from collections import Counter
from datetime import UTC, datetime

from pypdf import PdfReader

from rag_engine.domain import BlockKind, ParsedDocument, ParseReason, RagError, parse_failed
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import CleanerSet, normalizer_version
from rag_engine.parsing.common import check_input_size, clean_metadata
from rag_engine.ports import ParseHints

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE_TABLE = str.maketrans(_LIGATURES)
_LIST_MARKER = re.compile(r"^(?:[•●▪■◦‣–—*-]\s|\d{1,3}[.)]\s|[a-zа-я][.)]\s)")
_SENTENCE_END = re.compile(r"[.!?…:;»”\"')\]]$")
_HYPHEN_BREAK = re.compile(r"[^\W\d_]-$")
_MIN_PAGES_FOR_FURNITURE = 3
_FURNITURE_LINES = 2  # lines examined at the top and at the bottom of every page
_FURNITURE_SHARE = 0.6


def _page_lines(text: str, cleaners: CleanerSet) -> list[str]:
    """Non-empty lines of a page; source cleaners work on these original lines."""
    cleaned = cleaners.apply(text.translate(_LIGATURE_TABLE))
    return [line.strip() for line in cleaned.split("\n") if line.strip()]


def _edge(lines: list[str]) -> int:
    """How many lines at the top and at the bottom may be furniture: short pages give one."""
    return _FURNITURE_LINES if len(lines) >= 3 * _FURNITURE_LINES else 1


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def _furniture(pages: list[list[str]]) -> set[str]:
    """Lines (with digits blurred) that open or close at least 60 % of the pages."""
    if len(pages) < _MIN_PAGES_FOR_FURNITURE:
        return set()
    seen: Counter[str] = Counter()
    for lines in pages:
        edge = {
            _furniture_key(line) for line in (*lines[:_FURNITURE_LINES], *lines[-_FURNITURE_LINES:])
        }
        seen.update(edge)
    needed = max(_MIN_PAGES_FOR_FURNITURE, round(_FURNITURE_SHARE * len(pages)))
    return {key for key, count in seen.items() if count >= needed}


def _strip_furniture(lines: list[str], furniture: set[str]) -> list[str]:
    if not furniture:
        return lines
    kept: list[str] = []
    n = _edge(lines)
    for index, line in enumerate(lines):
        at_edge = index < n or index >= len(lines) - n
        if not (at_edge and _furniture_key(line) in furniture):
            kept.append(line)
    return kept


def _starts_paragraph(previous: str, line: str, width: int) -> bool:
    if _LIST_MARKER.match(line):
        return True
    if _HYPHEN_BREAK.search(previous):
        return False
    new_sentence = not line[0].islower()
    if _SENTENCE_END.search(previous) and len(previous) < 0.9 * width and new_sentence:
        return True
    short = len(previous) < 0.5 * width and not previous.endswith((",", ";", "-"))
    return short and line[0].isupper()


def _join(previous: str, line: str) -> str:
    if _HYPHEN_BREAK.search(previous) and line[:1].islower():
        return previous[:-1] + line
    return f"{previous} {line}"


def paragraphs_of(lines: list[str]) -> list[str]:
    """Join the lines of one page into paragraphs."""
    if not lines:
        return []
    ordered = sorted(len(line) for line in lines)
    width = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    result = [lines[0]]
    for previous, line in itertools.pairwise(lines):
        if _starts_paragraph(previous, line, width):
            result.append(line)
        else:
            result[-1] = _join(result[-1], line)
    return result


def _created(reader: PdfReader) -> datetime | None:
    try:
        moment = reader.metadata.creation_date if reader.metadata else None
    except Exception:
        return None
    if moment is None:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class PdfParser:
    parser_id = "pdf"
    version = "1"

    def __init__(self, cleaners: CleanerSet | None = None) -> None:
        self._cleaners = cleaners or CleanerSet()

    @property
    def normalizer_version(self) -> str:
        return normalizer_version(self._cleaners)

    def supports(self, mime: str | None, extension: str | None) -> bool:
        if mime is not None:
            return mime == "application/pdf"
        return (extension or "").lower() == "pdf"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        try:
            return self._parse(data, hints)
        except RagError:
            raise
        except RecursionError:
            raise parse_failed(ParseReason.TOO_DEEP, "the PDF is nested too deeply") from None
        except MemoryError:
            raise parse_failed(ParseReason.MEMORY, "the PDF needs too much memory") from None
        except Exception:
            raise parse_failed(ParseReason.CORRUPT, "the PDF could not be read") from None

    def _parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        limits = hints.limits
        deadline = time.monotonic() + limits.timeout_s
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise parse_failed(ParseReason.ENCRYPTED, "the PDF is protected by a password")
        count = len(reader.pages)
        if count > limits.max_pages:
            raise parse_failed(
                ParseReason.TOO_MANY_PAGES,
                "the PDF has more pages than allowed",
                pages=count,
                limit=limits.max_pages,
            )
        pages: list[list[str]] = []
        unreadable = 0
        for page in reader.pages:
            if time.monotonic() > deadline:
                raise parse_failed(ParseReason.TIMEOUT, "reading the PDF took too long")
            try:
                pages.append(_page_lines(page.extract_text() or "", self._cleaners))
            except (RecursionError, MemoryError):
                raise
            except Exception:
                pages.append([])
                unreadable += 1
        if unreadable == count:
            raise parse_failed(ParseReason.CORRUPT, "no page of the PDF could be read")
        furniture = _furniture(pages)
        assembler = TextAssembler(limits, self._cleaners)
        empty = 0
        for number, lines in enumerate(pages, start=1):
            blocks = paragraphs_of(_strip_furniture(lines, furniture))
            added = [assembler.add(BlockKind.PARAGRAPH, text, page=number) for text in blocks]
            if not any(added):
                empty += 1
        if not assembler.build().blocks:
            raise parse_failed(
                ParseReason.NO_TEXT_LAYER, "the PDF has no text layer (a scan needs OCR)"
            )
        warnings = []
        if empty - unreadable > 0:
            warnings.append(f"pages_without_text:{empty - unreadable}")
        if unreadable:
            warnings.append(f"pages_unreadable:{unreadable}")
        if furniture:
            warnings.append("headers_footers_removed")
        info = reader.metadata
        return assembler.parsed(
            title=clean_metadata(info.title if info else None),
            author=clean_metadata(info.author if info else None),
            created=_created(reader),
            warnings=tuple(warnings),
        )
