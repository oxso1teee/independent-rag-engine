"""Plain-text parser: paragraphs separated by blank lines."""

from __future__ import annotations

from rag_engine.domain import BlockKind, ParsedDocument
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import CleanerSet, normalizer_version
from rag_engine.parsing.common import check_input_size, decode_text, split_paragraphs
from rag_engine.ports import ParseHints

_EXTENSIONS = frozenset({"txt", "text", "log"})


class TxtParser:
    parser_id = "txt"
    version = "1"

    def __init__(self, cleaners: CleanerSet | None = None) -> None:
        self._cleaners = cleaners or CleanerSet()

    @property
    def normalizer_version(self) -> str:
        return normalizer_version(self._cleaners)

    def supports(self, mime: str | None, extension: str | None) -> bool:
        if mime is not None:
            return mime == "text/plain"
        return (extension or "").lower() in _EXTENSIONS

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        check_input_size(data, hints)
        text, warnings = decode_text(data, hints)
        assembler = TextAssembler(hints.limits, self._cleaners)
        for paragraph in split_paragraphs(text):
            assembler.add(BlockKind.PARAGRAPH, paragraph)
        document = assembler.parsed(warnings=tuple(warnings))
        if not document.blocks:
            return assembler.parsed(warnings=(*warnings, "empty_document"))
        return document
