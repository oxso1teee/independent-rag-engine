"""Builds ``normalized_text`` and its blocks together, so offsets are right by construction.

Parsers feed blocks in document order; the assembler normalises each one, drops the empty
ones, tracks the heading path and the page, enforces the limits and records the position of
every block in the final text. Blocks are separated by one blank line.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from rag_engine.domain import (
    Block,
    BlockKind,
    Language,
    ParsedDocument,
    ParseReason,
    detect_language,
    parse_failed,
)
from rag_engine.parsing.cleaners import CleanerSet
from rag_engine.parsing.normalize import normalize_text
from rag_engine.ports import ParseLimits

_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Assembled:
    text: str
    blocks: tuple[Block, ...]
    dropped_blocks: int  # blocks that were empty after normalisation and cleaning


class TextAssembler:
    def __init__(self, limits: ParseLimits, cleaners: CleanerSet | None = None) -> None:
        self._limits = limits
        self._cleaners = cleaners or CleanerSet()
        self._parts: list[str] = []
        self._blocks: list[Block] = []
        self._length = 0
        self._headings: list[tuple[int, str]] = []
        self._dropped = 0

    @property
    def heading_path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self._headings)

    def add(
        self,
        kind: BlockKind,
        text: str,
        *,
        level: int | None = None,
        page: int | None = None,
        attrs: Mapping[str, str | int | float | bool] | None = None,
    ) -> Block | None:
        """Append a block; returns it, or ``None`` when nothing is left of ``text``."""
        code = kind is BlockKind.CODE
        cleaned = normalize_text(text, code=code)
        if not code:
            cleaned = normalize_text(self._cleaners.apply(cleaned))
        if kind is BlockKind.HEADING:
            cleaned = " ".join(cleaned.split())
        if not cleaned:
            self._dropped += 1
            return None
        if len(self._blocks) >= self._limits.max_blocks:
            raise parse_failed(
                ParseReason.TOO_MANY_BLOCKS,
                "the document has more blocks than allowed",
                limit=self._limits.max_blocks,
            )
        start = self._length + (len(_SEPARATOR) if self._parts else 0)
        end = start + len(cleaned)
        if end > self._limits.max_text_chars:
            raise parse_failed(
                ParseReason.TOO_LARGE,
                "the text of the document is longer than allowed",
                limit=self._limits.max_text_chars,
            )
        if kind is BlockKind.HEADING:
            heading_level = level if level is not None else 1
            while self._headings and self._headings[-1][0] >= heading_level:
                self._headings.pop()
            self._headings.append((heading_level, cleaned))
            level = heading_level
        block = Block(
            kind=kind,
            text=cleaned,
            char_start=start,
            char_end=end,
            level=level,
            page=page,
            heading_path=self.heading_path,
            attrs=attrs or {},
        )
        self._parts.append(cleaned)
        self._blocks.append(block)
        self._length = end
        return block

    def build(self) -> Assembled:
        return Assembled(_SEPARATOR.join(self._parts), tuple(self._blocks), self._dropped)

    def parsed(
        self,
        *,
        title: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
        language: Language | None = None,
        warnings: tuple[str, ...] = (),
    ) -> ParsedDocument:
        """The finished document; the language is detected from the text unless given."""
        assembled = self.build()
        return ParsedDocument(
            normalized_text=assembled.text,
            blocks=assembled.blocks,
            title=title,
            author=author,
            created=created,
            language=language if language is not None else detect_language(assembled.text),
            warnings=warnings,
        )
