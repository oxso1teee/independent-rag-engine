"""Parser port: raw bytes to a structured, normalised document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.domain import ParsedDocument, SourceKey


@dataclass(frozen=True, slots=True)
class ParseHints:
    mime: str | None = None
    extension: str | None = None
    source_key: SourceKey | None = None


@runtime_checkable
class Parser(Protocol):
    @property
    def parser_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def supports(self, mime: str | None, extension: str | None) -> bool: ...

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        """Raises ``E_PARSE_FAILED`` or ``E_UNSUPPORTED_FORMAT``; never returns partial garbage."""
        ...
