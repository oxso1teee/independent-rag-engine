"""Parser port: raw bytes to a structured, normalised document."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.domain import ParsedDocument, SourceKey, require


@dataclass(frozen=True, slots=True)
class ParseLimits:
    """Upper bounds a parser must enforce; exceeding one is an error, never a truncation."""

    max_input_bytes: int = 50 * 1024 * 1024
    max_text_chars: int = 20_000_000
    max_blocks: int = 200_000
    max_pages: int = 2_000
    max_archive_bytes: int = 200 * 1024 * 1024  # uncompressed size of a container format
    max_archive_ratio: int = 100  # uncompressed / compressed
    max_archive_entries: int = 10_000
    max_nesting: int = 200
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        require(
            min(
                self.max_input_bytes,
                self.max_text_chars,
                self.max_blocks,
                self.max_pages,
                self.max_archive_bytes,
                self.max_archive_ratio,
                self.max_archive_entries,
                self.max_nesting,
            )
            >= 1
            and self.timeout_s > 0,
            "parse limits must be positive",
        )


@dataclass(frozen=True, slots=True)
class ParseHints:
    mime: str | None = None
    extension: str | None = None
    source_key: SourceKey | None = None
    encoding: str | None = None  # detected text encoding, when known
    limits: ParseLimits = ParseLimits()


@runtime_checkable
class Parser(Protocol):
    @property
    def parser_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def normalizer_version(self) -> str:
        """Version of the normalisation rules (including cleaners) applied to the text."""
        ...

    def supports(self, mime: str | None, extension: str | None) -> bool: ...

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        """Raises ``E_PARSE_FAILED`` or ``E_UNSUPPORTED_FORMAT``; never returns partial garbage."""
        ...
