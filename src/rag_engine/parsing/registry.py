"""The parsers available in this installation."""

from __future__ import annotations

from rag_engine.parsing.cleaners import CleanerSet
from rag_engine.parsing.html import HtmlParser
from rag_engine.parsing.md import MarkdownParser
from rag_engine.parsing.txt import TxtParser
from rag_engine.ports import Parser


def default_parsers(cleaners: CleanerSet | None = None) -> list[Parser]:
    """Built-in parsers first; optional formats are added when their extra is installed."""
    return [MarkdownParser(cleaners), HtmlParser(cleaners), TxtParser(cleaners)]
