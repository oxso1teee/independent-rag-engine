"""The parsers available in this installation."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from importlib.util import find_spec

from rag_engine.parsing.cleaners import CleanerSet, compile_cleaners
from rag_engine.parsing.html import HtmlParser
from rag_engine.parsing.isolation import IsolatedParser, ParserWorker
from rag_engine.parsing.md import MarkdownParser
from rag_engine.parsing.txt import TxtParser
from rag_engine.ports import Parser

Builder = Callable[[CleanerSet | None], Parser]
FACTORY = "rag_engine.parsing.registry:make_parser"
"""What a worker calls to create a parser: ``make_parser(parser_id, cleaner_patterns)``."""


def _builders() -> dict[str, Builder]:
    """Built-in parsers first; optional formats are added when their extra is installed."""
    builders: dict[str, Builder] = {
        "md": MarkdownParser,
        "html": HtmlParser,
        "txt": TxtParser,
    }
    if find_spec("pypdf") is not None:
        from rag_engine.parsing.pdf import PdfParser

        builders["pdf"] = PdfParser
    if find_spec("docx") is not None:
        from rag_engine.parsing.docx import DocxParser

        builders["docx"] = DocxParser
    return builders


def make_parser(parser_id: str, patterns: Sequence[str] = ()) -> Parser:
    """One parser by id with the given cleaner patterns (how the worker builds its parsers)."""
    return _builders()[parser_id](compile_cleaners(patterns) if patterns else None)


def default_parsers(
    cleaners: CleanerSet | None = None,
    *,
    isolate: Collection[str] = (),
    worker: ParserWorker | None = None,
    memory_mb: int = 1024,
) -> list[Parser]:
    """The installed parsers; those named in ``isolate`` run in a separate worker process."""
    shared = worker if worker is not None else (ParserWorker(memory_mb) if isolate else None)
    patterns = list(cleaners.patterns) if cleaners else []
    parsers: list[Parser] = []
    for parser_id, build in _builders().items():
        parser = build(cleaners)
        if shared is not None and parser_id in isolate:
            parser = IsolatedParser(parser, shared, FACTORY, [parser_id, patterns])
        parsers.append(parser)
    return parsers
