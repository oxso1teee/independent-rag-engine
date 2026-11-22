"""Contract every ``Parser`` implementation must satisfy.

New parsers are added to ``CASES`` together with a few valid samples; the same checks then run
against them.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_engine.domain import ErrorCode, ParsedDocument, RagError
from rag_engine.domain.invariants import check_block_provenance
from rag_engine.parsing.html import HtmlParser
from rag_engine.parsing.md import MarkdownParser
from rag_engine.parsing.txt import TxtParser
from rag_engine.ports import ParseHints, ParseLimits, Parser

CLEAN_FAILURES = {ErrorCode.E_PARSE_FAILED, ErrorCode.E_UNSUPPORTED_FORMAT}


@dataclass(frozen=True)
class Case:
    """A parser together with the inputs it is expected to handle."""

    make: Callable[[], Parser]
    mime: str
    extension: str
    samples: tuple[bytes, ...]


CASES: dict[str, Case] = {
    "html": Case(
        make=HtmlParser,
        mime="text/html",
        extension="html",
        samples=(
            b"<html><body><h1>Title</h1><p>A paragraph.</p><ul><li>one</li></ul></body></html>",
            "<h2>Раздел</h2><p>Текст &amp; <b>выделение</b>.</p><pre>code</pre>".encode(),
            b"<table><tr><th>a</th></tr><tr><td>1</td></tr></table><blockquote>q</blockquote>",
            b"plain text without any markup",
        ),
    ),
    "md": Case(
        make=MarkdownParser,
        mime="text/markdown",
        extension="md",
        samples=(
            b"# Title\n\nA paragraph.\n\n## Part\n\n- one\n- two\n",
            "---\ntitle: T\n---\n\n# Заголовок\n\nТекст `код` и **выделение**.\n".encode(),
            b"| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n```\n\n> quote\n",
            b"plain text without any markup",
        ),
    ),
    "txt": Case(
        make=TxtParser,
        mime="text/plain",
        extension="txt",
        samples=(
            b"First paragraph.\n\nSecond paragraph\nwith two lines.\n",
            "Привет, мир.\r\n\r\nВторой абзац.".encode(),
            b"   \n\n  indented   and   spaced   \n",
            b"single line without newline",
        ),
    ),
}


@pytest.fixture(params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> Case:
    return CASES[request.param]


def attempt(
    parser: Parser, data: bytes, hints: ParseHints
) -> tuple[ParsedDocument | None, RagError | None]:
    """Parse ``data``; a parser may refuse, but only with a documented error."""
    try:
        return parser.parse(data, hints), None
    except RagError as error:
        return None, error


def check_outcome(document: ParsedDocument | None, error: RagError | None) -> None:
    if error is not None:
        assert error.code in CLEAN_FAILURES
    else:
        assert document is not None
        check_block_provenance(document.blocks)


def hints_for(case: Case) -> ParseHints:
    return ParseHints(mime=case.mime, extension=case.extension)


def test_satisfies_the_protocol(case: Case) -> None:
    parser = case.make()
    assert isinstance(parser, Parser)
    assert parser.parser_id and parser.version and parser.normalizer_version
    assert parser.supports(case.mime, case.extension)
    assert not parser.supports("application/x-nothing", "nothing")


def test_samples_give_valid_documents(case: Case) -> None:
    parser = case.make()
    for sample in case.samples:
        document = parser.parse(sample, hints_for(case))
        assert isinstance(document, ParsedDocument)
        assert document.blocks, "a sample with content must produce blocks"
        check_block_provenance(document.blocks)
        for block in document.blocks:
            assert document.normalized_text[block.char_start : block.char_end] == block.text
            assert block.text == block.text.strip()
        assert document.normalized_text == document.normalized_text.strip()


def test_parsing_is_deterministic(case: Case) -> None:
    parser = case.make()
    for sample in case.samples:
        assert parser.parse(sample, hints_for(case)) == parser.parse(sample, hints_for(case))


def test_empty_input_is_a_valid_empty_document(case: Case) -> None:
    for data in (b"", b"   \n\n  \t"):
        document, error = attempt(case.make(), data, hints_for(case))
        if error is not None:  # binary formats may refuse empty input, but only cleanly
            assert error.code in CLEAN_FAILURES
        else:
            assert document is not None
            assert document.blocks == () and document.normalized_text == ""
            assert "empty_document" in document.warnings


def test_the_input_size_limit_is_enforced(case: Case) -> None:
    hints = ParseHints(
        mime=case.mime, extension=case.extension, limits=ParseLimits(max_input_bytes=10)
    )
    with pytest.raises(RagError) as info:
        case.make().parse(b"x" * 11 + case.samples[0], hints)
    assert info.value.code is ErrorCode.E_PARSE_FAILED
    assert info.value.context["reason"] == "too_large"


@given(st.binary(max_size=2_000))
@settings(max_examples=60, deadline=None)
def test_arbitrary_bytes_never_escape_as_unexpected_errors(data: bytes) -> None:
    for entry in CASES.values():
        check_outcome(*attempt(entry.make(), data, hints_for(entry)))


def test_mutated_samples_never_escape_as_unexpected_errors(case: Case) -> None:
    rng = random.Random(7)  # noqa: S311 - a fixed seed makes the mutations reproducible
    for sample in case.samples:
        for _ in range(40):
            mutated = bytearray(sample)
            for _ in range(rng.randint(1, 4)):
                if mutated:
                    mutated[rng.randrange(len(mutated))] = rng.randrange(256)
            check_outcome(*attempt(case.make(), bytes(mutated), hints_for(case)))
