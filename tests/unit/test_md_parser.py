"""Markdown parser: structure, inline text, tables, code, front matter, limits."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_engine.domain import BlockKind, Language, ParsedDocument, RagError
from rag_engine.domain.invariants import check_block_provenance
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.frontmatter import parse_date, split_front_matter
from rag_engine.parsing.md import MarkdownParser
from rag_engine.ports import ParseHints, ParseLimits


def parse(text: str, parser: MarkdownParser | None = None, **hints: object) -> ParsedDocument:
    document = (parser or MarkdownParser()).parse(text.encode("utf-8"), ParseHints(**hints))  # type: ignore[arg-type]
    check_block_provenance(document.blocks)
    for block in document.blocks:
        assert document.normalized_text[block.char_start : block.char_end] == block.text
    return document


def shape(document: ParsedDocument) -> list[tuple[str, int | None, str]]:
    return [(b.kind.value, b.level, b.text) for b in document.blocks]


SAMPLE = """\
---
title: Field Guide
author: A. Writer
date: 2024-05-01
---

# Field Guide

Intro paragraph with **bold**, *italic*, `code` and a [link](http://example.com).

## Setup

1. First step
2. Second step
   - nested bullet

> Quoted line one
> still the quote

```python
def f():
    return 1
```

| Name | Value |
|------|-------|
| a    | 1     |
| b    | 2     |

### Details

Closing paragraph.
"""


def test_a_structured_document() -> None:
    document = parse(SAMPLE)
    assert shape(document) == [
        ("heading", 1, "Field Guide"),
        ("paragraph", None, "Intro paragraph with bold, italic, code and a link."),
        ("heading", 2, "Setup"),
        ("list_item", 1, "First step"),
        ("list_item", 1, "Second step"),
        ("list_item", 2, "nested bullet"),
        ("quote", None, "Quoted line one still the quote"),
        ("code", None, "def f():\n    return 1"),
        ("table", None, "Name | Value\na | 1\nb | 2"),
        ("heading", 3, "Details"),
        ("paragraph", None, "Closing paragraph."),
    ]
    paths = [b.heading_path for b in document.blocks]
    assert paths[0] == ("Field Guide",) and paths[3] == ("Field Guide", "Setup")
    assert paths[-1] == ("Field Guide", "Setup", "Details")
    assert document.title == "Field Guide" and document.author == "A. Writer"
    assert document.created == datetime(2024, 5, 1, tzinfo=UTC)
    assert document.language is Language.EN and document.warnings == ()
    code = next(b for b in document.blocks if b.kind is BlockKind.CODE)
    assert code.attrs == {"lang": "python", "lines": 2}
    table = next(b for b in document.blocks if b.kind is BlockKind.TABLE)
    assert table.attrs == {"rows": 3, "columns": 2}


def test_entities_and_escapes_are_decoded_exactly_once() -> None:
    document = parse("Fish &amp; chips, &amp;lt;tag&amp;gt; and &copy; \\*not italic\\*\n")
    assert document.blocks[0].text == "Fish & chips, &lt;tag&gt; and © *not italic*"


def test_inline_markup_is_reduced_to_its_text() -> None:
    document = parse("Line one  \nhard break\nsoft break ![alt text](pic.png) <span>x</span>\n")
    assert document.blocks[0].text == "Line one\nhard break soft break alt text x"


def test_setext_headings_and_rules() -> None:
    document = parse("Title\n=====\n\nSub\n---\n\n***\n\ntext\n")
    assert shape(document) == [
        ("heading", 1, "Title"),
        ("heading", 2, "Sub"),
        ("paragraph", None, "text"),
    ]


def test_list_items_carry_their_nesting_level_and_continuations() -> None:
    document = parse("- a\n  - b\n    - c\n- d\n\n  second paragraph of d\n")
    assert shape(document) == [
        ("list_item", 1, "a"),
        ("list_item", 2, "b"),
        ("list_item", 3, "c"),
        ("list_item", 1, "d\nsecond paragraph of d"),
    ]


def test_block_quotes_keep_their_text_together() -> None:
    document = parse("> first\n>\n> second\n>\n> - item in quote\n\nafter\n")
    assert document.blocks[0].kind is BlockKind.QUOTE
    assert document.blocks[0].text == "first\nsecond\nitem in quote"
    assert document.blocks[1].text == "after"


def test_code_keeps_indentation_and_ignores_cleaners() -> None:
    parser = MarkdownParser(compile_cleaners([r"noise"]))
    document = parse("```\n  indented\n\tnoise\n```\n\nnoise\n\n    indented block\n", parser)
    assert [b.kind for b in document.blocks] == [BlockKind.CODE, BlockKind.CODE]
    assert document.blocks[0].text == "  indented\n    noise"
    assert document.blocks[1].text == "indented block"


def test_a_table_with_formatting_and_missing_cells() -> None:
    document = parse("| **A** | B |\n|---|---|\n| `x` | y |\n| only |\n")
    assert document.blocks[0].text == "A | B\nx | y\nonly"


def test_raw_html_blocks_are_reduced_to_text() -> None:
    document = parse(
        "before\n\n<div>raw <b>html</b> &amp; text<script>evil()</script></div>\n\nafter\n"
    )
    assert [b.text for b in document.blocks] == ["before", "raw html & text", "after"]
    assert "evil" not in document.normalized_text


def test_front_matter_variants() -> None:
    meta, body, warnings = split_front_matter(
        "---\ntitle: \"Quoted\"\nauthor: 'Me'\ntags: [a, b]\n  nested: x\n---\nBody"
    )
    assert meta == {"title": "Quoted", "author": "Me", "tags": "[a, b]"}
    assert body == "Body" and warnings == ["front_matter_partial"]
    assert split_front_matter("no front matter")[0] == {}
    unclosed = "---\ntitle: x\nbody continues"
    assert split_front_matter(unclosed) == ({}, unclosed, [])
    assert split_front_matter("--- not a block\ntext")[0] == {}
    assert parse_date("2024-05-01T10:30:00+03:00") == datetime(2024, 5, 1, 7, 30, tzinfo=UTC)
    assert parse_date("2024-05-01 10:30") == datetime(2024, 5, 1, 10, 30, tzinfo=UTC)
    assert parse_date("yesterday") is None


def test_the_title_falls_back_to_the_first_h1() -> None:
    assert parse("# Heading title\n\ntext\n").title == "Heading title"
    assert parse("## only h2\n").title is None
    assert parse("---\ntitle: From front matter\n---\n# Other\n").title == "From front matter"
    bad_date = parse("---\ndate: someday\n---\ntext\n")
    assert bad_date.created is None and "front_matter_date_ignored" in bad_date.warnings


def test_only_front_matter_is_an_empty_document_that_keeps_its_metadata() -> None:
    document = parse("---\ntitle: Empty\n---\n")
    assert document.blocks == () and "empty_document" in document.warnings
    assert document.title == "Empty"
    assert parse("").warnings == ("empty_document",)


def test_russian_documents_are_detected_and_normalised() -> None:
    document = parse("# Руководство\n\nЭто русский   текст.\r\n")
    assert document.language is Language.RU
    assert document.blocks[1].text == "Это русский текст."


def test_nesting_beyond_the_limit_is_an_error() -> None:
    deep = "".join("  " * i + "- item\n" for i in range(30))
    with pytest.raises(RagError) as info:
        MarkdownParser().parse(deep.encode(), ParseHints(limits=ParseLimits(max_nesting=10)))
    assert info.value.context["reason"] == "too_deep"


@pytest.mark.parametrize(
    ("mime", "extension", "expected"),
    [
        ("text/markdown", None, True),
        ("text/plain", "md", False),
        (None, "md", True),
        (None, "Markdown", True),
        (None, "txt", False),
        ("text/html", "md", False),
    ],
)
def test_supports(mime: str | None, extension: str | None, expected: bool) -> None:
    assert MarkdownParser().supports(mime, extension) is expected
