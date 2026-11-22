"""DOCX parsing: structure from styles, tables, metadata, failures and archive limits."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from rag_engine.domain import BlockKind, ErrorCode, ParsedDocument, ParseReason, RagError
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.docx import DOCX_MIME, DocxParser
from rag_engine.ports import ParseHints, ParseLimits
from tests.docx_builder import (
    build_docx,
    docx_with_part,
    encrypted_docx_stub,
    zip_with,
)

PARSER = DocxParser()


def parse(data: bytes, **limits: int) -> ParsedDocument:
    return PARSER.parse(data, ParseHints(mime=DOCX_MIME, limits=ParseLimits(**limits)))


def reason_of(data: bytes, **limits: int) -> str:
    with pytest.raises(RagError) as info:
        parse(data, **limits)
    assert info.value.code is ErrorCode.E_PARSE_FAILED
    return str(info.value.context["reason"])


def kinds(document: ParsedDocument) -> list[tuple[BlockKind, str]]:
    return [(b.kind, b.text) for b in document.blocks]


def test_the_parser_accepts_docx_by_mime_or_extension() -> None:
    assert PARSER.supports(DOCX_MIME, None) and PARSER.supports(None, "DOCX")
    assert not PARSER.supports("application/pdf", "docx") and not PARSER.supports(None, "doc")
    assert PARSER.parser_id == "docx" and PARSER.normalizer_version == "norm/1"


def test_styles_decide_headings_paragraphs_lists_quotes_and_code() -> None:
    document = parse(
        build_docx(
            [
                ("title", "Handbook"),
                ("h2", "Setup"),
                ("p", "Install the tool."),
                ("li", "first item"),
                ("li2", "nested item"),
                ("quote", "a quoted line"),
                ("quote", "second quoted line"),
                ("code", "x = 1\ny = 2"),
                ("code", "z = 3"),
                ("h3", "Deeper"),
            ]
        )
    )
    assert kinds(document) == [
        (BlockKind.HEADING, "Handbook"),
        (BlockKind.HEADING, "Setup"),
        (BlockKind.PARAGRAPH, "Install the tool."),
        (BlockKind.LIST_ITEM, "first item"),
        (BlockKind.LIST_ITEM, "nested item"),
        (BlockKind.QUOTE, "a quoted line\nsecond quoted line"),
        (BlockKind.CODE, "x = 1\ny = 2\nz = 3"),
        (BlockKind.HEADING, "Deeper"),
    ]
    assert [b.level for b in document.blocks if b.kind is BlockKind.HEADING] == [1, 3, 4]
    assert [b.level for b in document.blocks if b.kind is BlockKind.LIST_ITEM] == [1, 2]
    assert document.blocks[-1].heading_path == ("Handbook", "Setup", "Deeper")


def test_without_a_title_heading_levels_follow_the_styles() -> None:
    document = parse(build_docx([("h1", "A"), ("h2", "B"), ("h1", "C")]))
    assert [(b.level, b.heading_path) for b in document.blocks] == [
        (1, ("A",)),
        (2, ("A", "B")),
        (1, ("C",)),
    ]


def test_headings_are_capped_at_level_six() -> None:
    document = parse(build_docx([("h9", "Deep")]))
    assert document.blocks[0].level == 6


def test_tables_become_rows_of_cells() -> None:
    document = parse(
        build_docx([("table", [["Name", "Qty"], ["Bolt", "12"], ["", ""], ["Nut", "7"]])])
    )
    (table,) = document.blocks
    assert table.kind is BlockKind.TABLE
    assert table.text == "Name | Qty\nBolt | 12\nNut | 7"
    assert dict(table.attrs) == {"rows": 3, "columns": 2}


def test_text_keeps_its_order_between_paragraphs_and_tables() -> None:
    document = parse(build_docx([("p", "Before."), ("table", [["a", "b"]]), ("p", "After.")]))
    assert [b.kind for b in document.blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.PARAGRAPH,
    ]


def test_russian_text_is_read() -> None:
    document = parse(build_docx([("h1", "Заголовок"), ("p", "Это русский текст документа.")]))
    assert document.language.value == "ru" and document.title == "Заголовок"


def test_metadata_is_read_and_the_first_heading_is_the_fallback_title() -> None:
    document = parse(
        build_docx(
            [("h1", "Heading")],
            title=" Annual  report ",
            author="Анна",
            created=datetime(2024, 3, 5, 10, 30, tzinfo=UTC),
        )
    )
    assert (document.title, document.author) == ("Annual report", "Анна")
    assert document.created == datetime(2024, 3, 5, 10, 30, tzinfo=UTC)
    bare = parse(build_docx([("h1", "Only heading"), ("p", "Text.")]))
    assert bare.title == "Only heading" and bare.author is None
    assert parse(build_docx([("p", "No heading at all.")])).title is None


def test_a_document_without_text_is_empty_not_an_error() -> None:
    document = parse(build_docx([("p", ""), ("table", [["", ""]])]))
    assert document.blocks == () and "empty_document" in document.warnings


def test_source_cleaners_apply_to_docx_text() -> None:
    cleaned = DocxParser(compile_cleaners([r"Confidential.*"]))
    document = cleaned.parse(
        build_docx([("p", "Confidential: internal use"), ("p", "Real content.")]),
        ParseHints(mime=DOCX_MIME),
    )
    assert [b.text for b in document.blocks] == ["Real content."]
    assert cleaned.normalizer_version != PARSER.normalizer_version


def test_a_password_protected_file_is_reported_as_encrypted() -> None:
    assert reason_of(encrypted_docx_stub()) == ParseReason.ENCRYPTED.value


def test_damaged_or_foreign_archives_are_corrupt() -> None:
    data = build_docx([("p", "Some text.")])
    assert reason_of(b"") == ParseReason.CORRUPT.value
    assert reason_of(b"PK\x03\x04 not an archive") == ParseReason.CORRUPT.value
    assert reason_of(data[: len(data) // 2]) == ParseReason.CORRUPT.value
    assert reason_of(zip_with({"readme.txt": b"hello"})) == ParseReason.CORRUPT.value
    assert reason_of(zip_with({"word/document.xml": b"<not-xml"})) == ParseReason.CORRUPT.value


def test_archives_that_expand_too_much_are_refused() -> None:
    data = build_docx([("p", "Some text.")])
    bomb = docx_with_part(data, "word/media/blob.bin", b"\x00" * 5_000_000)
    assert len(bomb) < 60_000
    assert reason_of(bomb) == ParseReason.ZIP_BOMB.value  # ratio
    assert reason_of(bomb, max_archive_ratio=100_000, max_archive_bytes=1_000_000) == "zip_bomb"
    assert reason_of(data, max_archive_entries=3) == ParseReason.ZIP_BOMB.value
    assert parse(bomb, max_archive_ratio=100_000, max_archive_bytes=10_000_000).blocks


def test_the_text_and_block_limits_are_enforced() -> None:
    assert reason_of(build_docx([("p", "Word " * 40)]), max_text_chars=50) == "too_large"
    many = build_docx([("p", f"paragraph {i}") for i in range(5)])
    assert reason_of(many, max_blocks=3) == ParseReason.TOO_MANY_BLOCKS.value


def test_tracked_deletions_are_dropped_and_insertions_kept() -> None:
    from docx import Document
    from docx.oxml import parse_xml

    document = Document()
    paragraph = document.add_paragraph("Visible")
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    deleted = f'<w:del {ns} w:id="1" w:author="a"><w:r><w:delText>Gone</w:delText></w:r></w:del>'
    inserted = f'<w:ins {ns} w:id="2" w:author="a"><w:r><w:t> Added</w:t></w:r></w:ins>'
    paragraph._p.append(parse_xml(deleted))
    paragraph._p.append(parse_xml(inserted))
    buffer = io.BytesIO()
    document.save(buffer)
    assert parse(buffer.getvalue()).normalized_text == "Visible Added"
