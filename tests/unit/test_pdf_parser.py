"""PDF parsing: paragraphs, pages, metadata, failures and the repeated page furniture."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_engine.domain import BlockKind, ErrorCode, ParseReason, RagError
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.pdf import PdfParser, paragraphs_of
from rag_engine.ports import ParseHints, ParseLimits
from tests.pdf_builder import build_pdf, encrypt_pdf

PARSER = PdfParser()
FULL = "x" * 40  # a full-width line makes the shorter ones below look like paragraph ends


def parse(data: bytes, **limits: int) -> object:
    hints = ParseHints(mime="application/pdf", extension="pdf", limits=ParseLimits(**limits))
    return PARSER.parse(data, hints)


def reason_of(data: bytes, **limits: int) -> str:
    hints = ParseHints(mime="application/pdf", extension="pdf", limits=ParseLimits(**limits))
    with pytest.raises(RagError) as info:
        PARSER.parse(data, hints)
    assert info.value.code is ErrorCode.E_PARSE_FAILED
    return str(info.value.context["reason"])


def test_the_parser_accepts_pdf_by_mime_or_extension() -> None:
    assert PARSER.supports("application/pdf", None) and PARSER.supports(None, "PDF")
    assert not PARSER.supports("text/plain", "pdf") and not PARSER.supports(None, "txt")
    assert PARSER.parser_id == "pdf" and PARSER.normalizer_version == "norm/1"


def test_text_becomes_paragraph_blocks_with_their_page_numbers() -> None:
    data = build_pdf(
        [
            ["The first page starts with a long line of ordinary words", "and ends here."],
            ["The second page has one more sentence on it."],
        ]
    )
    document = PARSER.parse(data, ParseHints(mime="application/pdf"))
    assert [(b.kind, b.page) for b in document.blocks] == [
        (BlockKind.PARAGRAPH, 1),
        (BlockKind.PARAGRAPH, 2),
    ]
    assert document.blocks[0].text == (
        "The first page starts with a long line of ordinary words and ends here."
    )
    assert document.normalized_text.count("\n\n") == 1


def test_russian_text_is_extracted() -> None:
    data = build_pdf([["Это русский текст документа.", "Он записан в два слоя."]])
    document = PARSER.parse(data, ParseHints(mime="application/pdf"))
    assert "русский текст" in document.normalized_text
    assert document.language.value == "ru"


def test_lines_are_joined_and_paragraph_ends_are_recognised() -> None:
    lines = [
        FULL + " and the line goes on",
        FULL + " until the paragraph is over.",
        "Short final line.",
        "A new paragraph begins here with a",
        "line that continues further.",
    ]
    assert paragraphs_of(lines) == [
        f"{FULL} and the line goes on {FULL} until the paragraph is over. Short final line.",
        "A new paragraph begins here with a line that continues further.",
    ]


def test_list_items_and_short_headline_lines_start_new_paragraphs() -> None:
    lines = [
        FULL + " first line of the text",
        "of the text ends here.",
        "Heading",
        "Body text follows here again",
        "- first item",
        "- second item",
        "1. numbered item",
    ]
    assert paragraphs_of(lines) == [
        FULL + " first line of the text of the text ends here.",
        "Heading",
        "Body text follows here again",
        "- first item",
        "- second item",
        "1. numbered item",
    ]


def test_words_broken_by_a_hyphen_are_rejoined() -> None:
    assert paragraphs_of(["The coop-", "eration works well."]) == ["The cooperation works well."]
    assert paragraphs_of(["A well-known case", "of reuse."]) == ["A well-known case of reuse."]
    assert paragraphs_of(["Value 5-", "6 units"]) == ["Value 5- 6 units"]


def test_ligatures_are_expanded() -> None:
    document = PARSER.parse(
        build_pdf([["ﬁne ﬂow of oﬃce work"]]), ParseHints(mime="application/pdf")
    )
    assert document.normalized_text == "fine flow of office work"


def test_metadata_is_read() -> None:
    data = build_pdf(
        [["Some text on the page."]],
        title="  Annual   report ",
        author="Анна",
        created=datetime(2024, 3, 5, 10, 30, tzinfo=UTC),
    )
    document = PARSER.parse(data, ParseHints(mime="application/pdf"))
    assert (document.title, document.author) == ("Annual report", "Анна")
    assert document.created == datetime(2024, 3, 5, 10, 30, tzinfo=UTC)
    bare = PARSER.parse(build_pdf([["Text only."]]), ParseHints(mime="application/pdf"))
    assert (bare.title, bare.author, bare.created) == (None, None, None)


def test_pages_without_text_are_reported_but_do_not_fail_the_document() -> None:
    document = PARSER.parse(
        build_pdf([["Text on the first page."], None, ["Text on the third page."]]),
        ParseHints(mime="application/pdf"),
    )
    assert [b.page for b in document.blocks] == [1, 3]
    assert "pages_without_text:1" in document.warnings


def test_a_pdf_without_any_text_layer_is_refused() -> None:
    assert reason_of(build_pdf([None, None])) == ParseReason.NO_TEXT_LAYER.value


def test_a_password_protected_pdf_is_refused_but_an_owner_only_one_is_read() -> None:
    plain = build_pdf([["Readable text."]])
    assert reason_of(encrypt_pdf(plain, "secret")) == ParseReason.ENCRYPTED.value
    document = PARSER.parse(
        encrypt_pdf(plain, "owner", owner_only=True), ParseHints(mime="application/pdf")
    )
    assert document.normalized_text == "Readable text."


def test_damaged_files_are_reported_as_corrupt() -> None:
    data = build_pdf([["Some text."]])
    assert reason_of(b"%PDF-1.7\nnot really a pdf") == ParseReason.CORRUPT.value
    assert reason_of(data[: len(data) // 3]) == ParseReason.CORRUPT.value
    assert reason_of(b"") == ParseReason.CORRUPT.value


def test_the_page_limit_is_enforced() -> None:
    data = build_pdf([["One."], ["Two."], ["Three."]])
    assert reason_of(data, max_pages=2) == ParseReason.TOO_MANY_PAGES.value
    assert PARSER.parse(data, ParseHints(mime="application/pdf", limits=ParseLimits(max_pages=3)))


def test_the_text_and_block_limits_are_enforced() -> None:
    data = build_pdf([["Word " * 40]])
    assert reason_of(data, max_text_chars=50) == ParseReason.TOO_LARGE.value
    many = build_pdf([["- one", "- two", "- three"]])
    assert reason_of(many, max_blocks=2) == ParseReason.TOO_MANY_BLOCKS.value


def test_the_time_limit_is_enforced() -> None:
    data = build_pdf([["Text on a page."]] * 3)
    hints = ParseHints(mime="application/pdf", limits=ParseLimits(timeout_s=1e-9))
    with pytest.raises(RagError) as info:
        PARSER.parse(data, hints)
    assert info.value.context["reason"] == ParseReason.TIMEOUT.value


def test_running_titles_and_page_numbers_are_removed() -> None:
    pages = [
        [
            f"Annual report {number}",
            f"Chapter {number} has its own text on this page.",
            f"Page {number}",
        ]
        for number in range(1, 6)
    ]
    document = PARSER.parse(build_pdf(pages), ParseHints(mime="application/pdf"))
    assert "headers_footers_removed" in document.warnings
    assert "Annual report" not in document.normalized_text
    assert "Page 3" not in document.normalized_text
    assert [b.text for b in document.blocks][:2] == [
        "Chapter 1 has its own text on this page.",
        "Chapter 2 has its own text on this page.",
    ]


def test_a_line_that_repeats_on_few_pages_is_kept() -> None:
    pages = [
        ["Shared line", "Own text one."],
        ["Own text two."],
        ["Own text three."],
        ["Own four."],
    ]
    document = PARSER.parse(build_pdf(pages), ParseHints(mime="application/pdf"))
    assert "Shared line" in document.normalized_text
    assert "headers_footers_removed" not in document.warnings


def test_short_documents_never_lose_lines_as_furniture() -> None:
    pages = [["Same title", "Text A."], ["Same title", "Text B."]]
    document = PARSER.parse(build_pdf(pages), ParseHints(mime="application/pdf"))
    assert document.normalized_text.count("Same title") == 2


def test_source_cleaners_apply_to_pdf_text() -> None:
    cleaned = PdfParser(compile_cleaners([r"Draft.*"]))
    data = build_pdf([["Draft copy, not final", "Real content of the page."]])
    document = cleaned.parse(data, ParseHints(mime="application/pdf"))
    assert "Draft" not in document.normalized_text and "Real content" in document.normalized_text
    assert cleaned.normalizer_version != PARSER.normalizer_version
