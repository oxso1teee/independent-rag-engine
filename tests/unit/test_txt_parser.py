"""Plain-text parser: encodings, paragraphs, warnings, cleaners."""

from __future__ import annotations

import pytest

from rag_engine.domain import BlockKind, ErrorCode, Language, RagError
from rag_engine.parsing.cleaners import compile_cleaners, normalizer_version
from rag_engine.parsing.txt import TxtParser
from rag_engine.ports import ParseHints, ParseLimits

RUSSIAN = "Привет, это достаточно длинный русский текст для определения кодировки. " * 3


def parse(data: bytes, **hints: object) -> object:
    return TxtParser().parse(data, ParseHints(**hints))  # type: ignore[arg-type]


def test_paragraphs_are_blank_line_separated_and_lines_are_kept() -> None:
    document = TxtParser().parse(b"One\ntwo\n\n\n\nThree\r\n\r\nFour", ParseHints())
    assert [b.text for b in document.blocks] == ["One\ntwo", "Three", "Four"]
    assert {b.kind for b in document.blocks} == {BlockKind.PARAGRAPH}
    assert document.normalized_text == "One\ntwo\n\nThree\n\nFour"
    assert document.title is None and document.warnings == ()


def test_the_language_is_detected() -> None:
    assert TxtParser().parse(RUSSIAN.encode(), ParseHints()).language is Language.RU
    assert (
        TxtParser().parse(b"An ordinary English sentence here.", ParseHints()).language
        is Language.EN
    )


@pytest.mark.parametrize("encoding", ["utf_8", "utf_8_sig", "cp1251", "koi8_r", "cp866", "utf_16"])
def test_encodings_given_by_the_hint_are_decoded(encoding: str) -> None:
    document = TxtParser().parse(RUSSIAN.encode(encoding), ParseHints(encoding=encoding))
    assert document.normalized_text == RUSSIAN.strip()
    assert document.warnings == ()


def test_the_encoding_is_detected_without_a_hint() -> None:
    document = TxtParser().parse(RUSSIAN.encode("cp1251"), ParseHints())
    assert document.normalized_text == RUSSIAN.strip()
    assert document.warnings == ("decoded_with_replacement",)


def test_a_wrong_hint_falls_back_to_replacement_with_a_warning() -> None:
    document = TxtParser().parse("Привет мир".encode("cp1251"), ParseHints(encoding="utf_8"))
    assert "decoded_with_replacement" in document.warnings
    assert document.normalized_text  # decoded, not dropped
    unknown = TxtParser().parse(b"plain ascii", ParseHints(encoding="no-such-codec"))
    assert unknown.normalized_text == "plain ascii"


def test_undecodable_garbage_is_an_error_not_silent_loss() -> None:
    import os

    with pytest.raises(RagError) as info:
        TxtParser().parse(bytes(range(128, 256)) * 4 + os.urandom(64), ParseHints())
    assert info.value.code is ErrorCode.E_PARSE_FAILED


@pytest.mark.parametrize("data", [b"", b"\n\n\n", b"  \t \r\n  ", "﻿".encode()])
def test_empty_documents_are_valid_and_flagged(data: bytes) -> None:
    document = TxtParser().parse(data, ParseHints())
    assert document.blocks == () and document.normalized_text == ""
    assert document.warnings == ("empty_document",)


def test_bom_zero_width_and_control_characters_do_not_reach_the_text() -> None:
    document = TxtParser().parse("﻿zero​width\x07 ok".encode(), ParseHints())
    assert document.normalized_text == "zerowidth ok"


def test_cleaners_remove_noise_lines_and_version_the_normaliser() -> None:
    cleaners = compile_cleaners([r"-+ page \d+ -+"])
    parser = TxtParser(cleaners)
    document = parser.parse(b"Body line\n--- page 3 ---\nnext line\n\n--- page 4 ---", ParseHints())
    assert document.normalized_text == "Body line\nnext line"
    assert parser.normalizer_version == normalizer_version(cleaners)
    assert parser.normalizer_version != TxtParser().normalizer_version


def test_text_and_block_limits_are_enforced() -> None:
    with pytest.raises(RagError) as too_long:
        TxtParser().parse(b"x" * 100, ParseHints(limits=ParseLimits(max_text_chars=50)))
    assert too_long.value.context["reason"] == "too_large"
    with pytest.raises(RagError) as too_many:
        TxtParser().parse(b"a\n\nb\n\nc", ParseHints(limits=ParseLimits(max_blocks=2)))
    assert too_many.value.context["reason"] == "too_many_blocks"


@pytest.mark.parametrize(
    ("mime", "extension", "expected"),
    [
        ("text/plain", None, True),
        ("text/plain", "md", True),  # the detected type decides
        ("text/markdown", "txt", False),
        ("text/html", None, False),
        (None, "txt", True),
        (None, "LOG", True),
        (None, "md", False),
        (None, None, False),
    ],
)
def test_supports(mime: str | None, extension: str | None, expected: bool) -> None:
    assert TxtParser().supports(mime, extension) is expected


def test_a_large_document_is_parsed_within_the_limits() -> None:
    text = ("Paragraph number %d with some words.\n\n" * 1) * 1
    data = "".join(f"Paragraph number {i} with some words.\n\n" for i in range(20_000)).encode()
    document = TxtParser().parse(data, ParseHints())
    assert len(document.blocks) == 20_000
    assert text  # keeps the template readable
