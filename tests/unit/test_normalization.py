"""Normalisation, cleaners and the text assembler (T3)."""

from __future__ import annotations

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import BlockKind, ErrorCode, ParseReason, RagError
from rag_engine.domain.invariants import check_block_provenance
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.parsing.cleaners import (
    MAX_CLEANERS,
    MAX_LINE_LENGTH,
    compile_cleaners,
    normalizer_version,
)
from rag_engine.parsing.normalize import NORMALIZER_BASE, normalize_text, unescape_entities_once
from rag_engine.ports import ParseLimits

# ------------------------------------------------------------------ normalizer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("  \n\t \n", ""),
        ("plain text", "plain text"),
        ("a\r\nb\rc\x0bd\x0ce", "a\nb\nc\nd\ne"),  # every line break becomes \n
        ("a b c\x85d", "a\nb\nc\nd"),
        ("many    spaces\tand\t\ttabs", "many spaces and tabs"),
        ("  leading and trailing  ", "leading and trailing"),
        ("a\n\n\n\n\nb", "a\n\nb"),  # at most one blank line
        ("a  \n  b", "a\nb"),
        ("non breaking em space", "non breaking em space"),
        ("zero​width⁠join﻿er", "zerowidthjoiner"),
        ("soft­hyphen", "softhyphen"),
        ("keep‍joiner‌s", "keep‍joiner‌s"),  # joiners carry meaning
        ("bell\x07 and null\x00 and esc\x1b", "bell and null and esc"),
        ("é", "é"),  # NFC
        ("e​́", "é"),  # removal happens before composition
        ("﻿BOM first", "BOM first"),
        ("Привет, мир!\r\n\r\n\r\nПока", "Привет, мир!\n\nПока"),
    ],
)
def test_prose_cases(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\n\n  def f():\n\treturn 1  \n\n", "  def f():\n    return 1"),  # indentation kept
        ("a    b\r\nc", "a    b\nc"),  # inner spacing kept
        ("x​y", "xy"),
        ("", ""),
    ],
)
def test_code_cases(raw: str, expected: str) -> None:
    assert normalize_text(raw, code=True) == expected


@given(st.text())
def test_normalisation_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


@given(st.text())
def test_code_normalisation_is_idempotent(text: str) -> None:
    once = normalize_text(text, code=True)
    assert normalize_text(once, code=True) == once


@given(st.text())
def test_normalised_text_has_no_forbidden_characters(text: str) -> None:
    result = normalize_text(text)
    assert not set(result) & set("\r\x0b\x0c  ​⁠﻿­\t")
    assert "\n\n\n" not in result and "  " not in result
    assert result == result.strip()


@given(st.text(alphabet=st.characters(codec="utf-8"), max_size=200))
def test_text_without_special_characters_keeps_its_words(text: str) -> None:
    words = text.split()
    if all(w.isalnum() for w in words):
        assert normalize_text(text).split() == [normalize_text(w) for w in words]


@pytest.mark.parametrize(
    ("raw", "decoded"),
    [
        ("&lt;b&gt;", "<b>"),
        ("&amp;lt;", "&lt;"),  # one level only: the regression of a double decode
        ("&amp;amp;", "&amp;"),
        ("&#1055;&#x440;", "Пр"),
        ("&unknown; &", "&unknown; &"),
        ("no entities", "no entities"),
    ],
)
def test_entities_are_decoded_exactly_once(raw: str, decoded: str) -> None:
    assert unescape_entities_once(raw) == decoded
    # decoding once and normalising never decodes a second time
    assert normalize_text(unescape_entities_once(raw)) == normalize_text(decoded)


# -------------------------------------------------------------------- cleaners


def test_a_line_matching_a_cleaner_entirely_is_removed() -> None:
    cleaners = compile_cleaners([r"©.*", r"Page \d+ of \d+", r"(?i)all rights reserved\.?"])
    text = "Real content\n© 2024 Example Corp\nPage 3 of 10\nAll Rights Reserved.\nMore content"
    assert cleaners.apply(text) == "Real content\nMore content"


def test_a_partial_match_leaves_the_line_alone() -> None:
    cleaners = compile_cleaners([r"Page \d+ of \d+"])
    text = "See Page 3 of 10 for details"
    assert cleaners.apply(text) == text


def test_placeholder_lines_are_dropped_before_chunking() -> None:
    stub = "Содержимое недоступно. Попробуйте позже."
    cleaners = compile_cleaners([re_escape(stub)])
    assembler = TextAssembler(ParseLimits(), cleaners)
    assembler.add(BlockKind.PARAGRAPH, stub)
    assembler.add(BlockKind.PARAGRAPH, "Настоящий текст документа.")
    assembler.add(BlockKind.PARAGRAPH, stub)
    built = assembler.build()
    assert built.text == "Настоящий текст документа."
    assert built.dropped_blocks == 2


def re_escape(text: str) -> str:
    import re

    return re.escape(text)


def test_no_cleaners_means_no_change_and_the_base_version() -> None:
    empty = compile_cleaners([])
    assert not empty and empty.apply("a\nb") == "a\nb"
    assert normalizer_version(empty) == normalizer_version() == NORMALIZER_BASE


def test_the_normalizer_version_follows_the_cleaners() -> None:
    a = normalizer_version(compile_cleaners([r"noise"]))
    b = normalizer_version(compile_cleaners([r"noise", r"more"]))
    again = normalizer_version(compile_cleaners([r"noise"]))
    assert a == again and a != b and a != NORMALIZER_BASE
    assert a.startswith(NORMALIZER_BASE + "+")


@pytest.mark.parametrize(
    ("pattern", "reason"),
    [
        ("", "empty pattern"),
        ("   ", "empty pattern"),
        ("x" * 301, "longer than"),
        ("(a+)+", "nested unbounded repetition"),
        ("(.*)*", "nested unbounded repetition"),
        (r"(\w+\s*)*", "nested unbounded repetition"),
        ("(x*)+y", "nested unbounded repetition"),
        ("[unclosed", "not a valid regular expression"),
        (".*", "matches an empty or blank line"),
        (r"\s*", "matches an empty or blank line"),
        ("a?", "matches an empty or blank line"),
    ],
)
def test_unsafe_or_invalid_cleaners_are_rejected(pattern: str, reason: str) -> None:
    with pytest.raises(RagError) as info:
        compile_cleaners([pattern])
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert reason in info.value.message
    assert info.value.context["cleaner"] == 1


def test_too_many_cleaners_are_rejected() -> None:
    compile_cleaners([f"noise{i}" for i in range(MAX_CLEANERS)])
    with pytest.raises(RagError):
        compile_cleaners([f"noise{i}" for i in range(MAX_CLEANERS + 1)])


def test_a_pattern_that_looks_dangerous_cannot_stall_a_scan() -> None:
    # accepted patterns are cheap even on adversarial input, very long lines are not tested
    cleaners = compile_cleaners([r"a+b", r"[a-z]+\d"])
    started = time.perf_counter()
    cleaners.apply("a" * 5_000 + "\n" + "a" * (MAX_LINE_LENGTH - 1) + "!")
    assert time.perf_counter() - started < 1.0
    huge = "x" * (MAX_LINE_LENGTH + 1)
    assert compile_cleaners([r"x+"]).apply(huge) == huge  # too long: left untouched


# ------------------------------------------------------------------- assembler


def test_offsets_are_exact_and_blocks_are_separated_by_a_blank_line() -> None:
    assembler = TextAssembler(ParseLimits())
    assembler.add(BlockKind.HEADING, "  Title  ", level=1)
    assembler.add(BlockKind.PARAGRAPH, "First\r\nparagraph")
    assembler.add(BlockKind.PARAGRAPH, "   ")  # nothing left: dropped
    assembler.add(BlockKind.CODE, "\n    x = 1\n\ty = 2\n", attrs={"lang": "python"})
    built = assembler.build()
    assert built.text == "Title\n\nFirst\nparagraph\n\n    x = 1\n    y = 2"
    assert built.dropped_blocks == 1
    for block in built.blocks:
        assert built.text[block.char_start : block.char_end] == block.text
    assert [b.kind for b in built.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.CODE,
    ]
    assert built.blocks[2].attrs["lang"] == "python"


def test_heading_paths_are_tracked_by_the_assembler() -> None:
    assembler = TextAssembler(ParseLimits())
    assembler.add(BlockKind.PARAGRAPH, "preamble")
    assembler.add(BlockKind.HEADING, "Guide", level=1)
    assembler.add(BlockKind.PARAGRAPH, "intro")
    assembler.add(BlockKind.HEADING, "Setup", level=2)
    assembler.add(BlockKind.HEADING, "Install", level=3)
    assembler.add(BlockKind.LIST_ITEM, "step one", level=1)
    assembler.add(BlockKind.HEADING, "Usage", level=2)  # closes Install and Setup
    assembler.add(BlockKind.PARAGRAPH, "body")
    assembler.add(BlockKind.HEADING, "Appendix", level=1)
    paths = [b.heading_path for b in assembler.build().blocks]
    assert paths == [
        (),
        ("Guide",),
        ("Guide",),
        ("Guide", "Setup"),
        ("Guide", "Setup", "Install"),
        ("Guide", "Setup", "Install"),
        ("Guide", "Usage"),
        ("Guide", "Usage"),
        ("Appendix",),
    ]
    check_block_provenance(assembler.build().blocks)


def test_headings_are_single_line_and_cleaners_do_not_touch_code() -> None:
    cleaners = compile_cleaners([r"noise"])
    assembler = TextAssembler(ParseLimits(), cleaners)
    heading = assembler.add(BlockKind.HEADING, "Two\nlines", level=2)
    code = assembler.add(BlockKind.CODE, "noise")
    prose = assembler.add(BlockKind.PARAGRAPH, "noise")
    assert heading is not None and heading.text == "Two lines" and heading.level == 2
    assert code is not None and code.text == "noise"
    assert prose is None


def test_pages_and_levels_are_passed_through() -> None:
    assembler = TextAssembler(ParseLimits())
    block = assembler.add(BlockKind.LIST_ITEM, "item", level=2, page=4)
    assert block is not None and (block.level, block.page) == (2, 4)


def test_the_text_and_block_limits_are_errors_not_truncation() -> None:
    small = ParseLimits(max_text_chars=20)
    assembler = TextAssembler(small)
    assembler.add(BlockKind.PARAGRAPH, "a" * 15)
    with pytest.raises(RagError) as info:
        assembler.add(BlockKind.PARAGRAPH, "b" * 15)
    assert info.value.context["reason"] == ParseReason.TOO_LARGE.value

    few = TextAssembler(ParseLimits(max_blocks=2))
    few.add(BlockKind.PARAGRAPH, "one")
    few.add(BlockKind.PARAGRAPH, "two")
    with pytest.raises(RagError) as many:
        few.add(BlockKind.PARAGRAPH, "three")
    assert many.value.context["reason"] == ParseReason.TOO_MANY_BLOCKS.value
    # empty blocks do not count towards the limit
    quiet = TextAssembler(ParseLimits(max_blocks=1))
    quiet.add(BlockKind.PARAGRAPH, "   ")
    quiet.add(BlockKind.PARAGRAPH, "fits")


def test_the_parsed_document_carries_metadata_and_a_detected_language() -> None:
    assembler = TextAssembler(ParseLimits())
    assembler.add(BlockKind.HEADING, "Руководство", level=1)
    assembler.add(BlockKind.PARAGRAPH, "Это русский текст документа.")
    document = assembler.parsed(title="Руководство", author="A", warnings=("w",))
    assert document.title == "Руководство" and document.author == "A"
    assert document.language.value == "ru" and document.warnings == ("w",)
    empty = TextAssembler(ParseLimits()).parsed()
    assert empty.normalized_text == "" and empty.blocks == ()


_kinds = st.sampled_from(
    [BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.QUOTE, BlockKind.CODE, BlockKind.TABLE]
)
_steps = st.lists(
    st.one_of(
        st.tuples(st.just("heading"), st.integers(1, 4), st.text(max_size=30)),
        st.tuples(st.just("block"), _kinds, st.text(max_size=120)),
    ),
    max_size=40,
)


@given(_steps)
def test_any_sequence_of_blocks_yields_a_consistent_document(
    steps: list[tuple[object, ...]],
) -> None:
    assembler = TextAssembler(ParseLimits())
    for step in steps:
        if step[0] == "heading":
            assembler.add(BlockKind.HEADING, str(step[2]), level=int(step[1]))  # type: ignore[call-overload]
        else:
            assembler.add(step[1], str(step[2]))  # type: ignore[arg-type]
    document = assembler.parsed()  # the model itself verifies offsets and ordering
    check_block_provenance(document.blocks)
    assert document.normalized_text == normalize_text(document.normalized_text) or any(
        b.kind is BlockKind.CODE for b in document.blocks
    )
    for first, second in zip(document.blocks, document.blocks[1:], strict=False):
        assert first.char_end + 2 == second.char_start
