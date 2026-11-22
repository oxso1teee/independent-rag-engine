"""Sentence, word, line and hard splitting used by the chunkers."""

from __future__ import annotations

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.chunking.split import split_hard, split_lines, split_sentences, split_words

# --------------------------------------------------------------------- sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two. Three.", ["One. ", "Two. ", "Three."]),
        ("Is it? Yes! Really.", ["Is it? ", "Yes! ", "Really."]),
        (
            "Первое предложение. Второе предложение.",
            ["Первое предложение. ", "Второе предложение."],
        ),
        ("Это так, т.е. правильно. Следующее.", ["Это так, т.е. правильно. ", "Следующее."]),
        (
            "Мы живём на ул. Ленина в г. Москве. Дом большой.",
            ["Мы живём на ул. Ленина в г. Москве. ", "Дом большой."],
        ),
        ("Dr. Smith met Mr. Jones. They talked.", ["Dr. Smith met Mr. Jones. ", "They talked."]),
        ("See e.g. the guide. Then continue.", ["See e.g. the guide. ", "Then continue."]),
        ("A. S. Pushkin wrote it. Later poems.", ["A. S. Pushkin wrote it. ", "Later poems."]),
        ("А. С. Пушкин написал. Дальше.", ["А. С. Пушкин написал. ", "Дальше."]),
        ("Pi is 3.14 exactly. Next.", ["Pi is 3.14 exactly. ", "Next."]),
        ("Version 2.0 is out. Upgrade.", ["Version 2.0 is out. ", "Upgrade."]),
        ("Visit example.com/path. Then leave.", ["Visit example.com/path. ", "Then leave."]),
        ('He said "stop." Then left.', ['He said "stop." ', "Then left."]),
        ("Wait... what happened? Nobody knows.", ["Wait... what happened? ", "Nobody knows."]),
        ("It ended... then it began again.", ["It ended... then it began again."]),
        ("1. First item. 2. Second item.", ["1. First item. ", "2. Second item."]),
        ("no punctuation at all", ["no punctuation at all"]),
        ("lowercase follows. so no split", ["lowercase follows. so no split"]),
        ("", []),
        ("   ", ["   "]),
        ("Trailing space. ", ["Trailing space. "]),
        ("Line one.\nLine two.", ["Line one.\n", "Line two."]),
    ],
)
def test_sentence_cases(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


@given(st.text(max_size=500))
def test_sentences_reassemble_exactly(text: str) -> None:
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert all(pieces)


def test_sentence_splitting_of_a_large_text_is_fast() -> None:
    text = "Это предложение. This is a sentence! Что дальше? " * 40_000
    started = time.perf_counter()
    pieces = split_sentences(text)
    assert len(pieces) == 120_000
    assert time.perf_counter() - started < 5.0


# ----------------------------------------------------------------- words, lines


def test_words_keep_their_following_whitespace() -> None:
    assert split_words("  hello   brave\nnew world ") == ["  hello   ", "brave\n", "new ", "world "]
    assert split_words("") == []
    assert split_words("single") == ["single"]
    assert split_words("   ") == ["   "]


@given(st.text(max_size=300))
def test_words_reassemble_exactly(text: str) -> None:
    pieces = split_words(text)
    assert "".join(pieces) == text
    assert all(pieces)


def test_lines_keep_their_breaks() -> None:
    assert split_lines("a\nb\n\nc") == ["a\n", "b\n", "\n", "c"]
    assert split_lines("a\n") == ["a\n"]
    assert split_lines("") == []
    assert split_lines("\n") == ["\n"]


@given(st.text(max_size=300))
def test_lines_reassemble_exactly(text: str) -> None:
    assert "".join(split_lines(text)) == text


# ------------------------------------------------------------------------- hard


def test_hard_splitting_respects_the_size() -> None:
    assert split_hard("abcdefghij", 4) == ["abcd", "efgh", "ij"]
    assert split_hard("abc", 10) == ["abc"]
    assert split_hard("", 3) == []
    with pytest.raises(ValueError, match="positive"):
        split_hard("abc", 0)


def test_hard_splitting_does_not_separate_marks_from_their_letters() -> None:
    text = "é" * 6  # six accented letters, two code points each
    pieces = split_hard(text, 3)
    assert "".join(pieces) == text
    assert all(not p.startswith("́") for p in pieces)
    joined = "a‍b" * 4
    assert all(not p.startswith("‍") for p in split_hard(joined, 2))


@given(st.text(max_size=300), st.integers(min_value=1, max_value=40))
def test_hard_pieces_reassemble_and_stay_within_the_bound(text: str, size: int) -> None:
    pieces = split_hard(text, size)
    assert "".join(pieces) == text
    assert all(0 < len(p) <= size for p in pieces)
