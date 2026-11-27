"""Truncation at a sentence, else a word, else a character, always an exact prefix."""

from __future__ import annotations

import pytest

from rag_engine.context.truncate import truncate_to_tokens

pytestmark = pytest.mark.unit


def count(text: str) -> int:
    return len(text.split())


def test_text_that_fits_is_returned_untouched() -> None:
    assert truncate_to_tokens("one two three.", 3, count) == "one two three."


def test_the_cut_falls_on_the_last_sentence_that_fits() -> None:
    text = "One two three. Four five six! Seven eight nine ten eleven."
    assert truncate_to_tokens(text, 7, count) == "One two three. Four five six!"
    assert truncate_to_tokens(text, 3, count) == "One two three."


def test_without_a_fitting_sentence_the_cut_is_at_a_word() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    assert truncate_to_tokens(text, 4, count) == "alpha beta gamma delta"


def test_a_closing_quote_stays_with_its_sentence() -> None:
    text = 'He said "stop." Then he left the room quietly.'
    assert truncate_to_tokens(text, 3, count) == 'He said "stop."'


def test_one_endless_word_is_cut_by_characters() -> None:
    text = "x" * 50
    cut = truncate_to_tokens(text, 1, lambda t: -(-len(t) // 10))  # 10 characters per token
    assert cut == "x" * 10


def test_nothing_fits_into_a_zero_limit() -> None:
    assert truncate_to_tokens("anything at all", 0, count) == ""


@pytest.mark.parametrize("limit", range(1, 12))
def test_the_result_is_always_an_exact_prefix_within_the_limit(limit: int) -> None:
    text = "First one. Second sentence is a bit longer! Third? Fourth and last sentence here."
    cut = truncate_to_tokens(text, limit, count)
    assert text.startswith(cut) and count(cut) <= limit
