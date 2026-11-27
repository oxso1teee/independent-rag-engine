"""The system prompt is pinned, small and says what generation relies on."""

from __future__ import annotations

import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.context.prompt import (
    PROMPT_SHA256,
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    TOKEN_LIMIT,
    system_prompt_sha256,
    system_prompt_tokens,
)

pytestmark = pytest.mark.unit


def test_the_prompt_is_pinned_so_a_change_is_a_deliberate_review_decision() -> None:
    assert system_prompt_sha256() == PROMPT_SHA256, (
        "the system prompt changed: update PROMPT_SHA256 and check TOKEN_LIMIT in the same commit"
    )


def test_the_prompt_stays_within_its_token_limit_with_a_conservative_counter() -> None:
    measured = system_prompt_tokens(HeuristicTokenCounter())
    assert 0 < measured <= TOKEN_LIMIT
    assert len(SYSTEM_PROMPT.split()) <= TOKEN_LIMIT  # even a word count fits


def test_the_prompt_states_the_rules_generation_depends_on() -> None:
    assert "[E1]" in SYSTEM_PROMPT and "[E2]" in SYSTEM_PROMPT  # the citation marker format
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT  # the refusal that generation recognises
    assert "only" in SYSTEM_PROMPT.lower() and "language of the question" in SYSTEM_PROMPT
    assert "{" not in SYSTEM_PROMPT  # no unfilled placeholder
