"""The system prompt and its measured size.

The prompt is part of the token budget (``ContextBuilder`` subtracts its measured length), so its
size is a quantity to watch, not an incidental detail. Any change must update the pinned hash and
the token limit together, which makes growth a decision in review.
"""

from __future__ import annotations

import hashlib

from rag_engine.ports import TokenCounter

PROMPT_ID = "rag-system/1"
REFUSAL_SENTENCE = "The provided documents do not contain enough information to answer."

SYSTEM_PROMPT = """\
You answer questions using only the numbered evidence passages provided between the markers \
[E1], [E2] and so on. Follow these rules.
1. Use only facts that appear in the evidence. Do not use outside knowledge and do not guess.
2. After every statement that relies on evidence, cite it with its marker, for example [E2]. \
Cite only markers that exist.
3. If the evidence does not contain the answer, reply exactly: "{refusal}" Do not cite anything \
in that case.
4. If passages disagree, say so and cite both.
5. Answer in the language of the question. Keep the answer short and direct.\
""".replace("{refusal}", REFUSAL_SENTENCE)

# Pinned on purpose: changing the prompt must change these in the same commit.
PROMPT_SHA256 = "d3527cd5213b0e17f3bbe584af25068b93b2e8c721dcfb33317dd43cc9818307"
TOKEN_LIMIT = 260


def system_prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def system_prompt_tokens(counter: TokenCounter) -> int:
    """Tokens the system prompt costs with the given counter."""
    return counter.count(SYSTEM_PROMPT)
