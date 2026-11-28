"""A chunk with no word characters is kept for coverage but never indexed.

Found by ingesting real documents: a section that was only a rule or a table separator became a
chunk whose analysed text is empty, and the whole document then failed to publish with
``E_INTEGRITY: chunks and embedding plan disagree``.
"""

from __future__ import annotations

import re

import pytest

from rag_engine.chunking.registry import STRATEGIES, ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import BlockKind, ChunkPolicy
from tests.chunking_checks import SMALL_POLICY, document_of, prose
from tests.factories import make_ids

COUNTER = HeuristicTokenCounter()
VERSION = make_ids().document_version_id()
P, H, T, C = BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.TABLE, BlockKind.CODE
WORD = re.compile(r"\w", re.UNICODE)


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "auto"])
@pytest.mark.parametrize("kind", [P, T, C])
@pytest.mark.parametrize(
    "rule", ["+------+------+", "----------", "***", "| --- | --- |", "= = = =", "…", "¶ §"]
)
def test_a_block_that_is_only_punctuation_is_not_indexed(
    strategy: str, kind: BlockKind, rule: str
) -> None:
    document = document_of(
        (H, "Intro", 1), (P, prose(2)), (H, "Divider", 1), (kind, rule),
        (H, "Next", 1), (P, prose(2, start=9)),
    )  # fmt: skip
    chunker = ChunkerRegistry(COUNTER, strategy).select(document)
    policy: ChunkPolicy = SMALL_POLICY
    chunks = list(chunker.chunk(document, policy, version_id=VERSION, normalizer_version="n/1"))
    assert any(c.indexed for c in chunks), "the real text is still indexed"
    for chunk in chunks:
        if chunk.indexed:
            assert WORD.search(chunk.text), f"an indexed chunk without a word: {chunk.text!r}"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "auto"])
def test_a_long_border_that_fills_whole_leaves_is_not_indexed(strategy: str) -> None:
    border = "\n".join("+" + "-" * 70 + "+" for _ in range(60))
    document = document_of((H, "Grid", 1), (T, border), (H, "After", 1), (P, prose(2)))
    chunker = ChunkerRegistry(COUNTER, strategy).select(document)
    chunks = list(
        chunker.chunk(document, SMALL_POLICY, version_id=VERSION, normalizer_version="n/1")
    )
    assert chunks
    assert all(WORD.search(c.text) for c in chunks if c.indexed)
    assert any(c.indexed for c in chunks) and any(not WORD.search(c.text) for c in chunks)
