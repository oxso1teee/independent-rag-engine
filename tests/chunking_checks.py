"""Checks every chunking result must pass, and builders of parsed documents for tests."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_engine.domain import (
    BlockKind,
    Chunk,
    ChunkPolicy,
    ParsedDocument,
)
from rag_engine.domain.invariants import (
    check_chunk_ids,
    check_chunk_layout,
    check_chunk_span,
    check_chunk_within_policy,
)
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.ports import ParseLimits, TokenCounter

DEFAULT_POLICY = ChunkPolicy()
SMALL_POLICY = ChunkPolicy(target_tokens=40, max_tokens=60, min_tokens=8, overlap_ratio=0.15)
NO_OVERLAP = ChunkPolicy(target_tokens=40, max_tokens=60, min_tokens=8, overlap_ratio=0.0)


def document_of(*blocks: tuple[BlockKind, str] | tuple[BlockKind, str, int]) -> ParsedDocument:
    """A parsed document from ``(kind, text)`` pairs; a heading takes its level as third item."""
    assembler = TextAssembler(ParseLimits())
    for item in blocks:
        kind, text = item[0], item[1]
        level = item[2] if len(item) == 3 else None
        assembler.add(kind, text, level=level)
    return assembler.parsed()


def sentence(index: int, words: int = 12) -> str:
    return " ".join(f"word{index}x{n}" for n in range(words)) + "."


def prose(sentences: int, words: int = 12, start: int = 0) -> str:
    return " ".join(sentence(start + i, words) for i in range(sentences))


def check_chunks(
    document: ParsedDocument,
    chunks: Sequence[Chunk],
    policy: ChunkPolicy,
    counter: TokenCounter,
    fingerprint: str,
) -> None:
    """The invariants of the data model plus coverage, bounds and overlap rules."""
    text = document.normalized_text
    for chunk in chunks:
        check_chunk_span(chunk, text)
        check_chunk_within_policy(chunk, policy)
        assert chunk.token_count == counter.count(chunk.text)
        assert chunk.text == chunk.text.strip() and chunk.text
        assert not chunk.oversize
        has_words = re.search(r"\w", chunk.text) is not None
        assert chunk.indexed == (chunk.level == 0 and not chunk.is_duplicate and has_words)
    check_chunk_ids(chunks, fingerprint)
    check_chunk_layout(chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    # every visible character of the document is inside some chunk
    covered = bytearray(len(text))
    for chunk in chunks:
        covered[chunk.char_start : chunk.char_end] = b"\x01" * (chunk.char_end - chunk.char_start)
    for index, char in enumerate(text):
        assert covered[index] or char.isspace(), f"character {index} is in no chunk"

    # indexed leaves never repeat a text
    seen = [c.text_hash for c in chunks if c.indexed]
    assert len(seen) == len(set(seen))

    # a parent contains its children
    by_id = {c.id: c for c in chunks}
    for chunk in chunks:
        if chunk.parent_id is not None:
            parent = by_id[chunk.parent_id]
            assert parent.char_start <= chunk.char_start and chunk.char_end <= parent.char_end


def overlap_tokens_between(first: Chunk, second: Chunk, counter: TokenCounter) -> int:
    """Tokens of the text the two chunks share (0 when they do not overlap)."""
    start, end = second.char_start, min(first.char_end, second.char_end)
    if end <= start:
        return 0
    return counter.count(first.text[start - first.char_start : end - first.char_start])
