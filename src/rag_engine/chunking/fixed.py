"""Fixed strategy: windows of about ``target_tokens`` over the text, cut at word boundaries.

For text without usable structure. Windows overlap by ``overlap_ratio`` of the target. There
are no parent chunks.
"""

from __future__ import annotations

import re
from array import array
from collections.abc import Sequence

from rag_engine.chunking.assemble import ChunkAssembler, Group, policy_fingerprint
from rag_engine.chunking.pack import Packer
from rag_engine.domain import (
    Chunk,
    ChunkPolicy,
    DocumentVersionId,
    ParsedDocument,
)
from rag_engine.ports import TokenCounter

_WORD = re.compile(r"\S+")


class FixedChunker:
    chunker_id = "fixed"
    version = "1"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def fingerprint(self, policy: ChunkPolicy, normalizer_version: str) -> str:
        return policy_fingerprint(self, self._counter, policy, normalizer_version)

    def chunk(
        self,
        document: ParsedDocument,
        policy: ChunkPolicy,
        *,
        version_id: DocumentVersionId,
        normalizer_version: str,
    ) -> Sequence[Chunk]:
        text = document.normalized_text
        if not text.strip():
            return []
        packer = Packer(text, self._counter, policy)
        starts = array("l", (m.start() for m in _WORD.finditer(text)))
        spans = self._windows(packer, starts)
        assembler = ChunkAssembler(
            document,
            self._counter,
            policy,
            version_id,
            self.fingerprint(policy, normalizer_version),
        )
        return assembler.assemble([Group(leaves=tuple(packer.merge_small_tail(spans)))], packer)

    @staticmethod
    def _windows(packer: Packer, starts: array[int]) -> list[tuple[int, int]]:
        words, text_end = len(starts), len(packer.text)

        def end_of(last_word: int) -> int:
            return starts[last_word + 1] if last_word + 1 < words else text_end

        def fits(first: int, last: int, limit: int) -> bool:
            return packer.count(starts[first], end_of(last)) <= limit

        def longest(first: int, limit: int, guess: int) -> int:
            """Index of the last word such that words ``first..last`` fit in ``limit`` tokens."""
            low, high = first, min(words - 1, first + max(guess, 1))
            while high < words - 1 and fits(first, high, limit):
                low, high = high, min(words - 1, high + max(guess, 1))
            if fits(first, high, limit):
                return high
            while high - low > 1:
                middle = (low + high) // 2
                if fits(first, middle, limit):
                    low = middle
                else:
                    high = middle
            return low

        spans: list[tuple[int, int]] = []
        first, rate = 0, 0.25  # words per token, refined as windows are measured
        while first < words:
            last = longest(first, packer.target, int(packer.target * rate))
            spans.append((starts[first], end_of(last)))
            if last >= words - 1:
                break
            rate = max((last - first + 1) / packer.target, 0.01)
            next_first = last + 1
            if packer.overlap_tokens:
                tail = longest_tail(
                    packer, starts, first, last, packer.overlap_tokens, end_of(last)
                )
                if tail is not None and tail > first:
                    next_first = tail
            first = next_first
        return spans


def longest_tail(
    packer: Packer, starts: array[int], first: int, last: int, limit: int, end: int
) -> int | None:
    """The first word of the longest tail of ``first..last`` that fits in ``limit`` tokens."""
    low, high = first + 1, last  # candidate start indices
    if low > high or packer.count(starts[high], end) > limit:
        return None
    while low < high:
        middle = (low + high) // 2
        if packer.count(starts[middle], end) <= limit:
            high = middle
        else:
            low = middle + 1
    return high
