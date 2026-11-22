"""Packing text pieces into chunk spans.

A *piece* is a span of the normalised text that the strategies treat as indivisible while
packing (a paragraph, a sentence, a table, a code block, a group of words). The packer joins
consecutive pieces up to the target size, adds overlap between neighbouring prose chunks and
guarantees that no span exceeds ``max_tokens``. Everything works on offsets into one text, so a
chunk's text is always an exact slice of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_engine.chunking.split import split_hard, split_lines, split_sentences, split_words
from rag_engine.domain import Block, BlockKind, ChunkPolicy
from rag_engine.ports import TokenCounter

ATOMIC_KINDS = frozenset({BlockKind.TABLE, BlockKind.CODE})


@dataclass(frozen=True, slots=True)
class Piece:
    start: int
    end: int
    tokens: int
    prose: bool  # may take part in overlap


class Packer:
    def __init__(self, text: str, counter: TokenCounter, policy: ChunkPolicy) -> None:
        self.text = text
        self.counter = counter
        self.target = policy.target_tokens
        self.max = policy.max_tokens
        self.min = policy.min_tokens
        self.overlap_tokens = round(policy.overlap_ratio * policy.target_tokens)

    # -- measuring ---------------------------------------------------------------------

    def count(self, start: int, end: int) -> int:
        return self.counter.count(self.text[start:end])

    # -- pieces ------------------------------------------------------------------------

    def pieces_of_block(self, block: Block) -> list[Piece]:
        """Pieces of one block: the whole block when it fits, otherwise its parts."""
        atomic = block.kind in ATOMIC_KINDS
        tokens = self.count(block.char_start, block.char_end)
        limit = self.max if atomic else self.target
        if tokens <= limit:
            return [Piece(block.char_start, block.char_end, tokens, not atomic)]
        if atomic:
            return self._pieces(split_lines(block.text), block.char_start, prose=False)
        return self._pieces(split_sentences(block.text), block.char_start, prose=True)

    def _pieces(self, parts: list[str], offset: int, *, prose: bool) -> list[Piece]:
        pieces: list[Piece] = []
        for part in parts:
            end = offset + len(part)
            tokens = self.count(offset, end)
            if tokens <= self.max:
                pieces.append(Piece(offset, end, tokens, prose))
            else:
                pieces.extend(self._split_oversized(offset, end, prose))
            offset = end
        return pieces

    def _split_oversized(self, start: int, end: int, prose: bool) -> list[Piece]:
        """A sentence or line above ``max_tokens``: groups of words, then characters."""
        pieces: list[Piece] = []
        group_start, group_tokens, offset = start, 0, start
        for word in split_words(self.text[start:end]):
            word_end = offset + len(word)
            tokens = self.count(offset, word_end)
            if tokens > self.max:
                if group_tokens:
                    pieces.append(Piece(group_start, offset, group_tokens, prose))
                pieces.extend(self._hard(offset, word_end, prose))
                group_start, group_tokens = word_end, 0
            elif group_tokens + tokens > self.target and group_tokens:
                pieces.append(Piece(group_start, offset, group_tokens, prose))
                group_start, group_tokens = offset, tokens
            else:
                group_tokens += tokens
            offset = word_end
        if group_tokens:
            pieces.append(Piece(group_start, end, group_tokens, prose))
        return pieces

    def _hard(self, start: int, end: int, prose: bool) -> list[Piece]:
        """Cut an unbreakable run into character pieces that fit ``max_tokens``."""
        tokens = max(self.count(start, end), 1)
        size = max(1, min(end - start, int((end - start) * self.target / tokens)))
        while True:
            parts = split_hard(self.text[start:end], size)
            counts = [self.counter.count(p) for p in parts]
            if all(c <= self.max for c in counts) or size == 1:
                break
            size = max(1, size // 2)
        pieces: list[Piece] = []
        offset = start
        for part, count in zip(parts, counts, strict=True):
            pieces.append(Piece(offset, offset + len(part), count, prose))
            offset += len(part)
        return pieces

    # -- packing -----------------------------------------------------------------------

    def pack(self, pieces: list[Piece]) -> list[tuple[int, int]]:
        """Spans of consecutive pieces; neighbouring prose chunks overlap."""
        spans: list[tuple[int, int]] = []
        if not pieces:
            return spans
        start, end, tokens = pieces[0].start, pieces[0].end, pieces[0].tokens
        last = pieces[0]
        for piece in pieces[1:]:
            if tokens + piece.tokens <= self.target:
                end, tokens, last = piece.end, tokens + piece.tokens, piece
                continue
            spans.append((start, end))
            chunk_start = start
            start, tokens = piece.start, piece.tokens
            overlap = self._overlap_start(chunk_start, last, piece)
            if overlap is not None:
                gained = self.count(overlap, piece.start)
                if gained + piece.tokens <= self.target:
                    start, tokens = overlap, tokens + gained
            end, last = piece.end, piece
        spans.append((start, end))
        return self.merge_small_tail(spans)

    def _overlap_start(self, chunk_start: int, last: Piece, following: Piece) -> int | None:
        """Where the next chunk starts so that it repeats the end of the previous prose."""
        if not (self.overlap_tokens and last.prose and following.prose):
            return None
        region = self.text[last.start : last.end]
        kept = 0
        offset = last.end
        for sentence in reversed(split_sentences(region)):
            tokens = self.counter.count(sentence)
            if kept + tokens > self.overlap_tokens:
                break
            kept += tokens
            offset -= len(sentence)
        if offset == last.end:  # not even one sentence fits: try whole words
            for word in reversed(split_words(region)):
                tokens = self.counter.count(word)
                if kept + tokens > self.overlap_tokens:
                    break
                kept += tokens
                offset -= len(word)
        if offset >= last.end or (offset <= last.start and last.start == last.end):
            return None
        return offset if offset < following.start else None

    def merge_small_tail(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(spans) >= 2 and self.count(*spans[-1]) < self.min:
            merged = (spans[-2][0], spans[-1][1])
            if self.count(*merged) <= self.max:
                return [*spans[:-2], merged]
        return spans

    def fit(self, start: int, end: int) -> list[tuple[int, int]]:
        """``[(start, end)]``, or the span cut in two (recursively) until every part fits."""
        if self.count(start, end) <= self.max or end - start <= 1:
            return [(start, end)]
        middle = self._split_point(start, end)
        return [*self.fit(start, middle), *self.fit(middle, end)]

    def _split_point(self, start: int, end: int) -> int:
        centre = (start + end) // 2
        window = self.text[start:end]
        before = window.rfind(" ", 0, centre - start)
        after = window.find(" ", centre - start)
        for found in (before, after):
            if 0 < found < end - start - 1:
                return start + found + 1
        return centre
