"""The packing core: pieces, overlap rules, tail merging and the size guarantee."""

from __future__ import annotations

import itertools

from rag_engine.chunking.pack import Packer, Piece
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import BlockKind, ChunkPolicy
from tests.chunking_checks import document_of

COUNTER = HeuristicTokenCounter()
P, T, C = BlockKind.PARAGRAPH, BlockKind.TABLE, BlockKind.CODE


def words(count: int) -> str:
    return " ".join(f"w{i % 10}" for i in range(count))


def setup(
    *blocks: tuple[BlockKind, str],
    target: int = 20,
    maximum: int = 30,
    minimum: int = 5,
    overlap: float = 0.0,
) -> tuple[Packer, list[Piece]]:
    """A packer over the document made of ``blocks`` and the pieces of all its blocks."""
    document = document_of(*blocks)
    policy = ChunkPolicy(
        target_tokens=target, max_tokens=maximum, min_tokens=minimum, overlap_ratio=overlap
    )
    packer = Packer(document.normalized_text, COUNTER, policy)
    pieces = [piece for block in document.blocks for piece in packer.pieces_of_block(block)]
    return packer, pieces


def contiguous(spans: list[tuple[int, int]] | list[Piece]) -> bool:
    ends = [(s.start, s.end) if isinstance(s, Piece) else s for s in spans]
    return all(a[1] == b[0] for a, b in itertools.pairwise(ends))


def test_a_block_that_fits_is_one_piece() -> None:
    text = "A short paragraph."
    packer, pieces = setup((P, text))
    assert pieces == [Piece(0, len(text), COUNTER.count(text), True)]
    assert packer.count(0, len(text)) == pieces[0].tokens


def test_a_long_paragraph_is_split_into_sentences() -> None:
    text = " ".join(f"Sentence number {i} has a few words." for i in range(12))
    _, pieces = setup((P, text))
    assert len(pieces) == 12 and all(p.prose for p in pieces)
    assert pieces[0].start == 0 and pieces[-1].end == len(text)
    assert contiguous(pieces)


def test_tables_and_code_are_atomic_and_split_by_lines_only_when_too_large() -> None:
    small = "a | 1\nb | 2"
    _, (piece,) = setup((T, small))
    assert piece.prose is False and (piece.start, piece.end) == (0, len(small))

    big = "\n".join(f"row{i % 10} | value{i % 10} | more{i % 10}" for i in range(40))
    packer, pieces = setup((C, big))
    assert len(pieces) == 40 and not any(p.prose for p in pieces)
    assert all(packer.text[p.start : p.end].endswith("\n") for p in pieces[:-1])
    assert contiguous(pieces)


def test_a_sentence_above_the_maximum_is_cut_at_word_boundaries() -> None:
    text = words(300) + "."
    packer, pieces = setup((P, text))
    assert len(pieces) > 3
    assert all(p.tokens <= packer.max for p in pieces)
    assert pieces[0].start == 0 and pieces[-1].end == len(text)
    assert contiguous(pieces)
    assert all(packer.text[p.start] != " " for p in pieces)  # words stay whole


def test_a_word_above_the_maximum_is_cut_by_characters() -> None:
    text = "x" * 2_000
    packer, pieces = setup((P, text))
    assert len(pieces) > 1
    assert all(p.tokens <= packer.max for p in pieces)
    assert "".join(packer.text[p.start : p.end] for p in pieces) == text


def test_pieces_are_packed_up_to_the_target() -> None:
    text = " ".join(f"Sentence {i} is here." for i in range(20))
    packer, pieces = setup((P, text))
    spans = packer.pack(pieces)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    counts = [packer.count(*span) for span in spans]
    assert all(count <= packer.target for count in counts)
    assert all(count >= packer.target * 0.6 for count in counts[:-1])


def test_overlap_repeats_whole_sentences_of_the_previous_prose() -> None:
    text = " ".join(f"Sentence number {i} has words." for i in range(30))
    packer, pieces = setup((P, text), target=30, maximum=45, overlap=0.4)
    spans = packer.pack(pieces)
    assert len(spans) > 2
    for (first_start, first_end), (second_start, _) in itertools.pairwise(spans):
        assert first_start < second_start < first_end  # the next chunk starts inside the previous
        assert packer.count(second_start, first_end) <= packer.overlap_tokens + 1
        assert packer.text[second_start:].startswith("Sentence")  # on a sentence boundary


def test_no_overlap_next_to_tables_or_code() -> None:
    intro = ("Intro sentence one. Intro sentence two. " * 4).strip()
    table = "\n".join(f"{i % 10} | {i % 10}" for i in range(80))
    packer, pieces = setup((P, intro), (T, table), overlap=0.4, target=15, maximum=25)
    spans = packer.pack(pieces)
    table_start = packer.text.index("0 | 0")
    inside = [span for span in spans if span[0] >= table_start]
    assert len(inside) > 3
    previous_end = max(end for start, end in spans if start < table_start)
    for start, end in inside:
        assert start >= previous_end  # no line of the table is repeated
        previous_end = end
    prose = [span for span in spans if span[1] <= table_start]
    assert any(a[1] > b[0] for a, b in itertools.pairwise(prose))  # the intro does overlap


def test_a_small_last_chunk_is_merged_into_its_neighbour_when_it_fits() -> None:
    text = words(10)
    cut = text.index("w8")
    roomy, _ = setup((P, text), target=20, maximum=30, minimum=5)
    assert roomy.count(cut, len(text)) < roomy.min
    assert roomy.count(0, len(text)) <= roomy.max
    assert roomy.merge_small_tail([(0, cut), (cut, len(text))]) == [(0, len(text))]

    tight, _ = setup((P, text), target=8, maximum=9, minimum=5)
    assert tight.count(0, len(text)) > tight.max
    assert tight.merge_small_tail([(0, cut), (cut, len(text))]) == [(0, cut), (cut, len(text))]
    assert tight.merge_small_tail([(0, len(text))]) == [(0, len(text))]


def test_fit_guarantees_the_maximum() -> None:
    text = words(500)
    packer, _ = setup((P, text))
    spans = packer.fit(0, len(text))
    assert len(spans) > 1
    assert all(packer.count(*span) <= packer.max for span in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert contiguous(spans)
    assert packer.fit(0, 4) == [(0, 4)]


def test_nothing_to_pack_gives_no_spans() -> None:
    packer, pieces = setup()
    assert pieces == [] and packer.pack(pieces) == []
