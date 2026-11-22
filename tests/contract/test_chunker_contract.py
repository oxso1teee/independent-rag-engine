"""Contract every ``Chunker`` implementation must satisfy.

A strategy is added to ``FACTORIES``; the same documents, policies and properties then run
against it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_engine.chunking.fixed import FixedChunker
from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import BlockKind, Chunk, ChunkPolicy, ParsedDocument
from rag_engine.ports import Chunker, TokenCounter
from tests.chunking_checks import (
    DEFAULT_POLICY,
    NO_OVERLAP,
    SMALL_POLICY,
    check_chunks,
    document_of,
    overlap_tokens_between,
    prose,
)
from tests.factories import make_ids

COUNTER = HeuristicTokenCounter()
VERSION = make_ids().document_version_id()
P, H, L, T, C, Q = (
    BlockKind.PARAGRAPH,
    BlockKind.HEADING,
    BlockKind.LIST_ITEM,
    BlockKind.TABLE,
    BlockKind.CODE,
    BlockKind.QUOTE,
)

FACTORIES: dict[str, Callable[[TokenCounter], Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
}


class OtherCounter:
    counter_id = "other/1"

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture(params=sorted(FACTORIES))
def chunker(request: pytest.FixtureRequest) -> Chunker:
    return FACTORIES[request.param](COUNTER)


def run(
    chunker: Chunker, document: ParsedDocument, policy: ChunkPolicy = SMALL_POLICY
) -> Sequence[Chunk]:
    chunks = chunker.chunk(document, policy, version_id=VERSION, normalizer_version="norm/1")
    check_chunks(document, chunks, policy, COUNTER, chunker.fingerprint(policy, "norm/1"))
    return chunks


DOCUMENTS: dict[str, ParsedDocument] = {
    "prose": document_of((P, prose(30)), (P, prose(5, start=100))),
    "paragraphs": document_of(*[(P, prose(2, start=i * 10)) for i in range(30)]),
    "structured": document_of(
        (H, "Guide", 1),
        (P, prose(12)),
        (H, "Setup", 2),
        (L, prose(1)),
        (L, prose(1, start=5)),
        (T, "\n".join(f"row{i} | value{i} | more{i}" for i in range(40))),
        (C, "\n".join(f"line_{i} = compute({i})" for i in range(60))),
        (Q, prose(3, start=50)),
    ),
    "one long sentence": document_of((P, " ".join(f"w{i}" for i in range(400)) + ".")),
    "one giant word": document_of((P, "x" * 5_000)),
    "russian": document_of(
        (P, " ".join(["Это достаточно длинное русское предложение с числами 123."] * 40))
    ),
    "repeated": document_of((P, " ".join(["same"] * 400) + ".")),
    "tiny": document_of((P, "short")),
    "table only": document_of((T, "\n".join(f"{i} | {i * i}" for i in range(200)))),
}


def test_satisfies_the_protocol(chunker: Chunker) -> None:
    assert isinstance(chunker, Chunker)
    assert chunker.chunker_id and chunker.version


def test_fingerprint_follows_every_parameter(chunker: Chunker) -> None:
    base = chunker.fingerprint(DEFAULT_POLICY, "norm/1")
    assert base == chunker.fingerprint(DEFAULT_POLICY, "norm/1")
    variants = {
        chunker.fingerprint(ChunkPolicy(target_tokens=200, max_tokens=384), "norm/1"),
        chunker.fingerprint(ChunkPolicy(max_tokens=400), "norm/1"),
        chunker.fingerprint(ChunkPolicy(min_tokens=40), "norm/1"),
        chunker.fingerprint(ChunkPolicy(overlap_ratio=0.2), "norm/1"),
        chunker.fingerprint(DEFAULT_POLICY, "norm/2"),
        FACTORIES[chunker.chunker_id](OtherCounter()).fingerprint(DEFAULT_POLICY, "norm/1"),
    }
    assert base not in variants and len(variants) == 6


def test_an_empty_document_has_no_chunks(chunker: Chunker) -> None:
    assert run(chunker, document_of()) == []
    assert run(chunker, ParsedDocument(normalized_text="")) == []


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
@pytest.mark.parametrize(
    "policy", [SMALL_POLICY, NO_OVERLAP, DEFAULT_POLICY], ids=["small", "no-overlap", "default"]
)
def test_documents_are_chunked_within_bounds(
    chunker: Chunker, name: str, policy: ChunkPolicy
) -> None:
    chunks = run(chunker, DOCUMENTS[name], policy)
    assert chunks
    assert all(c.token_count <= policy.max_tokens for c in chunks if c.level == 0)


def test_a_small_document_is_one_chunk_with_its_text(chunker: Chunker) -> None:
    (chunk,) = run(chunker, DOCUMENTS["tiny"], DEFAULT_POLICY)
    assert chunk.text == "short" and chunk.level == 0 and chunk.parent_id is None


def test_chunking_is_deterministic(chunker: Chunker) -> None:
    for document in DOCUMENTS.values():
        assert run(chunker, document) == run(chunker, document)


def test_repeated_text_is_marked_as_duplicate_and_not_indexed(chunker: Chunker) -> None:
    chunks = run(chunker, DOCUMENTS["repeated"], NO_OVERLAP)
    flags = [(c.is_duplicate, c.indexed) for c in chunks if c.level == 0]
    assert flags[0] == (False, True)
    assert any(duplicate and not indexed for duplicate, indexed in flags)


def test_overlap_repeats_the_end_of_the_previous_chunk(chunker: Chunker) -> None:
    document = DOCUMENTS["prose"]
    overlapping = run(chunker, document, SMALL_POLICY)
    plain = run(chunker, document, NO_OVERLAP)
    shared = [overlap_tokens_between(a, b, COUNTER) for a, b in itertools.pairwise(overlapping)]
    limit = SMALL_POLICY.target_tokens * SMALL_POLICY.overlap_ratio + 1
    assert any(tokens > 0 for tokens in shared)
    assert all(tokens <= limit for tokens in shared)
    assert all(overlap_tokens_between(a, b, COUNTER) == 0 for a, b in itertools.pairwise(plain))


def test_a_giant_word_is_cut_not_dropped(chunker: Chunker) -> None:
    chunks = run(chunker, DOCUMENTS["one giant word"], NO_OVERLAP)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == "x" * 5_000


def test_chunks_carry_provenance(chunker: Chunker) -> None:
    for chunk in run(chunker, DOCUMENTS["russian"]):
        assert chunk.language.value == "ru"
        assert chunk.document_version_id == VERSION
        assert chunk.heading_path == ()


_text = st.text(alphabet=st.sampled_from(list("abcdefgh ABCDEFGH.,!?\nЖзик0123 ")), max_size=400)
_blocks = st.lists(st.tuples(st.sampled_from([P, L, Q, T, C, H]), _text), max_size=12)


@given(_blocks)
@settings(max_examples=60, deadline=None)
def test_random_documents_satisfy_every_invariant(blocks: list[tuple[BlockKind, str]]) -> None:
    document = document_of(
        *[(kind, text, 1) if kind is H else (kind, text) for kind, text in blocks]
    )
    for factory in FACTORIES.values():
        chunker = factory(COUNTER)
        for policy in (SMALL_POLICY, NO_OVERLAP):
            run(chunker, document, policy)
