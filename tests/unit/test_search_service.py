"""The search service over a real flat vector index, a fake lexical channel and fake lookups."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence, Set
from dataclasses import dataclass, field

import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import (
    ChunkId,
    ChunkRef,
    ErrorCode,
    Language,
    RagError,
    RetrievalFilters,
    RetrievalMode,
    RetrievalQuery,
    ScoredChunk,
    TextHash,
)
from rag_engine.ports import ChunkInfo, Vectors
from rag_engine.retrieval.query import QueryProcessor
from rag_engine.retrieval.service import IndexSnapshot, RetrievalService, SearchOutcome
from rag_engine.storage.vector_flat import FlatVectorIndex
from rag_engine.testing import HashingEmbedder
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOCS = {name: IDS.document_id() for name in ("alpha", "beta", "gamma")}

CORPUS = {
    0: ("alpha", "The indexing pipeline stores chunks in SQLite and builds a vector matrix."),
    1: ("alpha", "Backups copy the database file while the writer lock is held."),
    2: ("beta", "Reciprocal rank fusion merges the dense and the lexical rankings."),
    3: ("beta", "Xylophone calibration table for serial number ZX-4471 instruments."),
    4: ("gamma", "The indexing pipeline stores chunks in SQLite and builds a vector matrix."),
    5: ("gamma", "Weekly menu: soup, bread and a salad with olive oil."),
}


def cid(ordinal: int) -> ChunkId:
    return ChunkId.derive(VERSION, "service", ordinal)


class Spy(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(64)
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> Vectors:
        self.queries.append(text)
        return await super().embed_query(text)


class WordIndex:
    """A lexical channel: score is the number of query words a chunk contains."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __len__(self) -> int:
        return len(CORPUS)

    def upsert(self, documents: Sequence[object]) -> None:  # pragma: no cover - unused port part
        raise NotImplementedError

    def remove(self, ids: Sequence[ChunkId]) -> None:  # pragma: no cover - unused port part
        raise NotImplementedError

    def search(self, query: str, k: int, allowed: Set[ChunkId] | None = None) -> list[ScoredChunk]:
        self.calls.append((query, k))
        words = set(query.lower().split())
        scored = []
        for ordinal, (_, text) in CORPUS.items():
            hits = len(words & set(text.lower().replace(",", " ").replace(".", " ").split()))
            if hits and (allowed is None or cid(ordinal) in allowed):
                scored.append(ScoredChunk(cid(ordinal), float(hits)))
        return sorted(scored, key=lambda item: (-item.score, str(item.chunk_id)))[:k]


@dataclass
class Lookups:
    allowed: frozenset[ChunkId] | None = None
    resolved: list[RetrievalFilters] = field(default_factory=list)

    def resolve(self, filters: RetrievalFilters) -> frozenset[ChunkId] | None:
        self.resolved.append(filters)
        return self.allowed

    def describe(self, ids: Sequence[ChunkId]) -> dict[ChunkId, ChunkInfo]:
        return {
            cid(o): ChunkInfo(
                ChunkRef(cid(o), DOCS[CORPUS[o][0]], VERSION), TextHash.of_text(CORPUS[o][1])
            )
            for o in range(len(CORPUS))
            if cid(o) in set(ids)
        }


def build(
    *, lookups: Lookups | None = None, snapshot: IndexSnapshot | None = None, prefix: str = ""
) -> tuple[RetrievalService, Spy, WordIndex]:
    embedder = Spy()
    vectors = FlatVectorIndex(embedder.model_id, embedder.dimension)
    matrix = asyncio.run(embedder.embed_documents([text for _, text in CORPUS.values()]))
    vectors.upsert([cid(o) for o in CORPUS], matrix)
    lexical = WordIndex()
    lookups = lookups or Lookups()
    state = snapshot or IndexSnapshot(generation=3, model_id=embedder.model_id)
    service = RetrievalService(
        embedder=embedder,
        query_prefix=prefix,
        vectors=vectors,
        lexical=lexical,
        filters=lookups,
        catalog=lookups,
        queries=QueryProcessor(HeuristicTokenCounter(), max_tokens=64),
        snapshot=lambda: state,
    )
    return service, embedder, lexical


def ask(
    service: RetrievalService, text: str, mode: RetrievalMode, **kwargs: object
) -> SearchOutcome:
    query = RetrievalQuery(request_id=IDS.request_id(), text=text, mode=mode, **kwargs)  # type: ignore[arg-type]
    return asyncio.run(service.search(query))


def names(outcome: SearchOutcome) -> list[int]:
    return [
        next(o for o in CORPUS if cid(o) == hit.chunk_ref.chunk_id) for hit in outcome.result.hits
    ]


def test_dense_mode_ranks_by_meaning_and_reports_only_dense_scores() -> None:
    service, _, lexical = build()
    outcome = ask(service, "how are chunks stored in SQLite", RetrievalMode.DENSE, top_k=3)
    assert names(outcome)[0] in {0, 4}
    hit = outcome.result.hits[0]
    assert hit.rank == 1 and hit.scores.dense is not None
    assert hit.scores.lexical is None and hit.scores.fused is not None
    assert lexical.calls == []  # the lexical channel is not touched in dense mode
    assert outcome.result.index_generation == 3


def test_lexical_mode_never_embeds_the_query() -> None:
    service, embedder, _ = build()
    outcome = ask(service, "ZX-4471 xylophone", RetrievalMode.LEXICAL, top_k=3)
    assert names(outcome) == [3] and embedder.queries == []
    assert outcome.result.hits[0].scores.dense is None


def test_hybrid_finds_a_rare_term_the_dense_channel_ranks_low() -> None:
    service, _, _ = build()
    outcome = ask(service, "serial ZX-4471", RetrievalMode.HYBRID, top_k=3)
    assert names(outcome)[0] == 3
    top = outcome.result.hits[0]
    assert top.scores.lexical is not None and top.scores.dense is not None


def test_exact_duplicate_texts_collapse_and_are_counted() -> None:
    service, _, _ = build()
    outcome = ask(service, "indexing pipeline stores chunks", RetrievalMode.HYBRID, top_k=6)
    assert sum(1 for o in names(outcome) if o in {0, 4}) == 1
    assert outcome.diagnostics.collapsed_duplicates == 1


def test_the_model_query_prefix_is_applied_once_and_candidate_k_reaches_the_channels() -> None:
    service, embedder, lexical = build(prefix="query: ")
    ask(service, "  vector   matrix ", RetrievalMode.HYBRID, top_k=2, candidate_k=7)
    assert embedder.queries == ["query: vector matrix"]
    assert lexical.calls == [("vector matrix", 7)]


def test_filters_restrict_both_channels() -> None:
    lookups = Lookups(allowed=frozenset({cid(5), cid(1)}))
    service, _, _ = build(lookups=lookups)
    outcome = ask(service, "database backups writer", RetrievalMode.HYBRID, top_k=5)
    assert set(names(outcome)) <= {1, 5}
    assert outcome.diagnostics.filter_matched == 2


def test_a_filter_that_matches_nothing_gives_an_empty_result_without_searching() -> None:
    service, embedder, lexical = build(lookups=Lookups(allowed=frozenset()))
    outcome = ask(service, "anything", RetrievalMode.HYBRID)
    assert outcome.result.hits == () and "filter_matched_zero" in outcome.result.fallbacks
    assert embedder.queries == [] and lexical.calls == []


def test_a_query_embedded_by_another_model_is_refused() -> None:
    other = HashingEmbedder(64, revision="2").model_id
    service, _, _ = build(snapshot=IndexSnapshot(generation=1, model_id=other))
    with pytest.raises(RagError) as info:
        ask(service, "text", RetrievalMode.DENSE)
    assert info.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH
    assert names(ask(service, "database", RetrievalMode.LEXICAL))  # lexical needs no model


def test_an_index_without_a_published_model_gives_an_empty_result() -> None:
    service, _, _ = build(snapshot=IndexSnapshot(generation=0, model_id=None))
    outcome = ask(service, "text", RetrievalMode.HYBRID)
    assert outcome.result.hits == () and "no_index" in outcome.result.fallbacks


def test_a_silent_channel_is_reported_as_a_fallback() -> None:
    service, _, _ = build()
    outcome = ask(service, "qqqq wwww", RetrievalMode.HYBRID, top_k=3)
    assert "lexical_empty" in outcome.result.fallbacks
    assert outcome.result.hits  # the dense channel still answers


def test_the_same_query_gives_the_same_answer() -> None:
    service, _, _ = build()
    first = ask(service, "fusion of rankings", RetrievalMode.HYBRID, top_k=4)
    second = ask(service, "fusion of rankings", RetrievalMode.HYBRID, top_k=4)
    one, two = first.result, second.result
    assert (one.hits, one.index_generation, one.fallbacks) == (
        two.hits,
        two.index_generation,
        two.fallbacks,
    )
    assert names(first)[0] == 2


def test_diagnostics_carry_channel_ranks_and_timings() -> None:
    service, _, _ = build()
    outcome = ask(service, "serial ZX-4471", RetrievalMode.HYBRID, top_k=2, debug=True)
    diagnostics = outcome.diagnostics
    top = outcome.result.hits[0].chunk_ref.chunk_id
    dense_rank, lexical_rank = diagnostics.channel_ranks[top]
    assert lexical_rank == 1 and dense_rank is not None
    assert set(diagnostics.timings_ms) >= {"query", "filter", "dense", "lexical", "fuse"}
    assert diagnostics.language is Language.EN


def test_missing_channels_for_a_mode_are_a_configuration_error() -> None:
    embedder = Spy()
    lookups = Lookups()
    bare = RetrievalService(
        embedder=None,
        query_prefix="",
        vectors=None,
        lexical=None,
        filters=lookups,
        catalog=lookups,
        queries=QueryProcessor(HeuristicTokenCounter(), max_tokens=64),
        snapshot=lambda: IndexSnapshot(1, embedder.model_id),
    )
    for mode in RetrievalMode:
        with pytest.raises(RagError) as info:
            ask(bare, "text", mode)
        assert info.value.code is ErrorCode.E_CONFIG_INVALID
