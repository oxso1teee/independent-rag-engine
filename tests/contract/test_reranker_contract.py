"""Every Reranker implementation orders a bounded pool by finite, descending scores."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from rag_engine.domain import ChunkRef
from rag_engine.ports import Candidate, RerankBudget, Reranker
from rag_engine.testing import FakeReranker
from tests.factories import make_ids

IDS = make_ids()
VERSION = IDS.document_version_id()
DOCUMENT = IDS.document_id()

TEXTS = [
    "backups run every night at 02:30",
    "the cafeteria opens at noon",
    "nightly backups are encrypted",
    "weekly menu with soup and bread",
    "restore tests of backups",
]


def candidates() -> list[Candidate]:
    from rag_engine.domain import ChunkId

    return [
        Candidate(
            ChunkRef(ChunkId.derive(VERSION, "rerank", i), DOCUMENT, VERSION), text, 1 - i / 10
        )
        for i, text in enumerate(TEXTS)
    ]


FACTORIES: dict[str, Callable[[], Reranker]] = {"fake": FakeReranker}


@pytest.fixture(params=list(FACTORIES))
def reranker(request: pytest.FixtureRequest) -> Reranker:
    return FACTORIES[request.param]()


def test_scores_are_finite_and_descending_and_cover_the_input(reranker: Reranker) -> None:
    pool = candidates()
    ranked = reranker.rerank("backups at night", pool, RerankBudget(pool_size=30))
    assert [item.score for item in ranked] == sorted((i.score for i in ranked), reverse=True)
    assert all(math.isfinite(item.score) for item in ranked)
    assert {item.candidate.chunk_ref.chunk_id for item in ranked} == {
        c.chunk_ref.chunk_id for c in pool
    }


def test_the_pool_size_bounds_the_result(reranker: Reranker) -> None:
    assert len(reranker.rerank("backups", candidates(), RerankBudget(pool_size=2))) <= 2


def test_an_empty_pool_gives_an_empty_result(reranker: Reranker) -> None:
    assert list(reranker.rerank("backups", [], RerankBudget())) == []


def test_the_result_is_deterministic(reranker: Reranker) -> None:
    first = reranker.rerank("backups at night", candidates(), RerankBudget())
    second = reranker.rerank("backups at night", candidates(), RerankBudget())
    assert list(first) == list(second)


def test_the_fake_prefers_the_passage_that_shares_the_query_words() -> None:
    ranked = FakeReranker().rerank("nightly backups", candidates(), RerankBudget())
    assert ranked[0].candidate.text == "nightly backups are encrypted"
