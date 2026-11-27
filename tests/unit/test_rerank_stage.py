"""The rerank stage never fails a query: any problem falls back to the fused order."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from rag_engine.domain import ChunkId, ChunkRef, RagError
from rag_engine.domain.errors import ErrorCode
from rag_engine.ports import Candidate, RerankBudget, RerankedCandidate
from rag_engine.retrieval.rerank import RerankStage
from rag_engine.testing import FakeReranker
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOCUMENT = IDS.document_id()


def pool(*texts: str) -> list[Candidate]:
    return [
        Candidate(ChunkRef(ChunkId.derive(VERSION, "stage", i), DOCUMENT, VERSION), text, 1.0)
        for i, text in enumerate(texts)
    ]


def test_a_working_reranker_reorders_and_reports_no_skip() -> None:
    stage = RerankStage(FakeReranker(), RerankBudget(pool_size=10, timeout_s=5))
    outcome = stage.run("nightly backups", pool("menu", "nightly backups run", "backups"))
    assert outcome.skipped is None
    assert [item.candidate.text for item in outcome.ranked] == [
        "nightly backups run",
        "backups",
        "menu",
    ]
    stage.close()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("model crashed"), MemoryError(), RagError(ErrorCode.E_RESOURCE_GATE, "no ram")],
)
def test_any_error_is_a_skip_with_the_reason_error(failure: Exception) -> None:
    stage = RerankStage(FakeReranker(fail=failure), RerankBudget(timeout_s=5))
    outcome = stage.run("q", pool("a", "b"))
    assert outcome.ranked == () and outcome.skipped == "error"
    stage.close()


def test_a_slow_reranker_times_out_and_a_second_call_is_refused_while_it_runs() -> None:
    release = threading.Event()

    class Slow:
        def rerank(
            self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
        ) -> Sequence[RerankedCandidate]:
            release.wait(timeout=2)
            return [RerankedCandidate(c, 1.0) for c in candidates]

    stage = RerankStage(Slow(), RerankBudget(timeout_s=0.05))
    try:
        assert stage.run("q", pool("a")).skipped == "timeout"
        assert stage.run("q", pool("a")).skipped == "busy"  # the first call still holds the model
    finally:
        release.set()
        stage.close()


def test_output_that_is_not_a_reordering_of_the_input_is_discarded() -> None:
    stranger = Candidate(
        ChunkRef(ChunkId.derive(VERSION, "stage", 99), DOCUMENT, VERSION), "not in the input", 1.0
    )

    class Wrong:
        def rerank(
            self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
        ) -> Sequence[RerankedCandidate]:
            return [RerankedCandidate(stranger, 1.0)]

    class Twice:
        def rerank(
            self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
        ) -> Sequence[RerankedCandidate]:
            return [RerankedCandidate(candidates[0], 1.0), RerankedCandidate(candidates[0], 0.5)]

    for reranker in (Wrong(), Twice()):
        stage = RerankStage(reranker, RerankBudget())
        assert stage.run("q", pool("a", "b")).skipped == "invalid_output"
        stage.close()


def test_an_empty_pool_is_not_a_skip() -> None:
    reranker = FakeReranker()
    stage = RerankStage(reranker, RerankBudget())
    outcome = stage.run("q", [])
    assert outcome.ranked == () and outcome.skipped is None and reranker.calls == 0
    stage.close()


def test_the_pool_is_cut_to_the_budget_before_the_model_sees_it() -> None:
    reranker = FakeReranker()
    stage = RerankStage(reranker, RerankBudget(pool_size=2))
    outcome = stage.run("a", pool("a one", "a two", "a three"))
    assert len(outcome.ranked) == 2
    stage.close()


def test_releasing_the_stage_releases_a_reranker_that_can_be_released() -> None:
    class Releasable(FakeReranker):
        released = False

        def release(self) -> None:
            self.released = True

    ranker = Releasable()
    RerankStage(ranker, RerankBudget(pool_size=5, timeout_s=1.0)).release()
    assert ranker.released
    RerankStage(FakeReranker(), RerankBudget(pool_size=5, timeout_s=1.0)).release()  # no-op
