"""The rerank stage: a bounded, time-limited call that can never fail a search.

Reranking is an optional refinement. Whatever goes wrong (an exception, a timeout, a model that
is still busy with an earlier call, output that is not a reordering of the input), the stage
returns no ranking and a reason, and the caller keeps the fused order and reports
``rerank_skipped``. A timed-out call cannot be interrupted, so the stage refuses new work until it
has finished instead of piling concurrent inference onto a small machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from rag_engine.ports import Candidate, RerankBudget, RerankedCandidate, Reranker


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    ranked: tuple[RerankedCandidate, ...]
    skipped: str | None = None


class RerankStage:
    def __init__(self, reranker: Reranker, budget: RerankBudget) -> None:
        self._reranker = reranker
        self._budget = budget
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-rerank")
        self._active: Future[Sequence[RerankedCandidate]] | None = None

    @property
    def budget(self) -> RerankBudget:
        return self._budget

    def run(self, query: str, candidates: Sequence[Candidate]) -> RerankOutcome:
        pool = list(candidates[: self._budget.pool_size])
        if not pool:
            return RerankOutcome(())
        if self._active is not None and not self._active.done():
            return RerankOutcome((), "busy")
        self._active = self._executor.submit(self._reranker.rerank, query, pool, self._budget)
        try:
            ranked = tuple(self._active.result(timeout=self._budget.timeout_s))
        except FutureTimeout:
            return RerankOutcome((), "timeout")
        except Exception:
            return RerankOutcome((), "error")
        if not _is_reordering(pool, ranked):
            return RerankOutcome((), "invalid_output")
        return RerankOutcome(ranked)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _is_reordering(pool: Sequence[Candidate], ranked: Sequence[RerankedCandidate]) -> bool:
    wanted = {item.chunk_ref.chunk_id for item in pool}
    seen = [item.candidate.chunk_ref.chunk_id for item in ranked]
    return len(seen) == len(set(seen)) and set(seen) <= wanted
