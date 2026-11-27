"""A deterministic reranker for tests: scores by the number of query words a passage contains."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

from rag_engine.ports import Candidate, RerankBudget, RerankedCandidate

_WORDS = re.compile(r"\w+")


def word_overlap(query: str, text: str) -> float:
    wanted = set(_WORDS.findall(query.casefold()))
    return float(len(wanted & set(_WORDS.findall(text.casefold()))))


class FakeReranker:
    """Orders by ``scorer(query, text)``; can be told to fail or to be slow."""

    def __init__(
        self,
        scorer: Callable[[str, str], float] = word_overlap,
        *,
        fail: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._scorer = scorer
        self._fail = fail
        self._delay_s = delay_s
        self.calls = 0

    def rerank(
        self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
    ) -> Sequence[RerankedCandidate]:
        self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._fail is not None:
            raise self._fail
        scored = [RerankedCandidate(c, self._scorer(query, c.text)) for c in candidates]
        scored.sort(key=lambda item: (-item.score, str(item.candidate.chunk_ref.chunk_id)))
        return scored[: budget.pool_size]
