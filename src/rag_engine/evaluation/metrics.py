"""Ranking metrics for one query.

A ranking is a sequence of ids, best first; repeated ids count once, at their first position.
``relevant`` maps an id to its grade; grade 0 means judged not relevant. Every metric needs at
least one relevant document: a query without one has no recall, and callers must filter those
out (they belong to the no-answer evaluation). ``nDCG`` uses the grade itself as the gain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _unique(ranked: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ranked))


def _positives(relevant: Mapping[str, int]) -> dict[str, int]:
    found = {key: grade for key, grade in relevant.items() if grade > 0}
    if not found:
        raise ValueError("the query has no relevant document")
    return found


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    _check_k(k)
    positives = _positives(relevant)
    found = sum(1 for key in _unique(ranked)[:k] if key in positives)
    return found / len(positives)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Relevant in the top ``k`` over ``k``: an empty slot counts as a miss."""
    _check_k(k)
    positives = _positives(relevant)
    return sum(1 for key in _unique(ranked)[:k] if key in positives) / k


def hit_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1.0 when any relevant document is in the top ``k``; its mean over queries is coverage."""
    _check_k(k)
    positives = _positives(relevant)
    return 1.0 if any(key in positives for key in _unique(ranked)[:k]) else 0.0


def reciprocal_rank(
    ranked: Sequence[str], relevant: Mapping[str, int], k: int | None = None
) -> float:
    if k is not None:
        _check_k(k)
    positives = _positives(relevant)
    for rank, key in enumerate(_unique(ranked)[:k], start=1):
        if key in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    _check_k(k)
    positives = _positives(relevant)
    gains = (positives.get(key, 0) for key in _unique(ranked)[:k])
    actual = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_gains = sorted(positives.values(), reverse=True)[:k]
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return actual / ideal


def ndcg_from_gains(gains: Sequence[float], ideal: Sequence[float], k: int) -> float:
    """nDCG@k when the gain of each ranked item and the best possible gains are already known.

    ``gains`` are in ranked order; ``ideal`` holds the gains of the best achievable ranking.
    """
    _check_k(k)
    best = sum(g / math.log2(rank + 1) for rank, g in enumerate(sorted(ideal, reverse=True)[:k], 1))
    if best <= 0:
        raise ValueError("the query has no relevant document")
    got = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains[:k], start=1))
    return got / best
