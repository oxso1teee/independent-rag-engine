"""Retrieval metrics against values worked out by hand, and a seeded bootstrap."""

from __future__ import annotations

import math

import pytest

from rag_engine.evaluation.bootstrap import bootstrap_mean, paired_bootstrap
from rag_engine.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit

# Ranking a, b, c, d, e with a and d relevant (grade 1)
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a": 1, "d": 1}


def test_recall_counts_relevant_found_over_all_relevant() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.5  # a of {a, d}
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5  # d is at rank 4
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0
    assert recall_at_k(RANKED, RELEVANT, 100) == 1.0  # k beyond the list is fine


def test_precision_divides_by_k_even_for_short_lists() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5  # 1 of 2
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5  # 2 of 4
    assert precision_at_k(["a"], RELEVANT, 4) == 0.25  # a missing slot counts as a miss


def test_reciprocal_rank_is_the_first_relevant_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == 1.0
    assert reciprocal_rank(["x", "y", "d"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "z"], RELEVANT) == 0.0
    assert reciprocal_rank(["x", "y", "d"], RELEVANT, k=2) == 0.0  # cut off before rank 3


def test_hit_is_one_when_any_relevant_is_in_the_top_k() -> None:
    assert hit_at_k(["x", "d"], RELEVANT, 2) == 1.0
    assert hit_at_k(["x", "d"], RELEVANT, 1) == 0.0


def test_binary_ndcg() -> None:
    # DCG = 1/log2(2) + 1/log2(5) ; ideal = 1/log2(2) + 1/log2(3)
    expected = (1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(expected)
    assert ndcg_at_k(["a", "d", "x"], RELEVANT, 3) == pytest.approx(1.0)


def test_graded_ndcg_uses_the_grade_as_the_gain() -> None:
    graded = {"c": 2, "a": 1}
    # ranked a, b, c: DCG = 1/1 + 2/log2(4) = 2.0 ; ideal order c, a: 2/1 + 1/log2(3)
    expected = 2.0 / (2.0 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], graded, 3) == pytest.approx(expected)


def test_ndcg_only_looks_at_the_first_k() -> None:
    # relevant d at rank 4 is invisible at k=3; ideal still has both within k
    expected = 1.0 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)


def test_duplicates_in_a_ranking_are_counted_once() -> None:
    assert recall_at_k(["a", "a", "a"], RELEVANT, 3) == 0.5
    assert precision_at_k(["a", "a", "a"], RELEVANT, 3) == pytest.approx(1 / 3)


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, ndcg_at_k, hit_at_k])
def test_a_query_without_relevant_documents_is_refused(metric: object) -> None:
    with pytest.raises(ValueError, match="relevant"):
        metric([], {}, 5)  # type: ignore[operator]
    with pytest.raises(ValueError, match="relevant"):
        metric(["a"], {"a": 0}, 5)  # type: ignore[operator]


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at_k(RANKED, RELEVANT, k)


def test_bootstrap_is_deterministic_for_a_seed_and_brackets_the_mean() -> None:
    values = [((index * 37) % 101) / 100 for index in range(40)]
    first = bootstrap_mean(values, seed=7, resamples=2000)
    assert first == bootstrap_mean(values, seed=7, resamples=2000)
    assert first != bootstrap_mean(values, seed=8, resamples=2000)
    assert first.mean == pytest.approx(sum(values) / len(values))
    assert first.low <= first.mean <= first.high


def test_bootstrap_of_constant_values_has_zero_width() -> None:
    result = bootstrap_mean([0.4] * 10, seed=1, resamples=500)
    assert (result.low, result.mean, result.high) == pytest.approx((0.4, 0.4, 0.4))


def test_paired_bootstrap_detects_a_consistent_gain_and_a_tie() -> None:
    base = [0.2, 0.4, 0.5, 0.3, 0.6, 0.1, 0.7, 0.5]
    better = [value + 0.1 for value in base]
    gain = paired_bootstrap(better, base, seed=3, resamples=2000)
    assert gain.mean == pytest.approx(0.1)
    assert gain.low == pytest.approx(0.1) and gain.high == pytest.approx(0.1)
    assert gain.excludes_zero
    tie = paired_bootstrap(base, base, seed=3, resamples=500)
    assert tie.mean == 0.0 and not tie.excludes_zero


def test_paired_bootstrap_needs_equal_non_empty_samples() -> None:
    with pytest.raises(ValueError, match="same"):
        paired_bootstrap([1.0], [1.0, 2.0], seed=1)
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean([], seed=1)
