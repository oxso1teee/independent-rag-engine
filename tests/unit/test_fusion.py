"""Reciprocal rank fusion and weighted-score fusion, checked against hand calculations."""

from __future__ import annotations

import pytest

from rag_engine.domain import ChunkId, ErrorCode, RagError, ScoredChunk
from rag_engine.retrieval.fusion import fuse_rrf, fuse_weighted, single_channel
from tests.factories import make_ids

pytestmark = pytest.mark.unit

VERSION = make_ids().document_version_id()
A, B, C, D = (ChunkId.derive(VERSION, "fusion", i) for i in range(4))


def ranked(*pairs: tuple[ChunkId, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk, score) for chunk, score in pairs]


def test_rrf_sums_weighted_reciprocal_ranks() -> None:
    dense = ranked((A, 0.9), (B, 0.8), (C, 0.1))
    lexical = ranked((C, 7.0), (A, 3.0), (D, 1.0))
    fused = fuse_rrf(dense, lexical, k=60, dense_weight=1.0, lexical_weight=0.5)
    expected = {
        A: 1 / 61 + 0.5 / 62,
        B: 1 / 62,
        C: 1 / 63 + 0.5 / 61,
        D: 0.5 / 63,  # absent from the dense channel: rank infinity, no contribution
    }
    assert {item.chunk_id: item.fused for item in fused} == pytest.approx(expected)
    assert [item.chunk_id for item in fused] == sorted(expected, key=lambda c: -expected[c])


def test_rrf_keeps_every_component_for_diagnostics() -> None:
    fused = fuse_rrf(ranked((A, 0.9), (B, 0.8)), ranked((B, 5.0)))
    by_id = {item.chunk_id: item for item in fused}
    assert (by_id[A].dense, by_id[A].dense_rank) == (0.9, 1)
    assert (by_id[A].lexical, by_id[A].lexical_rank) == (None, None)
    assert (by_id[B].dense, by_id[B].dense_rank, by_id[B].lexical, by_id[B].lexical_rank) == (
        0.8,
        2,
        5.0,
        1,
    )


def test_ties_are_broken_by_chunk_id_so_the_order_is_stable() -> None:
    first, second = sorted((A, B))
    dense = ranked((second, 1.0), (first, 0.5))
    lexical = ranked((first, 1.0), (second, 0.5))
    fused = fuse_rrf(dense, lexical, dense_weight=1.0, lexical_weight=1.0)  # symmetric: a true tie
    assert fused[0].fused == pytest.approx(fused[1].fused)
    assert [item.chunk_id for item in fused] == [first, second]
    assert fuse_rrf(dense, lexical, dense_weight=1.0, lexical_weight=1.0) == fused


def test_an_empty_channel_leaves_the_other_one() -> None:
    fused = fuse_rrf(ranked((A, 0.9), (B, 0.8)), [])
    assert [item.chunk_id for item in fused] == [A, B]
    assert fuse_rrf([], []) == []


def test_weighted_fusion_normalises_each_channel_to_unit_range() -> None:
    dense = ranked((A, 0.9), (B, 0.5), (C, 0.1))  # normalised 1.0, 0.5, 0.0
    lexical = ranked((C, 8.0), (A, 4.0))  # normalised 1.0, 0.0
    fused = fuse_weighted(dense, lexical, alpha=0.7)
    expected = {A: 0.7 * 1.0 + 0.3 * 0.0, B: 0.7 * 0.5, C: 0.7 * 0.0 + 0.3 * 1.0}
    assert {item.chunk_id: item.fused for item in fused} == pytest.approx(expected)
    assert [item.chunk_id for item in fused] == [A, B, C]


def test_weighted_fusion_of_equal_scores_does_not_divide_by_zero() -> None:
    fused = fuse_weighted(ranked((A, 0.5), (B, 0.5)), ranked((A, 2.0)), alpha=0.5)
    assert {item.chunk_id: item.fused for item in fused} == pytest.approx({A: 1.0, B: 0.5})


def test_single_channel_keeps_the_order_and_the_raw_score() -> None:
    dense_only = single_channel(ranked((B, 0.8), (A, 0.9)), channel_name="dense")
    assert [item.chunk_id for item in dense_only] == [B, A]  # the channel's own order is kept
    assert dense_only[0].dense == 0.8 and dense_only[0].lexical is None
    assert dense_only[0].fused == 0.8
    lexical_only = single_channel(ranked((A, 3.0)), channel_name="lexical")
    assert lexical_only[0].lexical == 3.0 and lexical_only[0].dense is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: fuse_rrf(ranked((A, 1.0), (A, 0.5)), []),
        lambda: fuse_weighted([], ranked((B, 1.0), (B, 1.0)), alpha=0.5),
        lambda: single_channel(ranked((C, 1.0), (C, 1.0)), channel_name="dense"),
    ],
)
def test_a_chunk_twice_in_one_channel_is_an_integrity_error(call: object) -> None:
    with pytest.raises(RagError) as info:
        call()  # type: ignore[operator]
    assert info.value.code is ErrorCode.E_INTEGRITY


@pytest.mark.parametrize("k", [0, -5])
def test_rrf_constant_must_be_positive(k: int) -> None:
    with pytest.raises(RagError) as info:
        fuse_rrf([], [], k=k)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_must_be_between_zero_and_one(alpha: float) -> None:
    with pytest.raises(RagError) as info:
        fuse_weighted([], [], alpha=alpha)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
