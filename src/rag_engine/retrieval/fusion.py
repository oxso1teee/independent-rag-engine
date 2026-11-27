"""Combining the dense and lexical channels into one ranking.

Both channels arrive best-first with their raw scores. Fusion keeps every component of every
candidate, so a result can always be explained; the fused score only orders candidates inside one
query and must never be read as a probability or compared between queries. Ties are broken by
chunk id, which makes the order a function of the inputs alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_engine.domain import ChunkId, ErrorCode, RagError, ScoredChunk


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk_id: ChunkId
    fused: float
    dense: float | None = None
    dense_rank: int | None = None
    lexical: float | None = None
    lexical_rank: int | None = None


def _positions(channel: Sequence[ScoredChunk]) -> dict[ChunkId, tuple[int, float]]:
    positions: dict[ChunkId, tuple[int, float]] = {}
    for rank, item in enumerate(channel, start=1):
        if item.chunk_id in positions:
            raise RagError(ErrorCode.E_INTEGRITY, "a chunk appears twice in one retrieval channel")
        positions[item.chunk_id] = (rank, item.score)
    return positions


def _ordered(items: list[FusedCandidate]) -> list[FusedCandidate]:
    return sorted(items, key=lambda item: (-item.fused, str(item.chunk_id)))


def _merge(
    dense: dict[ChunkId, tuple[int, float]],
    lexical: dict[ChunkId, tuple[int, float]],
    score: dict[ChunkId, float],
) -> list[FusedCandidate]:
    return _ordered(
        [
            FusedCandidate(
                chunk_id,
                value,
                dense=dense[chunk_id][1] if chunk_id in dense else None,
                dense_rank=dense[chunk_id][0] if chunk_id in dense else None,
                lexical=lexical[chunk_id][1] if chunk_id in lexical else None,
                lexical_rank=lexical[chunk_id][0] if chunk_id in lexical else None,
            )
            for chunk_id, value in score.items()
        ]
    )


def fuse_rrf(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 0.5,
) -> list[FusedCandidate]:
    """``score(d) = sum_c w_c / (k + rank_c(d))``; a channel that lacks ``d`` adds nothing."""
    if k < 1:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "the RRF constant must be positive")
    dense_at, lexical_at = _positions(dense), _positions(lexical)
    score: dict[ChunkId, float] = {}
    for weight, positions in ((dense_weight, dense_at), (lexical_weight, lexical_at)):
        for chunk_id, (rank, _) in positions.items():
            score[chunk_id] = score.get(chunk_id, 0.0) + weight / (k + rank)
    return _merge(dense_at, lexical_at, score)


def _normalised(positions: dict[ChunkId, tuple[int, float]]) -> dict[ChunkId, float]:
    if not positions:
        return {}
    values = [score for _, score in positions.values()]
    low, high = min(values), max(values)
    if high == low:
        return dict.fromkeys(positions, 1.0)  # no spread to rescale: every hit is as good as any
    return {key: (score - low) / (high - low) for key, (_, score) in positions.items()}


def fuse_weighted(
    dense: Sequence[ScoredChunk], lexical: Sequence[ScoredChunk], *, alpha: float = 0.5
) -> list[FusedCandidate]:
    """``alpha * dense + (1 - alpha) * lexical`` after min-max scaling inside each channel."""
    if not 0.0 <= alpha <= 1.0:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "alpha must be between 0 and 1")
    dense_at, lexical_at = _positions(dense), _positions(lexical)
    dense_scaled, lexical_scaled = _normalised(dense_at), _normalised(lexical_at)
    score = {
        chunk_id: alpha * dense_scaled.get(chunk_id, 0.0)
        + (1 - alpha) * lexical_scaled.get(chunk_id, 0.0)
        for chunk_id in dict.fromkeys([*dense_at, *lexical_at])
    }
    return _merge(dense_at, lexical_at, score)


def single_channel(
    channel: Sequence[ScoredChunk], *, channel_name: Literal["dense", "lexical"]
) -> list[FusedCandidate]:
    """One channel on its own: its order and raw scores are kept, the score is the fused value."""
    positions = _positions(channel)
    return [
        FusedCandidate(
            item.chunk_id,
            item.score,
            dense=item.score if channel_name == "dense" else None,
            dense_rank=positions[item.chunk_id][0] if channel_name == "dense" else None,
            lexical=item.score if channel_name == "lexical" else None,
            lexical_rank=positions[item.chunk_id][0] if channel_name == "lexical" else None,
        )
        for item in channel
    ]
