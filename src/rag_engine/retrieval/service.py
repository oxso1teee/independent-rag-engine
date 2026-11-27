"""The search pipeline: query, filter, dense and lexical channels, fusion, diversity.

The service owns ordering and nothing else: it reads published data through ports and never
touches SQL or a model directly. One call is a function of the query and the index snapshot, so
repeating it returns the same ranking; timings live only in the diagnostics.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from rag_engine.domain import (
    ChunkId,
    EmbeddingModelId,
    ErrorCode,
    HitScores,
    Language,
    RagError,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
    ScoredChunk,
)
from rag_engine.ports import (
    ChunkCatalog,
    ChunkFilter,
    EmbeddingProvider,
    LexicalIndex,
    VectorIndex,
)
from rag_engine.retrieval.diversity import diversify
from rag_engine.retrieval.fusion import (
    FusedCandidate,
    fuse_rrf,
    fuse_weighted,
    single_channel,
)
from rag_engine.retrieval.query import QueryProcessor


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """What the index holds right now; the provider of this value has already refreshed it."""

    generation: int
    model_id: EmbeddingModelId | None


@dataclass(frozen=True, slots=True)
class FusionSettings:
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 0.5
    alpha: float = 0.5


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    mode: RetrievalMode
    language: Language
    query_tokens: int
    filter_matched: int | None
    dense_candidates: int
    lexical_candidates: int
    fused_candidates: int
    collapsed_duplicates: int
    capped: int
    unknown: int
    document_share: float
    channel_ranks: Mapping[ChunkId, tuple[int | None, int | None]] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    result: RetrievalResult
    diagnostics: RetrievalDiagnostics


class RetrievalService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider | None,
        query_prefix: str,
        vectors: VectorIndex | None,
        lexical: LexicalIndex | None,
        filters: ChunkFilter,
        catalog: ChunkCatalog,
        queries: QueryProcessor,
        snapshot: Callable[[], IndexSnapshot],
        fusion: FusionSettings | None = None,
        max_per_document: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._prefix = query_prefix
        self._vectors = vectors
        self._lexical = lexical
        self._filters = filters
        self._catalog = catalog
        self._queries = queries
        self._snapshot = snapshot
        self._fusion = fusion or FusionSettings()
        self._max_per_document = max_per_document

    def _require_channels(self, mode: RetrievalMode) -> None:
        if mode is not RetrievalMode.LEXICAL and (self._embedder is None or self._vectors is None):
            raise RagError(
                ErrorCode.E_CONFIG_INVALID, "dense retrieval needs an embedding model and index"
            )
        if mode is not RetrievalMode.DENSE and self._lexical is None:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "lexical retrieval needs a lexical index")

    def _fuse(self, dense: list[ScoredChunk], lexical: list[ScoredChunk]) -> list[FusedCandidate]:
        settings = self._fusion
        if settings.method == "weighted":
            return fuse_weighted(dense, lexical, alpha=settings.alpha)
        return fuse_rrf(
            dense,
            lexical,
            k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            lexical_weight=settings.lexical_weight,
        )

    async def search(self, query: RetrievalQuery) -> SearchOutcome:
        mode = query.mode
        self._require_channels(mode)
        timings: dict[str, float] = {}
        clock = time.perf_counter

        started = clock()
        processed = self._queries.process(query.text, query.language_hint)
        timings["query"] = (clock() - started) * 1000

        state = self._snapshot()
        fallbacks: list[str] = []

        def finish(
            hits: tuple[RetrievalHit, ...],
            *,
            matched: int | None = None,
            dense: int = 0,
            lexical: int = 0,
            fused: int = 0,
            diversity: tuple[int, int, int, float] = (0, 0, 0, 0.0),
            ranks: Mapping[ChunkId, tuple[int | None, int | None]] | None = None,
        ) -> SearchOutcome:
            result = RetrievalResult(
                request_id=query.request_id,
                index_generation=state.generation,
                embedding_model_id=state.model_id,
                hits=hits,
                fallbacks=tuple(fallbacks),
            )
            return SearchOutcome(
                result,
                RetrievalDiagnostics(
                    mode,
                    processed.language,
                    processed.tokens,
                    matched,
                    dense,
                    lexical,
                    fused,
                    *diversity,
                    channel_ranks=dict(ranks or {}),
                    timings_ms=dict(timings),
                ),
            )

        if state.model_id is None:
            fallbacks.append("no_index")
            return finish(())
        uses_dense = mode is not RetrievalMode.LEXICAL
        if uses_dense and self._embedder is not None and self._embedder.model_id != state.model_id:
            raise RagError(
                ErrorCode.E_EMBED_MODEL_MISMATCH, "the query model differs from the index model"
            )

        started = clock()
        allowed = self._filters.resolve(query.filters)
        timings["filter"] = (clock() - started) * 1000
        matched = None if allowed is None else len(allowed)
        if allowed is not None and not allowed:
            fallbacks.append("filter_matched_zero")
            return finish((), matched=0)

        dense_hits: list[ScoredChunk] = []
        started = clock()
        if uses_dense and self._embedder is not None and self._vectors is not None:
            vector = await self._embedder.embed_query(self._prefix + processed.text)
            dense_hits = list(self._vectors.search(vector, query.candidate_k, allowed))
        timings["dense"] = (clock() - started) * 1000

        lexical_hits: list[ScoredChunk] = []
        started = clock()
        if mode is not RetrievalMode.DENSE and self._lexical is not None:
            lexical_hits = list(self._lexical.search(processed.text, query.candidate_k, allowed))
        timings["lexical"] = (clock() - started) * 1000

        started = clock()
        if mode is RetrievalMode.DENSE:
            fused = single_channel(dense_hits, channel_name="dense")
        elif mode is RetrievalMode.LEXICAL:
            fused = single_channel(lexical_hits, channel_name="lexical")
        else:
            if not dense_hits:
                fallbacks.append("dense_empty")
            if not lexical_hits:
                fallbacks.append("lexical_empty")
            fused = self._fuse(dense_hits, lexical_hits)
        timings["fuse"] = (clock() - started) * 1000

        started = clock()
        catalog = self._catalog.describe([item.chunk_id for item in fused])
        diverse = diversify(
            fused, catalog, top_k=query.top_k, max_per_document=self._max_per_document
        )
        timings["diversity"] = (clock() - started) * 1000

        hits = tuple(
            RetrievalHit(
                chunk_ref=info.ref,
                rank=rank,
                scores=HitScores(
                    dense=item.dense, lexical=item.lexical, fused=item.fused, rerank=None
                ),
            )
            for rank, (item, info) in enumerate(diverse.kept, start=1)
        )
        ranks = {item.chunk_id: (item.dense_rank, item.lexical_rank) for item, _ in diverse.kept}
        return finish(
            hits,
            matched=matched,
            dense=len(dense_hits),
            lexical=len(lexical_hits),
            fused=len(fused),
            diversity=(
                diverse.collapsed_duplicates,
                diverse.capped,
                diverse.unknown,
                diverse.document_share,
            ),
            ranks=ranks,
        )
