"""Flat vector index contract: exact scores, filtering, replacement and stable ties."""

from __future__ import annotations

import numpy as np
import pytest

from rag_engine.domain import ChunkId, EmbeddingModelId, ErrorCode, RagError
from rag_engine.ports import VectorIndex
from rag_engine.storage.vector_flat import FlatVectorIndex
from tests.factories import make_ids

MODEL = EmbeddingModelId.build(
    provider="fake", name="flat", revision="1", dimension=4, normalized=True
)
VERSION = make_ids().document_version_id()
IDS = [ChunkId.derive(VERSION, "flat", i) for i in range(4)]


def test_exact_search_and_allowed_filter() -> None:
    index: VectorIndex = FlatVectorIndex(MODEL, 4)
    assert isinstance(index, VectorIndex)
    index.upsert(
        IDS[:3],
        np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.float32),
    )
    hits = index.search(np.asarray([1, 0, 0, 0], dtype=np.float32), 3)
    assert [hit.chunk_id for hit in hits] == IDS[:3]
    assert [hit.score for hit in hits] == pytest.approx([1, 0, -1])
    assert [
        hit.chunk_id
        for hit in index.search(np.asarray([1, 0, 0, 0], dtype=np.float32), 3, {IDS[2]})
    ] == [IDS[2]]
    index.upsert([IDS[1]], np.asarray([[1, 0, 0, 0]], dtype=np.float32))
    tied = index.search(np.asarray([1, 0, 0, 0], dtype=np.float32), 2)
    assert [hit.chunk_id for hit in tied] == sorted(IDS[:2], key=str)
    index.remove([IDS[0], IDS[3]])
    assert len(index) == 2


def test_rejects_bad_shape_and_memory_budget() -> None:
    index = FlatVectorIndex(MODEL, 4, max_memory_mb=1)
    with pytest.raises(RagError) as bad:
        index.upsert([IDS[0]], np.zeros((1, 3), dtype=np.float32))
    assert bad.value.code is ErrorCode.E_EMBED_DIM_MISMATCH
    with pytest.raises(RagError) as over:
        index.upsert([IDS[0]], np.full((1, 4), 1e10, dtype=np.float32))
    assert over.value.code is ErrorCode.E_EMBED_DIM_MISMATCH
    large_model = EmbeddingModelId.build(
        provider="fake", name="large", revision="1", dimension=1024, normalized=True
    )
    small_budget = FlatVectorIndex(large_model, 1024, max_memory_mb=1)
    many_ids = [ChunkId.derive(VERSION, "large", i) for i in range(512)]
    with pytest.raises(RagError) as budget:
        small_budget.upsert(many_ids, np.zeros((512, 1024), dtype=np.float32))
    assert budget.value.code is ErrorCode.E_RESOURCE_GATE
    assert len(small_budget) == 0
