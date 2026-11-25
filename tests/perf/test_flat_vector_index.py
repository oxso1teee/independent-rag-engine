"""Dense search acceptance workload: 10,000 synthetic 1024-dimensional vectors."""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from rag_engine.domain import ChunkId, EmbeddingModelId
from rag_engine.storage.vector_flat import FlatVectorIndex
from tests.factories import make_ids


@pytest.mark.perf
def test_10k_dense_search_p95_under_100ms() -> None:
    rng = np.random.default_rng(104)
    vectors = rng.standard_normal((10_000, 1024), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = EmbeddingModelId.build(
        provider="fake", name="perf", revision="1", dimension=1024, normalized=True
    )
    version = make_ids().document_version_id()
    ids = [ChunkId.derive(version, "perf", i) for i in range(len(vectors))]
    index = FlatVectorIndex(model, 1024, max_memory_mb=64)
    index.upsert(ids, vectors)
    queries = vectors[:25]
    elapsed: list[float] = []
    for query in queries:
        start = time.perf_counter()
        hits = index.search(query, 10)
        elapsed.append((time.perf_counter() - start) * 1000)
        assert len(hits) == 10
    p95_ms = sorted(elapsed)[23]
    assert p95_ms <= 100, (
        f"dense search p95={p95_ms:.1f}ms; median={statistics.median(elapsed):.1f}ms"
    )
