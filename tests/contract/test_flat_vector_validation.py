"""The flat index rejects malformed input and never serves a damaged snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from rag_engine.domain import ChunkId, EmbeddingModelId, ErrorCode, RagError
from rag_engine.storage.vector_flat import FlatVectorIndex
from tests.factories import make_ids

MODEL = EmbeddingModelId.build(
    provider="fake", name="flat", revision="1", dimension=4, normalized=True
)
VERSION = make_ids().document_version_id()
IDS = [ChunkId.derive(VERSION, "flat", i) for i in range(3)]
UNIT = np.asarray([[1, 0, 0, 0]], dtype=np.float32)


@pytest.mark.parametrize(("dimension", "memory"), [(0, 8), (3, 8), (4, 0)])
def test_invalid_limits_are_refused(dimension: int, memory: int) -> None:
    with pytest.raises(RagError) as info:
        FlatVectorIndex(MODEL, dimension, max_memory_mb=memory)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


@pytest.mark.parametrize(
    ("ids", "vectors", "code"),
    [
        (IDS[:2], UNIT, ErrorCode.E_EMBED_DIM_MISMATCH),
        (
            IDS[:1],
            np.asarray([[1, 0, 0, np.nan]], dtype=np.float32),
            ErrorCode.E_EMBED_DIM_MISMATCH,
        ),
        (IDS[:1], np.asarray([[1e30, 0, 0, 0]], dtype=np.float32), ErrorCode.E_EMBED_DIM_MISMATCH),
        ([IDS[0], IDS[0]], np.tile(UNIT, (2, 1)), ErrorCode.E_INTEGRITY),
    ],
    ids=["shape", "nan", "fp16-overflow", "duplicate-ids"],
)
def test_malformed_upserts_change_nothing(
    ids: list[ChunkId], vectors: np.ndarray, code: ErrorCode
) -> None:
    index = FlatVectorIndex(MODEL, 4)
    with pytest.raises(RagError) as info:
        index.upsert(ids, vectors)
    assert info.value.code is code
    assert len(index) == 0


def test_memory_limit_is_enforced_before_growth() -> None:
    index = FlatVectorIndex(MODEL, 4, max_memory_mb=1)
    many = np.ones((200_000, 4), dtype=np.float32)
    with pytest.raises(RagError) as info:
        index.upsert([ChunkId.derive(VERSION, "flat", i) for i in range(200_000)], many)
    assert info.value.code is ErrorCode.E_RESOURCE_GATE
    assert len(index) == 0


def test_remove_ignores_empty_and_drops_only_named_ids() -> None:
    index = FlatVectorIndex(MODEL, 4)
    index.upsert(IDS, np.eye(4, dtype=np.float32)[:3])
    index.remove([])
    assert len(index) == 3
    index.remove([IDS[1]])
    hits = index.search(np.asarray([0, 1, 0, 0], dtype=np.float32), 3)
    assert IDS[1] not in {hit.chunk_id for hit in hits} and len(index) == 2


def test_search_validates_the_query() -> None:
    index = FlatVectorIndex(MODEL, 4)
    index.upsert(IDS[:1], UNIT)
    assert index.search(UNIT[0], 0) == []
    assert [hit.chunk_id for hit in index.search(UNIT, 1)] == IDS[:1]  # shape (1, d) is accepted
    for bad in (np.zeros(4), np.asarray([np.nan, 0, 0, 0]), np.ones(3)):
        with pytest.raises(RagError) as info:
            index.search(bad.astype(np.float32), 1)
        assert info.value.code is ErrorCode.E_EMBED_DIM_MISMATCH


def test_refresh_refuses_a_foreign_snapshot(tmp_path: Path) -> None:
    broken = tmp_path / "not-sqlite.db"
    broken.write_bytes(b"x" * 4096)
    with pytest.raises((RagError, sqlite3.Error)):
        FlatVectorIndex(MODEL, 4).refresh(broken)
