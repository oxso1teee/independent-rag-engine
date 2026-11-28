"""Latency and memory of the whole search pipeline over a real published index.

Publishes ``N`` chunks (with FTS rows and fp16 embeddings) through the production writer, then
times ``RetrievalService.search`` in dense, lexical and hybrid mode in a fresh process. The query
embedding is a precomputed random unit vector, so the numbers exclude the embedding model, as the
requirements' NFR-03 does. Run: ``uv run python eval/scripts/benchmark_search.py 10000``.
"""

from __future__ import annotations

import asyncio
import json
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import (
    Chunk,
    ChunkId,
    ChunkKind,
    ContentHash,
    Document,
    DocumentStatus,
    DocumentVersion,
    EmbeddingModel,
    EmbeddingModelId,
    Language,
    RetrievalMode,
    RetrievalQuery,
    Source,
    SourceKey,
    SourceKind,
    SourcePolicy,
    TextHash,
)
from rag_engine.domain.ids import Sha256Hex
from rag_engine.lexical.analysis import ANALYZER_ID, analyze
from rag_engine.ports import StoredChunk, StoredEmbedding, Vectors
from rag_engine.retrieval.query import QueryProcessor
from rag_engine.retrieval.service import IndexSnapshot, RetrievalService
from rag_engine.storage.filters import SqliteChunkCatalog, SqliteChunkFilter
from rag_engine.storage.lexical import SqliteLexicalIndex
from rag_engine.storage.sqlite import open_store
from rag_engine.storage.vector_flat import FlatVectorIndex
from rag_engine.system import SystemClock, system_id_factory

DIMENSION = 1024
PER_DOCUMENT = 100
NOW = datetime(2026, 1, 1, tzinfo=UTC)
VOCABULARY = [f"term{i}" for i in range(4000)]
MODEL_ID = EmbeddingModelId.build(
    provider="bench", name="synthetic", revision="1", dimension=DIMENSION, normalized=True
)


class RandomQueryEmbedder:
    """Returns a fixed random unit vector per call; only the search cost is of interest."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(5)

    @property
    def model_id(self) -> EmbeddingModelId:
        return MODEL_ID

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def max_input_tokens(self) -> int:
        return 512

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        raise NotImplementedError

    async def embed_query(self, text: str) -> Vectors:
        vector = self._rng.standard_normal((1, DIMENSION)).astype(np.float32)
        return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def build(database: Path, count: int) -> None:
    clock = SystemClock()
    ids = system_id_factory()
    rng = random.Random(11)  # noqa: S311 - reproducible synthetic text, not a secret
    weights = [1 / (rank + 1) for rank in range(len(VOCABULARY))]
    vectors = np.random.default_rng(3)
    with open_store(database, clock) as store:
        source = Source(
            id=ids.source_id(),
            name="bench",
            kind=SourceKind.FS,
            locator="/bench",
            locator_fingerprint=Sha256Hex.of_text("/bench"),
            policy=SourcePolicy(),
            created_at=NOW,
            updated_at=NOW,
        )
        model = EmbeddingModel(
            id=MODEL_ID,
            provider="bench",
            name="synthetic",
            revision="1",
            dimension=DIMENSION,
            normalized=True,
            max_input_tokens=512,
            created_at=NOW,
        )
        with store.unit_of_work().begin() as writer:
            writer.put_source(source)
            writer.put_embedding_model(model)
        for number in range(count // PER_DOCUMENT):
            document = Document(
                id=ids.document_id(),
                source_id=source.id,
                source_key=SourceKey(f"doc{number}.md"),
                current_version_id=None,
                status=DocumentStatus.ACTIVE,
                created_at=NOW,
                updated_at=NOW,
            )
            version = DocumentVersion(
                id=ids.document_version_id(),
                document_id=document.id,
                seq=1,
                content_hash=ContentHash.of_bytes(f"{number}".encode()),
                size_bytes=1,
                mtime_ns=1,
                mime="text/markdown",
                detected_by="extension",
                parser_id="md",
                parser_version="1",
                normalizer_version="1",
                chunker_fingerprint="bench",
                created_at=NOW,
            )
            stored: list[StoredChunk] = []
            embeddings: list[StoredEmbedding] = []
            for ordinal in range(PER_DOCUMENT):
                words = rng.choices(VOCABULARY, weights, k=30)
                text = " ".join(words) + f" unique{number}x{ordinal}"
                key = TextHash.of_text(text)
                chunk = Chunk(
                    id=ChunkId.derive(version.id, "bench", ordinal),
                    document_version_id=version.id,
                    ordinal=ordinal,
                    level=0,
                    parent_id=None,
                    kind=ChunkKind.PARAGRAPH,
                    text=text,
                    text_hash=key,
                    token_count=40,
                    char_start=ordinal * 1000,
                    char_end=ordinal * 1000 + len(text),
                    language=Language.EN,
                    indexed=True,
                )
                stored.append(StoredChunk(chunk, key, analyze(text, Language.EN), ""))
                vector = vectors.standard_normal(DIMENSION).astype(np.float32)
                embeddings.append(StoredEmbedding(key, vector / np.linalg.norm(vector), 1.0, NOW))
            with store.unit_of_work().begin() as writer:
                writer.put_document(document)
                writer.put_version(version)
                writer.put_many(model.id, embeddings)
                writer.put_chunks(version.id, stored)
                writer.index_chunks(version.id)
                writer.put_document(replace(document, current_version_id=version.id))
        with store.unit_of_work().begin() as writer:
            metadata = writer.get_index_metadata()
            assert metadata is not None  # noqa: S101 - a fresh store always has its singleton
            writer.update_index_metadata(
                replace(
                    metadata, active_embedding_model_id=MODEL_ID, lexical_analyzer_id=ANALYZER_ID
                )
            )
            writer.bump_generation()


def measure(database: Path) -> dict[str, object]:
    embedder = RandomQueryEmbedder()
    vectors = FlatVectorIndex(MODEL_ID, DIMENSION)
    started = time.perf_counter()
    vectors.refresh(database)
    load_seconds = time.perf_counter() - started
    service = RetrievalService(
        embedder=embedder,
        query_prefix="",
        vectors=vectors,
        lexical=SqliteLexicalIndex(database),
        filters=SqliteChunkFilter(database),
        catalog=SqliteChunkCatalog(database),
        queries=QueryProcessor(HeuristicTokenCounter(), max_tokens=64),
        snapshot=lambda: IndexSnapshot(1, MODEL_ID),
    )
    ids = system_id_factory()
    rng = random.Random(21)  # noqa: S311 - reproducible queries
    queries = [" ".join(rng.choices(VOCABULARY[:500], k=3)) for _ in range(60)]
    result: dict[str, object] = {"load_seconds": load_seconds, "chunks": len(vectors)}
    for mode in RetrievalMode:
        times: list[float] = []
        for text in queries:
            request = RetrievalQuery(request_id=ids.request_id(), text=text, mode=mode)
            began = time.perf_counter()
            outcome = asyncio.run(service.search(request))
            times.append((time.perf_counter() - began) * 1000)
            if not outcome.result.hits:
                raise SystemExit(f"{mode.value}: a query returned nothing")
        ordered = sorted(times[5:])  # drop warm-up calls
        result[mode.value] = {
            "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
        }
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print(json.dumps(measure(Path(sys.argv[2]))))
        return
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    with tempfile.TemporaryDirectory(prefix="rag-search-") as folder:
        database = Path(folder) / "bench.db"
        started = time.perf_counter()
        build(database, count)
        built = time.perf_counter() - started
        child = subprocess.run(  # noqa: S603
            [sys.executable, __file__, "--child", str(database)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(
            json.dumps(
                {
                    "chunks": count,
                    "database_mib": database.stat().st_size / 1048576,
                    "build_seconds": built,
                    **json.loads(child.stdout),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
