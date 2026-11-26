"""Kill an indexed publication after FTS writes but before its transaction commits."""

from __future__ import annotations

import os
import signal
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from rag_engine.domain import EmbeddingModel, EmbeddingModelId, JobKind, SyncStatus, TextHash
from rag_engine.embedding.service import EmbeddingPlan
from rag_engine.ingestion.jobs import JobQueue
from rag_engine.ingestion.publishing import IndexPublisher
from rag_engine.ports import StoredChunk, StoredEmbedding
from rag_engine.storage.sqlite import open_store
from rag_engine.storage.sqlite.repository import SqliteWriter
from rag_engine.testing import FakeClock
from tests.factories import (
    T0,
    make_chunk,
    make_document,
    make_ids,
    make_source,
    make_sync_state,
    make_version,
)


def main() -> None:
    database = Path(sys.argv[1])
    mode = sys.argv[2]
    ids = make_ids()
    clock = FakeClock()
    source = make_source(ids)
    document = make_document(document_id=ids.document_id(), source_id=source.id, version_id=None)
    version = replace(
        make_version(version_id=ids.document_version_id(), document_id=document.id),
        normalized_text=zlib.compress(b"hello world"),
    )
    with open_store(database, clock) as store:
        if mode == "init":
            state = make_sync_state(
                source.id,
                document.source_key.value,
                status=SyncStatus.CHANGED,
                last_content_hash=version.content_hash,
            )
            with store.unit_of_work().begin() as writer:
                writer.put_source(source)
                writer.put_document(document)
                writer.put_sync_state(state)
            queue = JobQueue(store, clock, ids)
            queue.enqueue(
                JobKind.INGEST_DOC,
                source.id,
                document.id,
                {"content_hash": version.content_hash.hex},
            )
            assert queue.claim() is not None
            return
        (job,) = store.reader().list_jobs()
        model = EmbeddingModel(
            id=EmbeddingModelId.build(
                provider="mock", name="hash", revision="1", dimension=4, normalized=True
            ),
            provider="mock",
            name="hash",
            revision="1",
            dimension=4,
            normalized=True,
            max_input_tokens=64,
            created_at=T0,
        )
        chunk = make_chunk(version_id=version.id)
        key = TextHash.of_text(chunk.text)
        stored = StoredChunk(chunk, key, chunk.text, "")
        plan = EmbeddingPlan(
            MappingProxyType({chunk.id: key}),
            (StoredEmbedding(key, np.array([1, 0, 0, 0], dtype=np.float32), 1.0, T0),),
        )
        if mode == "kill":
            original = SqliteWriter.index_chunks

            def kill_after_fts(self: SqliteWriter, version_id: object) -> None:
                original(self, version_id)  # type: ignore[arg-type]
                os.kill(os.getpid(), signal.SIGKILL)

            SqliteWriter.index_chunks = kill_after_fts  # type: ignore[method-assign]
        assert IndexPublisher(store, clock).publish(job, version, [stored], plan, model, None)


if __name__ == "__main__":
    main()
