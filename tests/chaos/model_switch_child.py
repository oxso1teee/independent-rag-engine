"""Subprocess used to kill a model switch inside its final SQLite transaction."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import EmbeddingModel
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.maintenance import IndexMaintenance
from rag_engine.storage.sqlite import open_store
from rag_engine.storage.sqlite.repository import SqliteWriter
from rag_engine.system import SystemClock
from rag_engine.testing import HashingEmbedder


def main() -> None:
    database = Path(sys.argv[1])
    kill = sys.argv[2] == "kill"
    clock = SystemClock()
    provider = HashingEmbedder(8)
    model = EmbeddingModel(
        id=provider.model_id,
        provider="fake",
        name="hashing",
        revision="1",
        dimension=8,
        normalized=True,
        max_input_tokens=provider.max_input_tokens,
        created_at=clock.now(),
    )
    if kill:

        def kill_before_commit(self: SqliteWriter) -> int:
            del self
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL did not terminate the process")

        SqliteWriter.bump_generation = kill_before_commit  # type: ignore[method-assign]
    with open_store(database, clock) as store:
        service = EmbeddingService(provider, model, HeuristicTokenCounter(), clock)
        IndexMaintenance(store, clock).switch_model(model, service)


if __name__ == "__main__":
    main()
