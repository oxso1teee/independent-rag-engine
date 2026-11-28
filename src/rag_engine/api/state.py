"""What the routes share: the container, the settings and the long-lived query services.

Management routes open the store per request, exactly like the command line does. The query
services are heavier (an embedding model, the vector matrix, the chat provider), so they are
opened once, on the first query, and kept until the server stops; a server that never receives a
query never loads them.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from rag_engine.api.limits import IdempotencyStore
from rag_engine.cli.context import AskServices, Container, SearchServices
from rag_engine.config import Settings
from rag_engine.config.settings import ApiSettings
from rag_engine.domain import ErrorCode, RagError
from rag_engine.observability import get_logger
from rag_engine.pipeline import RagPipeline
from rag_engine.storage.sqlite import StoreNotInitialised

_log = get_logger("rag_engine.api")


@dataclass(frozen=True, slots=True)
class QueryBackend:
    search: SearchServices
    pipeline: RagPipeline | None
    ask: AskServices | None


@dataclass
class ApiState:
    container: Container
    config: Path | None
    db: Path
    settings: Settings
    bind: ApiSettings
    token: str | None
    managed_root: Path
    idempotency: IdempotencyStore
    _stack: ExitStack = field(default_factory=ExitStack)
    _backend: QueryBackend | None = None
    _open_lock: threading.Lock = field(default_factory=threading.Lock)
    _index_lock: threading.Lock = field(default_factory=threading.Lock)
    _index_requested: bool = False

    # ---------------------------------------------------------------- query services

    def query_backend(self) -> QueryBackend:
        """The query services, opened on first use (blocking: call from a worker thread)."""
        with self._open_lock:
            if self._backend is None:
                try:
                    if self.settings.llm.enabled:
                        ask = self._stack.enter_context(
                            self.container.ask(self.config, self.db, release_models=False)
                        )
                        self._backend = QueryBackend(ask.search, ask.pipeline, ask)
                    else:
                        search = self._stack.enter_context(
                            self.container.search(self.config, self.db)
                        )
                        self._backend = QueryBackend(search, None, None)
                except StoreNotInitialised:
                    raise RagError(
                        ErrorCode.E_SOURCE_NOT_FOUND,
                        "there is no index yet; add and ingest a source",
                    ) from None
            return self._backend

    def close(self) -> None:
        with self._open_lock:
            self._stack.close()
            self._backend = None

    # ----------------------------------------------------------------- indexing

    def drain_queue(self) -> None:
        """Process queued indexing jobs; one drain at a time, re-run if asked meanwhile."""
        self._index_requested = True
        if not self._index_lock.acquire(blocking=False):
            return  # the running drain will see the flag and go round again
        try:
            while self._index_requested:
                self._index_requested = False
                try:
                    with self.container.open(self.config, self.db, read_only=False) as services:
                        if services.index_worker is None:
                            _log.warning("api.index_skipped", reason="embedding_not_configured")
                            return
                        outcomes = services.index_worker.run_until_idle()
                        _log.info("api.indexed", jobs=len(outcomes))
                except RagError as error:
                    _log.error("api.index_failed", error_code=error.code.value)
                    return
        finally:
            self._index_lock.release()
