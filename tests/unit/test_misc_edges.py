from __future__ import annotations

import asyncio
import io
import json
import socket

import numpy as np
import pytest

from rag_engine.domain import ChunkId, ChunkRef, RagError, Sha256Hex
from rag_engine.observability import configure_logging, get_logger
from rag_engine.ports import Candidate, RerankBudget, RerankedCandidate
from rag_engine.testing import HashingEmbedder
from tests.factories import make_chunk, make_ids

IDS = make_ids()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_network_access_is_blocked_in_tests() -> None:
    with pytest.raises(Exception, match="socket"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    with pytest.raises(Exception, match="socket"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_identifiers_render_as_their_canonical_text() -> None:
    assert str(Sha256Hex.of_text("x")) == Sha256Hex.of_text("x").hex
    chunk_id = ChunkId.derive(IDS.document_version_id(), "fp", 0)
    assert str(chunk_id) == chunk_id.value


def test_sets_and_lists_in_log_events_are_scrubbed() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("x").info("batch", ids=["a", "b"], flags={"x"}, secrets=["hidden"])
    event = json.loads(stream.getvalue())
    assert event["ids"] == ["a", "b"]
    assert event["secrets"] == "[REDACTED]"


def test_rerank_types_validate_their_inputs() -> None:
    version = IDS.document_version_id()
    chunk = make_chunk(version_id=version)
    ref = ChunkRef(chunk.id, IDS.document_id(), version)
    candidate = Candidate(ref, "text", 0.1)
    assert RerankedCandidate(candidate, -2.5).score == -2.5
    with pytest.raises(RagError):
        RerankedCandidate(candidate, float("nan"))
    for bad in ({"pool_size": 0}, {"timeout_s": 0.0}):
        with pytest.raises(RagError):
            RerankBudget(**bad)


def test_hashing_embedder_survives_feature_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = HashingEmbedder(16)
    slots = iter([(3, 1.0), (3, -1.0), (5, 1.0)])
    monkeypatch.setattr(embedder, "_slot", lambda _feature: next(slots))
    monkeypatch.setattr(embedder, "_features", lambda _text: [("a", 1.0), ("b", 1.0)])
    vector = asyncio.run(embedder.embed_documents(["anything"]))[0]
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)
    assert vector[5] == pytest.approx(1.0)
