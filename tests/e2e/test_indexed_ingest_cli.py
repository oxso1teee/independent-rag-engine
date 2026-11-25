"""Configured rag ingest scans, embeds and publishes before it returns."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.domain import SourceKey
from rag_engine.ports import LexicalDocument, LexicalIndex
from rag_engine.storage.lexical import SqliteLexicalIndex
from rag_engine.storage.sqlite import open_store_read_only
from rag_engine.storage.vector_flat import FlatVectorIndex

DIGEST = "a" * 64


def test_configured_ingest_publishes_current_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["truncate"] is False
        return httpx.Response(
            200, json={"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in payload["input"]]}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            base_url=kwargs["base_url"], transport=httpx.MockTransport(handler)
        ),
    )
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# Heading\n\nA paragraph about local indexing and storage.\n")
    db = tmp_path / "rag.db"
    config = tmp_path / "rag.toml"
    config.write_text(
        "[parsing]\nisolate = []\n[embedding]\nenabled = true\n"
        f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\n'
        'dimension = 4\nbase_url = "http://test"\n'
    )
    container = DefaultContainer()
    prefix = ["--config", str(config), "--db", str(db)]
    assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
    assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
    store = open_store_read_only(db)
    try:
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        document = store.reader().get_document_by_key(source.id, SourceKey("a.md"))
        assert document is not None and document.current_version_id is not None
        assert store.reader().list_chunks(document.current_version_id)
        metadata = store.reader().get_index_metadata()
        assert metadata is not None and metadata.index_generation == 1
    finally:
        store.close()
    assert metadata.active_embedding_model_id is not None
    index = FlatVectorIndex(metadata.active_embedding_model_id, 4)
    assert index.refresh(db)
    assert not index.refresh(db)
    assert len(index) >= 1
    assert index.search(np.asarray([1, 0, 0, 0], dtype=np.float32), 1)
    lexical = SqliteLexicalIndex(db)
    english_hits = lexical.search("indexing", 5)
    assert english_hits
    assert lexical.search("indexing", 5, set()) == []
    assert lexical.search("indexing", 5, {english_hits[0].chunk_id}) == english_hits[:1]
    (root / "a.md").write_text("# Новые заметки\n\nМашины быстро едут по дороге.\n")
    assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
    assert index.refresh(db)
    assert index.generation == 2
    assert not lexical.search("indexing", 5)
    assert isinstance(lexical, LexicalIndex)
    russian_hits = lexical.search("машина", 5)
    assert len(russian_hits) == 1
    store = open_store_read_only(db)
    try:
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        document = store.reader().get_document_by_key(source.id, SourceKey("a.md"))
        assert document is not None and document.current_version_id is not None
        chunk = next(
            item.chunk
            for item in store.reader().list_chunks(document.current_version_id)
            if item.chunk.id == russian_hits[0].chunk_id
        )
    finally:
        store.close()
    lexical.remove([chunk.id])
    assert not lexical.search("машина", 5)
    lexical.upsert(
        [LexicalDocument(chunk.id, chunk.text, " ".join(chunk.heading_path), chunk.language)]
    )
    assert lexical.search("машина", 5)
    (root / "a.md").unlink()
    assert (
        main(
            [*prefix, "ingest", "notes", "--accept-mass-delete", "--json"],
            container=container,
        )
        == 0
    )
    assert not lexical.search("машина", 5)
