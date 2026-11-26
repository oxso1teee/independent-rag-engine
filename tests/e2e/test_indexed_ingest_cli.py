"""Configured rag ingest scans, embeds and publishes before it returns."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.domain import EmbeddingModel, ErrorCode, ModelStatus, RagError, SourceKey, TextHash
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.maintenance import IndexMaintenance
from rag_engine.ports import LexicalDocument, LexicalIndex, StoredEmbedding
from rag_engine.storage.integrity import check_integrity
from rag_engine.storage.lexical import SqliteLexicalIndex
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.storage.vector_flat import FlatVectorIndex
from rag_engine.system import SystemClock
from rag_engine.testing import HashingEmbedder

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
    assert main([*prefix, "doctor", "--integrity", "--json"], container=container) == 0
    assert check_integrity(db).ok
    store = open_store_read_only(db)
    try:
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        document = store.reader().get_document_by_key(source.id, SourceKey("a.md"))
        assert document is not None and document.current_version_id is not None
        first_chunk = next(
            stored
            for stored in store.reader().list_chunks(document.current_version_id)
            if stored.chunk.indexed
        )
        first_embedding_key = first_chunk.embedding_input_hash
        assert first_embedding_key is not None
        metadata = store.reader().get_index_metadata()
        assert metadata is not None and metadata.index_generation == 1
    finally:
        store.close()
    assert metadata.active_embedding_model_id is not None
    corrupted = tmp_path / "corrupted.db"
    with (
        closing(sqlite3.connect(db)) as backup_source,
        closing(sqlite3.connect(corrupted)) as target,
    ):
        backup_source.backup(target)
    with closing(sqlite3.connect(corrupted)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("UPDATE embeddings SET norm = 5")
        connection.execute("UPDATE documents SET current_version_id = 'invalid'")
        connection.execute("UPDATE index_state SET index_generation = 0")
        connection.execute("UPDATE chunks SET text = 'tampered'")
        connection.execute("DELETE FROM chunks_fts_docsize")
        connection.commit()
    before = hashlib.sha256(corrupted.read_bytes()).digest()
    corruption = check_integrity(corrupted)
    assert hashlib.sha256(corrupted.read_bytes()).digest() == before
    assert not corruption.ok
    assert any("vector norm" in error for error in corruption.errors)
    assert any("current_version_id" in error for error in corruption.errors)
    assert any("chunk text offset mismatch" in error for error in corruption.errors)
    assert any("chunk text hash mismatch" in error for error in corruption.errors)
    assert any("positive generation" in error for error in corruption.errors)
    assert any("missing from FTS" in error for error in corruption.errors)
    orphaned = tmp_path / "orphaned.db"
    with (
        closing(sqlite3.connect(db)) as backup_source,
        closing(sqlite3.connect(orphaned)) as target,
    ):
        backup_source.backup(target)
    with closing(sqlite3.connect(orphaned)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("UPDATE embeddings SET model_id = 'orphan'")
        connection.commit()
    orphan_report = check_integrity(orphaned)
    assert any("foreign key violation" in error for error in orphan_report.errors)
    assert any("orphan embedding model" in error for error in orphan_report.errors)
    assert (
        main(["--db", str(corrupted), "doctor", "--integrity", "--json"], container=container) != 0
    )
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
    assert check_integrity(db).ok
    with open_store_read_only(db) as retained_store:
        assert first_embedding_key in retained_store.reader().get_many(
            metadata.active_embedding_model_id, [first_embedding_key]
        )
    assert index.refresh(db)
    assert index.generation == 2
    assert not lexical.search("indexing", 5)
    assert isinstance(lexical, LexicalIndex)
    russian_hits = lexical.search("машина", 5)
    assert len(russian_hits) == 1
    interrupted = tmp_path / "interrupted-switch.db"
    with (
        closing(sqlite3.connect(db)) as backup_source,
        closing(sqlite3.connect(interrupted)) as target,
    ):
        backup_source.backup(target)
    child = [sys.executable, "-m", "tests.chaos.model_switch_child", str(interrupted)]
    killed = subprocess.run(  # noqa: S603
        [*child, "kill"], capture_output=True, check=False, cwd=Path(__file__).resolve().parents[2]
    )
    assert killed.returncode == -9, killed.stderr.decode()
    assert check_integrity(interrupted).ok
    with open_store_read_only(interrupted) as interrupted_store:
        interrupted_metadata = interrupted_store.reader().get_index_metadata()
        assert interrupted_metadata is not None
        assert interrupted_metadata.active_embedding_model_id == metadata.active_embedding_model_id
        assert interrupted_metadata.index_generation == 2
    resumed = subprocess.run(  # noqa: S603
        [*child, "finish"],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert resumed.returncode == 0, resumed.stderr.decode()
    assert check_integrity(interrupted).ok
    with open_store_read_only(interrupted) as recovered_store:
        recovered_metadata = recovered_store.reader().get_index_metadata()
        assert recovered_metadata is not None
        assert recovered_metadata.active_embedding_model_id == HashingEmbedder(8).model_id
        assert recovered_metadata.index_generation == 3
    assert main([*prefix, "index", "switch-model", "--json"], container=container) == 0
    clock = SystemClock()
    provider = HashingEmbedder(8)
    replacement = EmbeddingModel(
        id=provider.model_id,
        provider="fake",
        name="hashing",
        revision="1",
        dimension=8,
        normalized=True,
        max_input_tokens=provider.max_input_tokens,
        created_at=clock.now(),
    )
    with open_store(db, clock) as store:
        maintenance = IndexMaintenance(store, clock)
        service = EmbeddingService(provider, replacement, HeuristicTokenCounter(), clock)
        with store.unit_of_work().begin() as writer:
            writer.put_embedding_model(replace(replacement, status=ModelStatus.RETIRED))
        with pytest.raises(RagError) as incomplete, store.unit_of_work().begin() as writer:
            writer.switch_active_embedding_model(replacement.id)
        assert incomplete.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH
        still_active = store.reader().get_index_metadata()
        assert still_active is not None
        assert still_active.active_embedding_model_id == metadata.active_embedding_model_id
        assert maintenance.switch_model(replacement, service)
        assert not maintenance.switch_model(replacement, service)
    switched = FlatVectorIndex(replacement.id, 8)
    assert switched.refresh(db)
    assert switched.search(np.ones(8, dtype=np.float32), 1)
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
    with open_store(db, clock) as store:
        maintenance = IndexMaintenance(store, clock)
        unused = TextHash.of_text("deliberately unreferenced cache entry")
        with store.unit_of_work().begin() as writer:
            writer.put_many(
                replacement.id,
                [StoredEmbedding(unused, np.eye(1, 8, dtype=np.float32)[0], 1.0, clock.now())],
            )
        assert unused in store.reader().get_many(replacement.id, [unused])
        collected = maintenance.collect(keep_versions=1)
        assert collected.versions_purged == 1
        assert collected.embeddings_removed >= 1
        assert unused not in store.reader().get_many(replacement.id, [unused])
        assert first_embedding_key not in store.reader().get_many(
            metadata.active_embedding_model_id, [first_embedding_key]
        )
        assert maintenance.collect(keep_versions=1).versions_purged == 0
        assert maintenance.rebuild_lexical(force=True)
        assert not maintenance.rebuild_lexical()
    rng = random.Random(4)  # noqa: S311 - reproducible operation sequence, not a security token
    visible = True
    lexical_document = LexicalDocument(
        chunk.id, chunk.text, " ".join(chunk.heading_path), chunk.language
    )
    for _ in range(1000):
        operation = rng.randrange(5)
        if operation == 0 and visible:
            lexical.remove([chunk.id])
            visible = False
        elif operation == 1 and not visible:
            lexical.upsert([lexical_document])
            visible = True
        elif operation == 2:
            with open_store(db, clock) as store:
                IndexMaintenance(store, clock).rebuild_lexical(force=True)
        elif operation == 3:
            with open_store(db, clock) as store:
                IndexMaintenance(store, clock).collect(keep_versions=1)
        else:
            assert bool(lexical.search("машина", 5)) is visible
        assert check_integrity(db).ok
    if not visible:
        lexical.upsert([lexical_document])
    assert main([*prefix, "index", "gc", "--json"], container=container) == 0
    assert main([*prefix, "index", "rebuild", "--lexical", "--json"], container=container) == 0
    (root / "a.md").unlink()
    assert (
        main(
            [*prefix, "ingest", "notes", "--accept-mass-delete", "--json"],
            container=container,
        )
        == 0
    )
    assert not lexical.search("машина", 5)


@pytest.mark.parametrize("with_headings", [False, True])
def test_one_paragraph_edit_reembeds_at_most_three_of_hundred_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_headings: bool
) -> None:
    embedded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        payload = json.loads(request.content)
        embedded.extend(payload["input"])
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
    path = root / "long.md"
    paragraphs = [
        f"Paragraph {number}. " + "This unique section contains documented facts. " * 17
        for number in range(98)
    ]

    def write_document() -> None:
        sections = [
            f"## Section {number}\n\n{paragraph}" if with_headings else paragraph
            for number, paragraph in enumerate(paragraphs)
        ]
        prefix = "# Long document\n\n" if with_headings else ""
        path.write_text(prefix + "\n\n".join(sections))

    write_document()
    db = tmp_path / "rag.db"
    config = tmp_path / "rag.toml"
    config.write_text(
        "[parsing]\nisolate = []\n[chunking]\ntarget_tokens = 100\nmax_tokens = 150\n"
        "[embedding]\nenabled = true\n"
        f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\n'
        'dimension = 4\nbase_url = "http://test"\n'
    )
    container = DefaultContainer()
    prefix = ["--config", str(config), "--db", str(db)]
    assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
    assert main([*prefix, "ingest", "notes"], container=container) == 0
    with open_store_read_only(db) as store:
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        document = store.reader().get_document_by_key(source.id, SourceKey("long.md"))
        assert document is not None and document.current_version_id is not None
        indexed = [
            c for c in store.reader().list_chunks(document.current_version_id) if c.chunk.indexed
        ]
        assert len(indexed) == 100
    embedded.clear()
    paragraphs[50] += " A changed sentence with a distinctive ending."
    write_document()
    assert main([*prefix, "ingest", "notes"], container=container) == 0
    assert 1 <= len(embedded) <= 3
    assert check_integrity(db).ok
