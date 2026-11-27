"""`doctor --integrity` finds each kind of corruption it claims to find, and never writes.

A valid indexed database is built once; every case works on its own copy and injects one fault
with the CHECK constraints disabled (a damaged file does not honour them).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from rag_engine.cli.main import main
from rag_engine.container import DefaultContainer
from rag_engine.domain import EmbeddingModelId, ErrorCode, RagError
from rag_engine.storage.integrity import check_integrity
from rag_engine.storage.lexical import SqliteLexicalIndex
from rag_engine.storage.vector_flat import FlatVectorIndex

DIGEST = "b" * 64


@pytest.fixture(scope="module")
def valid_database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": DIGEST}]})
        payload = json.loads(request.content)
        return httpx.Response(
            200, json={"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in payload["input"]]}
        )

    real_client = httpx.AsyncClient
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            base_url=kwargs["base_url"], transport=httpx.MockTransport(handler)
        ),
    )
    try:
        base = tmp_path_factory.mktemp("integrity")
        root = base / "notes"
        root.mkdir()
        (root / "a.md").write_text("# Heading\n\nA paragraph about local indexing and storage.\n")
        db = base / "rag.db"
        config = base / "rag.toml"
        config.write_text(
            "[parsing]\nisolate = []\n[embedding]\nenabled = true\n"
            f'provider = "ollama"\nmodel = "test"\nrevision = "{DIGEST}"\n'
            'dimension = 4\nbase_url = "http://test"\n'
        )
        container = DefaultContainer()
        prefix = ["--config", str(config), "--db", str(db)]
        assert main([*prefix, "source", "add", "notes", str(root)], container=container) == 0
        assert main([*prefix, "ingest", "notes", "--json"], container=container) == 0
        assert check_integrity(db).ok
        yield db
    finally:
        patcher.undo()


FAULTS = {
    "missing index_state": ("DELETE FROM index_state", "index_state is missing"),
    "negative generation": (
        "UPDATE index_state SET index_generation = -1",
        "negative index_generation",
    ),
    "model identity": (
        "UPDATE embedding_models SET name = 'renamed'",
        "model identity mismatch",
    ),
    "unparsable model identity": (
        "UPDATE embedding_models SET dimension = 'wide'",
        "invalid model identity",
    ),
    "retired active model": (
        "UPDATE embedding_models SET status = 'retired'",
        "active model count differs",
    ),
    "state without model": (
        "UPDATE index_state SET active_embedding_model_id = 'absent'",
        "missing or retired model",
    ),
    "purged text under chunks": (
        "UPDATE document_versions SET text_purged = 1",
        "purged version still has chunks",
    ),
    "garbled normalized text": (
        "UPDATE document_versions SET normalized_text = x'00ff00'",
        "invalid normalized text",
    ),
    "chunk identity": (
        "UPDATE chunks SET ordinal = ordinal + 100",
        "chunk identity mismatch",
    ),
    "deleted document keeps indexed chunks": (
        "UPDATE documents SET status = 'deleted'",
        "unpublished chunk is indexed",
    ),
    "missing embedding": ("DELETE FROM embeddings", "published chunk lacks active embedding"),
    "stale FTS row": ("UPDATE chunks SET indexed = 0", "stale FTS row"),
    "vector dtype": ("UPDATE embeddings SET dtype = 'f32le'", "invalid vector shape or dtype"),
    "short vector": (
        "UPDATE embeddings SET vector = substr(vector, 1, 4)",
        "invalid vector shape or dtype",
    ),
    "stored norm differs": ("UPDATE embeddings SET norm = 3.0", "invalid vector norm"),
    "unnormalized vector": (
        "UPDATE embeddings SET vector = x'0040000000000000', norm = 2.0",  # float16 [2, 0, 0, 0],
        "unnormalized vector",
    ),
}


@pytest.mark.parametrize(("statement", "expected"), FAULTS.values(), ids=FAULTS.keys())
def test_each_fault_is_reported_without_touching_the_file(
    valid_database: Path, tmp_path: Path, statement: str, expected: str
) -> None:
    damaged = tmp_path / "damaged.db"
    with (
        closing(sqlite3.connect(valid_database)) as source,
        closing(sqlite3.connect(damaged)) as target,
    ):
        source.backup(target)
    with closing(sqlite3.connect(damaged)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement)
        connection.commit()
    before = hashlib.sha256(damaged.read_bytes()).digest()
    report = check_integrity(damaged)
    assert hashlib.sha256(damaged.read_bytes()).digest() == before
    assert not report.ok
    assert any(expected in error for error in report.errors), report.errors


def test_a_file_that_is_not_a_database_is_reported_as_an_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 100)
    report = check_integrity(broken)
    assert not report.ok and report.errors


def copy_of(valid_database: Path, tmp_path: Path) -> Path:
    damaged = tmp_path / "snapshot.db"
    with (
        closing(sqlite3.connect(valid_database)) as source,
        closing(sqlite3.connect(damaged)) as target,
    ):
        source.backup(target)
    return damaged


STORED_MODEL = EmbeddingModelId.build(
    provider="ollama",
    name="test",
    revision=DIGEST,
    dimension=4,
    normalized=True,
    query_prefix="",
    document_prefix="",
)

VECTOR_FAULTS = {
    "wrong dtype": ("UPDATE embeddings SET dtype = 'f32le'", "invalid shape"),
    "short blob": ("UPDATE embeddings SET vector = substr(vector, 1, 4)", "invalid shape"),
    "NaN component": (
        "UPDATE embeddings SET vector = x'007e000000000000'",  # float16 NaN, 0, 0, 0
        "non-finite",
    ),
}


def test_a_healthy_snapshot_loads_once_per_generation(valid_database: Path) -> None:
    index = FlatVectorIndex(STORED_MODEL, 4)
    assert index.refresh(valid_database) and not index.refresh(valid_database)
    assert len(index) >= 1


@pytest.mark.parametrize(("statement", "message"), VECTOR_FAULTS.values(), ids=VECTOR_FAULTS.keys())
def test_a_damaged_stored_vector_is_never_loaded(
    valid_database: Path, tmp_path: Path, statement: str, message: str
) -> None:
    damaged = copy_of(valid_database, tmp_path)
    with closing(sqlite3.connect(damaged)) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement)
        connection.commit()
    index = FlatVectorIndex(STORED_MODEL, 4)
    with pytest.raises(RagError) as info:
        index.refresh(damaged)
    assert info.value.code is ErrorCode.E_INTEGRITY and message in info.value.message
    assert len(index) == 0  # the failed refresh left the (empty) previous matrix alone


def test_refresh_refuses_another_active_model(valid_database: Path) -> None:
    other = EmbeddingModelId.build(
        provider="ollama",
        name="other",
        revision=DIGEST,
        dimension=4,
        normalized=True,
        query_prefix="",
        document_prefix="",
    )
    with pytest.raises(RagError) as info:
        FlatVectorIndex(other, 4).refresh(valid_database)
    assert info.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH


def test_refresh_refuses_changed_model_metadata(valid_database: Path, tmp_path: Path) -> None:
    damaged = copy_of(valid_database, tmp_path)
    with closing(sqlite3.connect(damaged)) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE embedding_models SET dimension = 8")
        connection.commit()
    with pytest.raises(RagError) as info:
        FlatVectorIndex(STORED_MODEL, 4).refresh(damaged)
    assert info.value.code is ErrorCode.E_EMBED_MODEL_MISMATCH


def test_lexical_index_counts_and_finds_published_chunks(valid_database: Path) -> None:
    index = SqliteLexicalIndex(valid_database)
    assert len(index) >= 1
    assert index.search("paragraph indexing", 5)
    assert index.search("paragraph", 0) == []


def test_lexical_search_refuses_an_index_built_by_another_analyzer(
    valid_database: Path, tmp_path: Path
) -> None:
    damaged = copy_of(valid_database, tmp_path)
    with closing(sqlite3.connect(damaged)) as connection:
        connection.execute("UPDATE index_state SET lexical_analyzer_id = 'other/1'")
        connection.commit()
    with pytest.raises(RagError) as info:
        SqliteLexicalIndex(damaged).search("paragraph", 5)
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_lexical_search_reports_a_damaged_fts_table_as_an_integrity_error(
    valid_database: Path, tmp_path: Path
) -> None:
    damaged = copy_of(valid_database, tmp_path)
    with closing(sqlite3.connect(damaged)) as connection:
        connection.execute("DROP TABLE chunks_fts_data")
        connection.commit()
    with pytest.raises(RagError) as info:
        SqliteLexicalIndex(damaged).search("paragraph", 5)
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_doctor_warns_when_the_calibration_belongs_to_another_model(
    valid_database: Path, tmp_path: Path
) -> None:
    import json as jsonlib

    healthy = check_integrity(valid_database)
    assert healthy.ok and healthy.warnings == ()
    stale = copy_of(valid_database, tmp_path)
    with closing(sqlite3.connect(stale)) as connection:
        stored = jsonlib.dumps({"sufficiency": {"model_id": "another/model", "dense": 0.5}})
        connection.execute("UPDATE index_state SET calibration_json = ?", (stored,))
        connection.commit()
    report = check_integrity(stale)
    assert report.ok  # a stale calibration is a warning, not corruption
    assert any("recalibrate" in item for item in report.warnings)
