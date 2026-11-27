"""Metadata filters become parameterised SQL over published chunks, never raw SQL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_engine.domain import (
    ChunkId,
    ChunkKind,
    DocumentId,
    DocumentStatus,
    ErrorCode,
    IdFactory,
    Language,
    RagError,
    RetrievalFilters,
    Source,
    TextHash,
)
from rag_engine.ports import ChunkCatalog, ChunkFilter, StoredChunk
from rag_engine.storage.filters import SqliteChunkCatalog, SqliteChunkFilter
from rag_engine.storage.sqlite import SqliteStore
from tests.factories import make_chunk, make_document, make_ids, make_source, make_version


@dataclass
class Corpus:
    database: Path
    chunks: dict[str, ChunkId]
    documents: dict[str, DocumentId]


def ns(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)


def add_document(
    store: SqliteStore,
    ids: IdFactory,
    source: Source,
    key: str,
    *,
    tags: tuple[str, ...] = (),
    mtime_ns: int = 1,
    status: DocumentStatus = DocumentStatus.ACTIVE,
    chunks: list[tuple[str, ChunkKind, Language, str]],
) -> tuple[DocumentId, dict[str, ChunkId]]:
    """``chunks``: (name, kind, language, state) with state indexed, duplicate or unindexed."""
    document = make_document(
        document_id=ids.document_id(), source_id=source.id, version_id=None, key=key
    )
    version = replace(
        make_version(version_id=ids.document_version_id(), document_id=document.id),
        mtime_ns=mtime_ns,
    )
    stored: list[StoredChunk] = []
    named: dict[str, ChunkId] = {}
    for ordinal, (name, kind, language, state) in enumerate(chunks):
        base = make_chunk(name, version_id=version.id, ordinal=ordinal, start=ordinal * 100)
        chunk = replace(
            base,
            kind=kind,
            language=language,
            indexed=state == "indexed",
            is_duplicate=state == "duplicate",
        )
        key_hash = TextHash.of_text(chunk.text) if chunk.indexed else None
        stored.append(StoredChunk(chunk, key_hash, chunk.text if chunk.indexed else "", ""))
        named[f"{key}:{name}"] = chunk.id
    with store.unit_of_work().begin() as writer:
        writer.put_document(document)
        writer.put_version(version)
        writer.put_chunks(version.id, stored)
        writer.put_document(
            replace(document, current_version_id=version.id, tags=tags, status=status)
        )
    return document.id, named


@pytest.fixture
def corpus(tmp_path: Path, store: SqliteStore, ids: IdFactory) -> Corpus:
    notes, wiki = make_source(ids, name="notes"), make_source(ids, name="wiki", locator="/w")
    with store.unit_of_work().begin() as writer:
        writer.put_source(notes)
        writer.put_source(wiki)
    para, table, code = ChunkKind.PARAGRAPH, ChunkKind.TABLE, ChunkKind.CODE
    en, ru = Language.EN, Language.RU
    documents: dict[str, DocumentId] = {}
    chunks: dict[str, ChunkId] = {}
    rows = [
        (
            notes,
            "projects/alpha.md",
            ("work", "draft"),
            ns(2024),
            [("a1", para, en, "indexed"), ("a2", table, en, "indexed")],
        ),
        (notes, "projects/beta.txt", ("work",), ns(2025), [("b1", para, ru, "indexed")]),
        (
            notes,
            "journal/2026.md",
            ("private",),
            ns(2026),
            [
                ("j1", code, ru, "indexed"),
                ("j2", para, ru, "duplicate"),
                ("j3", para, ru, "unindexed"),
            ],
        ),
        (notes, "readme.md", (), ns(2023), [("r1", para, en, "indexed")]),
        (wiki, "guide.md", ("work", "public"), ns(2025), [("g1", para, en, "indexed")]),
        (wiki, "100%_done/x.md", (), ns(2025), [("p1", para, en, "indexed")]),
        (wiki, "gone.md", ("work",), ns(2025), [("d1", para, en, "indexed")]),
    ]
    for source, key, tags, mtime, items in rows:
        status = DocumentStatus.DELETED if key == "gone.md" else DocumentStatus.ACTIVE
        identifier, named = add_document(
            store,
            ids,
            source,
            key,
            tags=tags,
            mtime_ns=mtime,
            status=status,
            chunks=items,
        )
        documents[key] = identifier
        chunks.update(named)
    return Corpus(tmp_path / "rag.db", chunks, documents)


def resolve(corpus: Corpus, **filters: object) -> set[str]:
    found = SqliteChunkFilter(corpus.database).resolve(RetrievalFilters(**filters))  # type: ignore[arg-type]
    assert found is not None
    names = {value: key for key, value in corpus.chunks.items()}
    return {names[chunk_id].split(":")[1] for chunk_id in found}


def test_no_filter_means_no_restriction_and_the_adapters_satisfy_the_ports(corpus: Corpus) -> None:
    resolver = SqliteChunkFilter(corpus.database)
    assert isinstance(resolver, ChunkFilter)
    assert isinstance(SqliteChunkCatalog(corpus.database), ChunkCatalog)
    assert resolver.resolve(RetrievalFilters()) is None


def test_only_searchable_chunks_are_ever_returned(corpus: Corpus) -> None:
    everything = resolve(corpus, sources=("notes", "wiki"))
    assert everything == {
        "a1",
        "a2",
        "b1",
        "j1",
        "r1",
        "g1",
        "p1",
    }  # no duplicate, unindexed, deleted


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"sources": ("wiki",)}, {"g1", "p1"}),
        ({"path_prefix": "projects/"}, {"a1", "a2", "b1"}),
        ({"path_prefix": "Projects/"}, set()),  # case-sensitive
        ({"path_prefix": "100%_"}, {"p1"}),  # % and _ are literal
        ({"path_prefix": "1%"}, set()),
        ({"path_glob": "**/*.md"}, {"a1", "a2", "j1", "r1", "g1", "p1"}),
        ({"path_glob": "*.txt"}, {"b1"}),
        ({"path_glob": "projects/*"}, {"a1", "a2", "b1"}),
        ({"tags_any": ("private", "public")}, {"j1", "g1"}),
        ({"tags_all": ("work", "draft")}, {"a1", "a2"}),
        ({"tags_all": ("work",), "sources": ("wiki",)}, {"g1"}),
        ({"language": Language.RU}, {"b1", "j1"}),
        ({"kinds": (ChunkKind.TABLE, ChunkKind.CODE)}, {"a2", "j1"}),
        ({"modified_after": datetime(2025, 6, 1, tzinfo=UTC)}, {"j1"}),
        ({"modified_before": datetime(2024, 6, 1, tzinfo=UTC)}, {"a1", "a2", "r1"}),
        (
            {
                "modified_after": datetime(2023, 6, 1, tzinfo=UTC),
                "modified_before": datetime(2025, 6, 1, tzinfo=UTC),
            },
            {"a1", "a2", "b1", "g1", "p1"},
        ),
    ],
)
def test_each_filter_selects_what_it_says(
    corpus: Corpus, filters: dict[str, object], expected: set[str]
) -> None:
    assert resolve(corpus, **filters) == expected


def test_filters_combine_with_and(corpus: Corpus) -> None:
    assert resolve(
        corpus, sources=("notes",), tags_any=("work",), kinds=(ChunkKind.PARAGRAPH,)
    ) == {
        "a1",
        "b1",
    }
    assert resolve(corpus, sources=("wiki",), language=Language.RU) == set()


def test_document_ids_include_and_exclude(corpus: Corpus) -> None:
    alpha, beta = corpus.documents["projects/alpha.md"], corpus.documents["projects/beta.txt"]
    assert resolve(corpus, doc_ids=(alpha, beta)) == {"a1", "a2", "b1"}
    assert resolve(
        corpus, doc_ids=(alpha, beta), exclude_doc_ids=(corpus.documents["readme.md"],)
    ) == {
        "a1",
        "a2",
        "b1",
    }
    with pytest.raises(RagError):  # the domain refuses a document that is both in and out
        RetrievalFilters(doc_ids=(alpha,), exclude_doc_ids=(alpha,))
    assert resolve(corpus, exclude_doc_ids=(alpha,), sources=("notes",)) == {"b1", "j1", "r1"}


@pytest.mark.parametrize(
    "hostile",
    ["x' OR '1'='1", "notes'; DROP TABLE chunks; --", '"; SELECT 1; --', "%", "\\", "\x00"],
)
def test_hostile_values_are_data_not_sql(corpus: Corpus, hostile: str) -> None:
    for filters in (
        {"sources": (hostile,)},
        {"path_prefix": hostile},
        {"tags_any": (hostile,)},
        {"tags_all": (hostile,)},
    ):
        assert resolve(corpus, **filters) == set()
    still_there = resolve(corpus, sources=("notes",))
    assert "a1" in still_there  # the table survived


def test_a_malformed_glob_or_an_oversized_list_is_a_filter_error(corpus: Corpus) -> None:
    resolver = SqliteChunkFilter(corpus.database)
    for filters in (
        RetrievalFilters(path_glob="[unterminated"),
        RetrievalFilters(path_glob="../escape"),
        RetrievalFilters(sources=tuple(f"s{i}" for i in range(1001))),
    ):
        with pytest.raises(RagError) as info:
            resolver.resolve(filters)
        assert info.value.code is ErrorCode.E_FILTER_INVALID


def test_the_catalog_describes_published_chunks(corpus: Corpus) -> None:
    catalog = SqliteChunkCatalog(corpus.database)
    wanted = [corpus.chunks["projects/alpha.md:a1"], corpus.chunks["readme.md:r1"]]
    unknown = ChunkId.derive(make_ids().document_version_id(), "unknown", 0)
    described = catalog.describe([*wanted, unknown])
    assert set(described) == set(wanted)
    first = described[wanted[0]]
    assert first.ref.document_id == corpus.documents["projects/alpha.md"]
    assert first.text_hash == TextHash.of_text("a1")
    assert catalog.describe([]) == {}
