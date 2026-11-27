"""Memory and time of loading a published SQLite index into the flat vector index.

``benchmark_dense.py`` measures search over synthetic vectors held in memory. This script
measures the product path instead: a real schema-v3 database with ``count`` published chunks
and FP16 embeddings, loaded by ``FlatVectorIndex.refresh`` in a fresh process, which is what a
search worker or an index rebuild does. Run: ``uv run python eval/scripts/benchmark_refresh.py``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag_engine.domain import ChunkId, EmbeddingModelId
from rag_engine.storage.sqlite import open_store
from rag_engine.storage.vector_flat import FlatVectorIndex
from rag_engine.system import SystemClock, system_id_factory

DIMENSION = 1024
CHUNKS_PER_DOCUMENT = 100
NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
MODEL = EmbeddingModelId.build(
    provider="bench",
    name="synthetic",
    revision="1",
    dimension=DIMENSION,
    normalized=True,
    query_prefix="",
    document_prefix="",
)


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS unavailable")


def fill(connection: sqlite3.Connection, table: str, **given: Any) -> None:
    """Insert one row, giving every other NOT NULL column without a default a plain value."""
    values: dict[str, Any] = dict(given)
    for _, name, kind, notnull, default, _pk in connection.execute(f"PRAGMA table_info({table})"):
        if name in values or default is not None or not notnull:
            continue
        values[name] = 0 if kind.upper() in {"INTEGER", "REAL"} else ("" if kind != "BLOB" else b"")
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", list(values.values()))  # noqa: S608


def build(database: Path, count: int) -> None:
    open_store(database, SystemClock()).close()
    ids = system_id_factory()
    rng = np.random.default_rng(7)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        source = ids.source_id()
        fill(
            connection,
            "sources",
            id=str(source),
            name="bench",
            kind="fs",
            policy_json="{}",
            created_at=NOW,
            updated_at=NOW,
        )
        fill(
            connection,
            "embedding_models",
            id=str(MODEL),
            provider="bench",
            name="synthetic",
            revision="1",
            dimension=DIMENSION,
            normalized=1,
            query_prefix="",
            document_prefix="",
            max_input_tokens=512,
            status="active",
            created_at=NOW,
        )
        connection.execute(
            "UPDATE index_state SET active_embedding_model_id = ?, index_generation = 1",
            (str(MODEL),),
        )
        written = 0
        for number in range(count // CHUNKS_PER_DOCUMENT):
            document, version = ids.document_id(), ids.document_version_id()
            fill(
                connection,
                "document_versions",
                id=str(version),
                document_id=str(document),
                seq=1,
                created_at=NOW,
            )
            fill(
                connection,
                "documents",
                id=str(document),
                source_id=str(source),
                source_key=f"d{number}.md",
                current_version_id=str(version),
                status="active",
                created_at=NOW,
                updated_at=NOW,
            )
            vectors = rng.standard_normal((CHUNKS_PER_DOCUMENT, DIMENSION)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            for ordinal, vector in enumerate(vectors):
                key = hashlib.sha256(f"{written}".encode()).hexdigest()
                connection.execute(
                    "INSERT INTO embeddings VALUES (?, ?, ?, 'f16le', 1.0, ?)",
                    (key, str(MODEL), vector.astype("<f2").tobytes(), NOW),
                )
                fill(
                    connection,
                    "chunks",
                    id=str(ChunkId.derive(version, "bench", ordinal)),
                    document_version_id=str(version),
                    ordinal=ordinal,
                    level=0,
                    kind="paragraph",
                    text="x",
                    text_hash=key,
                    embedding_input_hash=key,
                    token_count=1,
                    char_end=1,
                    language="en",
                    indexed=1,
                    is_duplicate=0,
                    oversize=0,
                )
                written += 1
        connection.commit()


def child(database: Path) -> None:
    before = rss_mib()
    index = FlatVectorIndex(MODEL, DIMENSION)
    started = time.perf_counter()
    index.refresh(database)
    load_s = time.perf_counter() - started
    after = rss_mib()
    query = np.random.default_rng(1).standard_normal(DIMENSION).astype(np.float32)
    times = []
    for _ in range(40):
        t = time.perf_counter()
        index.search(query, 10)
        times.append((time.perf_counter() - t) * 1000)
    print(
        json.dumps(
            {
                "vectors": len(index),
                "load_seconds": load_s,
                "rss_before_load_mib": before,
                "rss_after_load_mib": after,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "search_p50_ms": statistics.median(times),
            }
        )
    )


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]))
        return
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    with tempfile.TemporaryDirectory(prefix="rag-refresh-") as folder:
        database = Path(folder) / "bench.db"
        started = time.perf_counter()
        build(database, count)
        built = time.perf_counter() - started
        result = subprocess.run(  # noqa: S603
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
                    **json.loads(result.stdout),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
