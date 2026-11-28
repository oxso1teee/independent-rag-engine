"""Read-only integrity checks for published SQLite indexes."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_engine.domain import ChunkId, DocumentVersionId, EmbeddingModelId, RagError
from rag_engine.domain.models import SUFFICIENCY_ALGORITHM
from rag_engine.storage.sqlite.connection import connect


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    errors: tuple[str, ...]
    published_chunks: int
    embeddings: int
    index_generation: int
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def _calibration_warnings(state: sqlite3.Row, active: str | None, documents: int) -> list[str]:
    """Missing, foreign, outdated, resized or below-target calibration, in words."""
    try:
        stored = json.loads(state["calibration_json"] or "{}").get("sufficiency")
    except (ValueError, AttributeError):
        return ["the stored calibration cannot be read"]
    if active is None:
        return []
    if not isinstance(stored, dict):
        return ["the sufficiency gate is not calibrated, so weak evidence is not refused before "
                "generation; run `rag calibrate run QUESTIONS`"]  # fmt: skip
    out: list[str] = []
    if stored.get("model_id") != active:
        out.append("the sufficiency calibration was made for another embedding model; recalibrate")
    if stored.get("algorithm") != SUFFICIENCY_ALGORITHM:
        out.append("the sufficiency calibration was made for an older decision rule; recalibrate")
    details = stored.get("details")
    made = details.get("documents") if isinstance(details, dict) else None
    if (
        isinstance(made, int)
        and not isinstance(made, bool)
        and made > 0
        and abs(documents - made) / made > 0.5
    ):
        out.append(
            f"the index has changed a lot since the calibration ({made} documents then, "
            f"{documents} now); recalibrate"
        )
    if stored.get("below_target") is True:
        out.append("the sufficiency calibration's estimate was below its targets")
    return out


def _active_documents(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'active'").fetchone()
    return int(row[0])


def check_integrity(database: Path) -> IntegrityReport:
    """Check one consistent snapshot without changing or repairing the database."""
    conn = connect(database, busy_timeout_ms=5000, read_only=True)
    errors: list[str] = []
    try:
        conn.execute("BEGIN")
        for row in conn.execute("PRAGMA integrity_check"):
            if row[0] != "ok":
                errors.append(f"SQLite integrity_check: {row[0]}")
        if not conn.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]:
            errors.append("SQLite FTS5 resource gate failed")
        for row in conn.execute("PRAGMA foreign_key_check"):
            errors.append(f"foreign key violation: {row[0]} row {row[1]}")

        state = conn.execute(
            "SELECT index_generation, active_embedding_model_id, calibration_json"
            " FROM index_state WHERE id = 1"
        ).fetchone()
        if state is None:
            errors.append("index_state is missing")
            return IntegrityReport(tuple(errors), 0, 0, 0)
        generation = int(state["index_generation"])
        if generation < 0:
            errors.append("negative index_generation")
        active = state["active_embedding_model_id"]
        models = list(conn.execute("SELECT * FROM embedding_models"))
        model_by_id = {str(row["id"]): row for row in models}
        for model in models:
            try:
                expected = EmbeddingModelId.build(
                    provider=str(model["provider"]),
                    name=str(model["name"]),
                    revision=str(model["revision"]),
                    dimension=int(model["dimension"]),
                    normalized=bool(model["normalized"]),
                    query_prefix=str(model["query_prefix"]),
                    document_prefix=str(model["document_prefix"]),
                )
                if str(expected) != model["id"]:
                    errors.append(f"model identity mismatch: {model['id']}")
            except (ValueError, TypeError, RagError):
                errors.append(f"invalid model identity: {model['id']}")
        active_rows = [row for row in models if row["status"] == "active"]
        if len(active_rows) != (1 if active is not None else 0):
            errors.append("active model count differs from index_state")
        if active is not None and (
            active not in model_by_id or model_by_id[active]["status"] != "active"
        ):
            errors.append("index_state points to a missing or retired model")

        for row in conn.execute(
            "SELECT d.id FROM documents AS d LEFT JOIN document_versions AS v"
            " ON v.id = d.current_version_id AND v.document_id = d.id"
            " WHERE d.current_version_id IS NOT NULL AND (v.id IS NULL OR v.text_purged = 1)"
        ):
            errors.append(f"invalid current_version_id: {row['id']}")
        for row in conn.execute(
            "SELECT c.id FROM chunks AS c JOIN document_versions AS v"
            " ON v.id = c.document_version_id"
            " JOIN documents AS d ON d.id = v.document_id"
            " WHERE c.indexed = 1 AND (d.current_version_id != v.id OR d.status != 'active'"
            " OR v.text_purged = 1)"
        ):
            errors.append(f"unpublished chunk is indexed: {row['id']}")

        for version in conn.execute(
            "SELECT id, normalized_text, chunker_fingerprint, text_purged"
            " FROM document_versions WHERE EXISTS"
            " (SELECT 1 FROM chunks WHERE document_version_id = document_versions.id)"
        ):
            if version["text_purged"] or version["normalized_text"] is None:
                errors.append(f"purged version still has chunks: {version['id']}")
                continue
            try:
                normalized = zlib.decompress(version["normalized_text"]).decode("utf-8")
            except (zlib.error, UnicodeError, TypeError):
                errors.append(f"invalid normalized text: {version['id']}")
                continue
            for chunk in conn.execute(
                "SELECT id, ordinal, text, text_hash, char_start, char_end FROM chunks"
                " WHERE document_version_id = ?",
                (version["id"],),
            ):
                key = chunk["id"]
                if normalized[chunk["char_start"] : chunk["char_end"]] != chunk["text"]:
                    errors.append(f"chunk text offset mismatch: {key}")
                if hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["text_hash"]:
                    errors.append(f"chunk text hash mismatch: {key}")
                try:
                    expected_id = ChunkId.derive(
                        DocumentVersionId.parse(version["id"]),
                        version["chunker_fingerprint"],
                        chunk["ordinal"],
                    )
                    if str(expected_id) != key:
                        errors.append(f"chunk identity mismatch: {key}")
                except (ValueError, RagError):
                    errors.append(f"invalid chunk identity: {key}")

        published = int(conn.execute("SELECT count(*) FROM chunks WHERE indexed = 1").fetchone()[0])
        if published and (active is None or generation == 0):
            errors.append("published chunks require an active model and positive generation")
        for row in conn.execute(
            "SELECT c.id FROM chunks AS c LEFT JOIN embeddings AS e"
            " ON e.embedding_input_hash = c.embedding_input_hash AND e.model_id = ?"
            " WHERE c.indexed = 1 AND e.model_id IS NULL",
            (active,),
        ):
            errors.append(f"published chunk lacks active embedding: {row['id']}")

        # External-content FTS tables expose content rows even if no term was indexed.
        # The docsize shadow table records the actual FTS row set.
        for row in conn.execute(
            "SELECT c.chunk_rowid FROM chunks AS c LEFT JOIN chunks_fts_docsize AS f"
            " ON f.id = c.chunk_rowid WHERE c.indexed = 1 AND f.id IS NULL"
        ):
            errors.append(f"published chunk missing from FTS: {row[0]}")
        for row in conn.execute(
            "SELECT f.id FROM chunks_fts_docsize AS f LEFT JOIN chunks AS c"
            " ON c.chunk_rowid = f.id WHERE c.chunk_rowid IS NULL OR c.indexed = 0"
        ):
            errors.append(f"stale FTS row: {row[0]}")

        count = 0
        for row in conn.execute(
            "SELECT e.embedding_input_hash, e.model_id, e.vector, e.norm, e.dtype, m.dimension"
            " FROM embeddings AS e LEFT JOIN embedding_models AS m ON m.id = e.model_id"
        ):
            count += 1
            key = f"{row['model_id']}:{row['embedding_input_hash']}"
            dimension = row["dimension"]
            blob = row["vector"]
            model = model_by_id.get(str(row["model_id"]))
            if model is None:
                errors.append(f"orphan embedding model: {key}")
            if dimension is None or row["dtype"] != "f16le" or len(blob) != dimension * 2:
                errors.append(f"invalid vector shape or dtype: {key}")
                continue
            vector = np.frombuffer(blob, dtype="<f2").astype(np.float32)
            norm = float(np.linalg.norm(vector))
            stored_norm = float(row["norm"])
            if (
                not math.isfinite(norm)
                or norm <= 0
                or not math.isfinite(stored_norm)
                or abs(norm - stored_norm) >= 1e-3
            ):
                errors.append(f"invalid vector norm: {key}")
            if model is not None and model["normalized"] and abs(norm - 1.0) >= 1e-3:
                errors.append(f"unnormalized vector: {key}")
        return IntegrityReport(
            tuple(errors),
            published,
            count,
            generation,
            tuple(_calibration_warnings(state, active, _active_documents(conn))),
        )
    except sqlite3.Error as exc:
        errors.append(f"SQLite integrity query failed: {exc}")
        return IntegrityReport(tuple(errors), 0, 0, 0)
    finally:
        conn.close()
