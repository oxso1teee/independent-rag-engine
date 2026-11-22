-- Schema v1: sources, documents, versions, sync state, jobs, index state.
-- Chunks, embeddings and the full-text index arrive with schema v2.

CREATE TABLE sources (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    kind                TEXT NOT NULL CHECK (kind IN ('fs', 'inline')),
    locator             TEXT,
    locator_fingerprint TEXT,
    policy_json         TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE documents (
    id                 TEXT PRIMARY KEY,
    source_id          TEXT NOT NULL REFERENCES sources (id),
    source_key         TEXT NOT NULL,
    current_version_id TEXT,
    status             TEXT NOT NULL CHECK (status IN ('pending', 'active', 'missing', 'deleted', 'failed')),
    duplicate_of       TEXT REFERENCES documents (id) ON DELETE SET NULL,
    missing_since      TEXT,
    title              TEXT,
    tags_json          TEXT NOT NULL DEFAULT '[]',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE (source_id, source_key)
);
CREATE INDEX documents_by_status ON documents (source_id, status);

CREATE TABLE versions (
    id                  TEXT PRIMARY KEY,
    document_id         TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    seq                 INTEGER NOT NULL CHECK (seq >= 1),
    content_hash        TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL CHECK (size_bytes >= 0),
    mtime_ns            INTEGER NOT NULL CHECK (mtime_ns >= 0),
    mime                TEXT NOT NULL,
    detected_by         TEXT NOT NULL,
    parser_id           TEXT NOT NULL,
    parser_version      TEXT NOT NULL,
    normalizer_version  TEXT NOT NULL,
    chunker_fingerprint TEXT NOT NULL,
    text_purged         INTEGER NOT NULL DEFAULT 0 CHECK (text_purged IN (0, 1)),
    parse_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at          TEXT NOT NULL,
    UNIQUE (document_id, seq)
);

CREATE TABLE sync_state (
    source_id         TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    source_key        TEXT NOT NULL,
    last_size_bytes   INTEGER,
    last_mtime_ns     INTEGER,
    last_content_hash TEXT,
    last_seen_at      TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('ok', 'changed', 'failed', 'missing')),
    attempts          INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error_code   TEXT,
    next_retry_at     TEXT,
    PRIMARY KEY (source_id, source_key)
) WITHOUT ROWID;

CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('scan', 'ingest_doc', 'reindex', 'gc')),
    source_id   TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    document_id TEXT REFERENCES documents (id) ON DELETE CASCADE,
    stage       TEXT NOT NULL CHECK (stage IN ('discovered', 'parsed', 'chunked', 'embedded', 'committed')),
    status      TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'done', 'failed', 'dead')),
    lease_until TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    error_code  TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    available_at TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX jobs_claim ON jobs (status, available_at, id);
-- At most one active job per (kind, source, document): enqueueing is idempotent.
CREATE UNIQUE INDEX jobs_one_active ON jobs (kind, source_id, COALESCE(document_id, ''))
    WHERE status IN ('queued', 'leased', 'failed');

CREATE TABLE index_state (
    id                          INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version              INTEGER NOT NULL CHECK (schema_version >= 1),
    index_generation            INTEGER NOT NULL DEFAULT 0 CHECK (index_generation >= 0),
    active_embedding_model_id   TEXT,
    lexical_analyzer_id         TEXT NOT NULL,
    chunker_default_fingerprint TEXT NOT NULL,
    vector_store_kind           TEXT NOT NULL,
    calibration_json            TEXT NOT NULL DEFAULT '{}',
    created_at                  TEXT NOT NULL,
    last_gc_at                  TEXT
);
