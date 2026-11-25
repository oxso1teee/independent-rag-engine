-- Schema v3: persistent chunks, content-addressed embeddings and lexical index.
-- The migration runner owns the schema-version bookkeeping.

CREATE TABLE embedding_models (
    id               TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    name             TEXT NOT NULL,
    revision         TEXT NOT NULL,
    dimension        INTEGER NOT NULL CHECK (dimension > 0),
    normalized       INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    query_prefix     TEXT NOT NULL,
    document_prefix  TEXT NOT NULL,
    max_input_tokens INTEGER NOT NULL CHECK (max_input_tokens > 0),
    status           TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    created_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX embedding_models_one_active ON embedding_models (status)
    WHERE status = 'active';

CREATE TABLE embeddings (
    embedding_input_hash TEXT NOT NULL,
    model_id             TEXT NOT NULL REFERENCES embedding_models (id),
    vector               BLOB NOT NULL,
    dtype                TEXT NOT NULL CHECK (dtype = 'f16le'),
    norm                 REAL NOT NULL CHECK (norm > 0.0),
    created_at           TEXT NOT NULL,
    PRIMARY KEY (embedding_input_hash, model_id),
    CHECK (length(embedding_input_hash) = 64)
) WITHOUT ROWID;

CREATE TABLE chunks (
    chunk_rowid           INTEGER PRIMARY KEY,
    id                    TEXT NOT NULL UNIQUE,
    document_version_id   TEXT NOT NULL REFERENCES document_versions (id) ON DELETE CASCADE,
    ordinal               INTEGER NOT NULL CHECK (ordinal >= 0),
    level                 INTEGER NOT NULL CHECK (level IN (0, 1)),
    parent_id             TEXT,
    kind                  TEXT NOT NULL CHECK (kind IN
                          ('paragraph', 'list', 'table', 'code', 'quote', 'mixed')),
    text                  TEXT NOT NULL CHECK (length(text) > 0),
    text_hash             TEXT NOT NULL CHECK (length(text_hash) = 64),
    embedding_input_hash  TEXT CHECK (embedding_input_hash IS NULL OR
                                     length(embedding_input_hash) = 64),
    token_count           INTEGER NOT NULL CHECK (token_count >= 0),
    char_start            INTEGER NOT NULL CHECK (char_start >= 0),
    char_end              INTEGER NOT NULL CHECK (char_end > char_start),
    heading_path_json     TEXT NOT NULL DEFAULT '[]',
    page_start            INTEGER,
    page_end              INTEGER,
    language              TEXT NOT NULL CHECK (language IN ('ru', 'en', 'mixed', 'und')),
    indexed               INTEGER NOT NULL CHECK (indexed IN (0, 1)),
    is_duplicate          INTEGER NOT NULL CHECK (is_duplicate IN (0, 1)),
    oversize              INTEGER NOT NULL CHECK (oversize IN (0, 1)),
    body_an               TEXT NOT NULL DEFAULT '',
    headings_an           TEXT NOT NULL DEFAULT '',
    UNIQUE (document_version_id, ordinal),
    UNIQUE (document_version_id, id),
    FOREIGN KEY (document_version_id, parent_id)
        REFERENCES chunks (document_version_id, id) DEFERRABLE INITIALLY DEFERRED,
    CHECK (level = 0 OR (parent_id IS NULL AND indexed = 0)),
    CHECK (is_duplicate = 0 OR indexed = 0),
    CHECK (indexed = 0 OR embedding_input_hash IS NOT NULL),
    CHECK ((page_start IS NULL AND page_end IS NULL) OR
           (page_start >= 1 AND page_end >= page_start))
);
CREATE INDEX chunks_by_version ON chunks (document_version_id, ordinal);
CREATE INDEX chunks_by_text_hash ON chunks (text_hash);
CREATE INDEX chunks_by_embedding_input ON chunks (embedding_input_hash) WHERE indexed = 1;

-- External content gives FTS a stable rowid and allows integrity checks against chunks.
-- The publisher updates FTS explicitly in the same transaction as current_version_id.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    body_an, headings_an,
    content='chunks', content_rowid='chunk_rowid',
    tokenize='unicode61 remove_diacritics 2'
);
