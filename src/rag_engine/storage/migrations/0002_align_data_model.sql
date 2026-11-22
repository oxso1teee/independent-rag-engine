-- migrate: foreign_keys=off
-- Schema v2: brings schema v1 in line with the data model.
--  * a document without a version is `active` with a NULL current_version_id (the separate
--    `pending` status is gone);
--  * `versions` becomes `document_versions`, gains `normalized_text` and the
--    UNIQUE (document_id, content_hash) rule.
-- `documents` is rebuilt because SQLite cannot change a CHECK constraint in place; the
-- runner keeps foreign keys off for the rebuild and verifies them before committing, so
-- versions and jobs that reference documents are never cascaded away.

CREATE TABLE documents_new (
    id                 TEXT PRIMARY KEY,
    source_id          TEXT NOT NULL REFERENCES sources (id),
    source_key         TEXT NOT NULL,
    current_version_id TEXT,
    status             TEXT NOT NULL CHECK (status IN ('active', 'missing', 'deleted', 'failed')),
    duplicate_of       TEXT REFERENCES documents (id) ON DELETE SET NULL,
    missing_since      TEXT,
    title              TEXT,
    tags_json          TEXT NOT NULL DEFAULT '[]',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE (source_id, source_key)
);

INSERT INTO documents_new (
    id, source_id, source_key, current_version_id, status, duplicate_of, missing_since,
    title, tags_json, created_at, updated_at
)
SELECT
    id, source_id, source_key, current_version_id,
    CASE status WHEN 'pending' THEN 'active' ELSE status END,
    duplicate_of, missing_since, title, tags_json, created_at, updated_at
FROM documents;

DROP TABLE documents;
ALTER TABLE documents_new RENAME TO documents;
CREATE INDEX documents_by_status ON documents (source_id, status);

ALTER TABLE versions RENAME TO document_versions;
ALTER TABLE document_versions ADD COLUMN normalized_text BLOB;
CREATE UNIQUE INDEX document_versions_by_content ON document_versions (document_id, content_hash);
