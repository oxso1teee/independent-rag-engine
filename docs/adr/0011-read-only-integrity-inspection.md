# 0011. Read-only integrity inspection

Status: Accepted (2026-11-26).

## Context

SQLite transactions protect publication, but corruption outside the publisher, a bad model
registration, or an interrupted operation must be observable without modifying the store.

## Decision

`rag doctor --integrity` opens a read-only, query-only SQLite snapshot. It checks database and
foreign-key integrity, version ownership, deterministic chunk identity and text provenance,
published FTS membership, active-model embedding coverage, model identity, vector dimensions,
finite norms and the generation required by a visible index. It returns structured errors and a
nonzero exit code. The command never repairs data. Recovery is a separate ingest, rebuild, or
model-switch operation under the normal single-writer lock.

The FTS5 external-content table cannot be counted through a plain `SELECT` because that reads
the content table. Doctor compares published chunks with the FTS docsize shadow row set and
reports stale entries. The check runs on one SQLite snapshot so writers cannot produce a false
cross-table mismatch during inspection.

## Consequences

Corrupt copies, random operation sequences and injected failures can be checked with the same
diagnostic used by operators. A successful check establishes the invariants visible to this
snapshot; it does not prove that files outside SQLite or provider model weights are intact.
