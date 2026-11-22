# 0002. A single `low` resource profile

Status: Accepted (2026-11-22)

## Context

Resource profiles bound concurrency, batch sizes and memory for the target machine class.
Only `low` has measured, justified limits (`llm_concurrency=1`, `embed_concurrency=1`,
`embed_batch=16`, `ingest_workers=1`, `max_loaded_vectors_mb=512`, reranker off). A second
profile (`standard`) was introduced without measurements.

## Decision

Only `low` is supported and is the default. The `standard` profile is removed; a value
other than `low` is rejected with `E_CONFIG_INVALID`. Individual limits can still be
overridden explicitly (`[limits]`). New profiles are added only together with measured
limits, in a new record.

## Consequences

- No unmeasured defaults can leak into later phases.
- `ResourceProfile` stays an enum, so adding a profile is a small, explicit change.
- Users with more capable machines raise individual limits explicitly.
