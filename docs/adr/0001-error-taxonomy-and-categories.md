# 0001. Error taxonomy, categories and `E_FILTER_INVALID`

Status: Accepted (2026-11-22)

## Context

Every failure is a `RagError(code, category, retryable, message, context)`. The
architecture defines the categories and a list of stable codes, but does not assign a
category to each code, and the retrieval design additionally needs a code for rejected
metadata filters. Categories drive CLI exit codes and API problem types, so they must be
explicit and stable.

## Decision

1. `E_FILTER_INVALID` is part of the taxonomy (category `user`).
2. Categories mean:

| Category | Meaning | Typical reaction |
|---|---|---|
| `user` | The caller supplied an invalid request | Fix the request |
| `config` | Configuration is invalid or incomplete | Fix configuration |
| `source` | A source is missing, moved or behaves unsafely | Operator action on the source |
| `parse` | A document cannot be parsed or has an unsupported format | Skip the document, report |
| `provider` | An external or model provider failed or returned unusable output | Retry if retryable, else report |
| `resource` | A budget, limit or time bound was exceeded | Reduce load or raise limits |
| `storage` | The database is locked or needs migration | Retry or migrate |
| `integrity` | Stored data or domain invariants are inconsistent | Stop; run integrity checks |
| `internal` | Reserved for unexpected faults; no code uses it yet | Bug report |

3. Code assignment (the `retryable` default is set per code and may be overridden per
   instance):

| Code | Category | Retryable | Rationale |
|---|---|---|---|
| `E_CONFIG_INVALID` | config | no | |
| `E_SOURCE_NOT_FOUND`, `E_SOURCE_LOCATOR_CHANGED`, `E_MASS_DELETE_GUARD` | source | no | needs operator decision |
| `E_PARSE_FAILED`, `E_UNSUPPORTED_FORMAT` | parse | no | deterministic for the same bytes |
| `E_EMBED_DIM_MISMATCH` | provider | no | the provider returned vectors of the wrong size |
| `E_EMBED_MODEL_MISMATCH` | integrity | no | the index was built with another model; a data-consistency problem |
| `E_PROVIDER_TIMEOUT`, `E_PROVIDER_UNAVAILABLE`, `E_RATE_LIMITED` | provider | yes | transient |
| `E_CONTEXT_TRUNCATED`, `E_DEGENERATE_ANSWER`, `E_CITATION_INVALID` | provider | no | unusable model output |
| `E_CONTEXT_BUDGET` | resource | no | evidence does not fit the budget |
| `E_RESOURCE_GATE` | resource | no | |
| `E_TIMEOUT` | resource | yes | a stage time budget was exceeded |
| `E_STORE_LOCKED` | storage | yes | another writer holds the lock |
| `E_MIGRATION_REQUIRED` | storage | no | |
| `E_INTEGRITY` | integrity | no | also raised for violated domain invariants |
| `E_FILTER_INVALID` | user | no | |
| `E_QUERY_INVALID` | user | no | added in Phase 5: blank, unusable or over-long query text |

4. Codes are never renamed or reused; a snapshot test fails on any change of code,
   category or default retryability.

## Consequences

- Transport layers can map categories to exit codes and HTTP statuses without a per-code
  table.
- Changing a category later is a breaking change and needs a new record.
- Malformed identifiers and constructor-level invariant violations use `E_INTEGRITY`;
  request-level validation (for example API input) will need its own `user` codes when
  those layers are built.
