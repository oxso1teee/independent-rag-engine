# 0006. Ingestion safety rules, retry policy, exit codes and MIME detection

Status: Accepted (2026-11-22).

## Context

The compliance review compared the ingestion rules of Phase 2 with `ARCHITECTURE` §A, §7 and
`TECHNOLOGY_DECISIONS`. Values the specification does not give were chosen during
implementation; this record states which are specified, which are assumptions, and why.

## Decisions

### Mass-delete guard
A scan that makes known documents vanish is refused with `E_MASS_DELETE_GUARD`, and nothing is
changed, when **any** of these holds (`ingestion/lifecycle.py`):

| Rule | Source |
|---|---|
| more than `max(10 files, 20 %)` of the known documents vanished | specification (§A) |
| the scan found no document at all (`empty_scan`) | owner, safety |
| every known document vanished at once (`all_known_gone`) | owner, safety |

The two extra rules catch a wrong or swapped root, which the threshold alone misses on small
corpora. The cost: deliberately deleting the last documents needs confirmation. The flag is
`--accept-mass-delete` (the specification's name). Documents moved or renamed with provably
the same content are not counted as vanished. Tombstones are reversible during the grace
period; a missing root is `E_SOURCE_NOT_FOUND`, never a deletion; an unreadable directory is
never a deletion.

### Retry policy of the job queue
Backoff with jitter, as specified. Concrete numbers are **assumptions** (the specification gives
none) and live in `[ingest]` settings:

| Parameter | Default | Basis |
|---|---|---|
| attempts | 5 | assumption; bounded, then `dead` |
| base delay / factor / cap | 30 s / 2 / 1 h | assumption |
| jitter | up to 20 % shorter, injectable randomness | specification requires jitter |
| lease | 300 s | assumption; above the 180 s provider timeout, extended by heartbeats |

Non-retryable errors go `dead` immediately, so a poison file never blocks the queue.

### Other values
Grace period 30 days and the 50 MB file limit are specified (the limit is implemented as
50 MiB = 52 428 800 bytes, 5 % above 50 MB); both are per-source policy.

### Exit codes follow the error category (`ARCHITECTURE` §7)
`0` ok · `1` internal/unexpected · `2` user/usage · `3` config · `4` source · `5` parse ·
`6` provider · `7` resource · `8` storage · `9` integrity. The precise code
(`E_MASS_DELETE_GUARD`, `E_STORE_LOCKED`, ...) is in the text and in `--json` output.

### MIME detection
As specified: signatures (`filetype`), text decodability and encoding (`charset-normalizer`),
extension only as a tie-breaker. Accepted text encodings: UTF-8, UTF-16/32 with a byte-order
mark, cp1251, koi8-r, cp866, ISO 8859-5, Mac Cyrillic, and the western single-byte family. The
encoding is recorded with the job. Limitation: a **short** non-UTF-8 file can be assigned a
similar single-byte encoding (detection needs a few dozen characters); parsers must decode
defensively.

### Logging
Ingestion emits `stage.start/done/failed` for `ingest.scan`, `ingest.apply`, `ingest.gc`, plus
`ingest.done`/`ingest.planned` with counts, all under one `request_id`. Events carry numbers and
codes only: no paths, names or text (a test checks this).

## Consequences
Parameters can be tuned without code changes; the safety rules are intentionally not
configurable. Both extra guard rules are visible in `E_MASS_DELETE_GUARD`'s `rule` context.
