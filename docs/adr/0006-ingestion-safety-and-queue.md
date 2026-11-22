# 0006. Ingestion safety rules, retry policy, exit codes and MIME detection

Status: Accepted (2026-11-22).

## Context

Ingestion must never lose documents because of a wrong root or a transient failure, must keep
working when a file is malformed, and must report failures in a form scripts can rely on. This
record states the rules and the default values that implement that.

## Decisions

### Mass-delete guard
A scan that makes known documents vanish is refused with `E_MASS_DELETE_GUARD`, and nothing is
changed, when **any** of these holds (`ingestion/lifecycle.py`):

| Rule | Meaning |
|---|---|
| `threshold` | more than `max(10 files, 20 %)` of the known documents vanished |
| `empty_scan` | the scan found no document at all |
| `all_known_gone` | every known document vanished at once |

The last two catch a wrong or swapped root, which the threshold alone misses on small corpora.
Deliberately deleting the last documents therefore needs confirmation: `--accept-mass-delete`.
Documents moved or renamed with provably identical content are not counted as vanished.
Tombstones are reversible during the grace period; a missing root is `E_SOURCE_NOT_FOUND`,
never a deletion; an unreadable directory is never a deletion.

### Retry policy of the job queue
Exponential backoff with jitter. The values are settings (`[ingest]`):

| Parameter | Default | Note |
|---|---|---|
| attempts | 5 | bounded, then `dead` |
| base delay / factor / cap | 30 s / 2 / 1 h | |
| jitter | up to 20 % shorter, injectable randomness | spreads retries; never exceeds the cap |
| lease | 300 s | longer than a provider call, extended by heartbeats |

Non-retryable errors go `dead` immediately, so a poison file never blocks the queue.

### Other values
Grace period 30 days and the file limit (50 MiB = 52 428 800 bytes) are per-source policy.

### Exit codes follow the error category
`0` ok, `1` internal/unexpected, `2` user/usage, `3` config, `4` source, `5` parse, `6` provider,
`7` resource, `8` storage, `9` integrity. The precise code (`E_MASS_DELETE_GUARD`,
`E_STORE_LOCKED`, ...) is in the text and in `--json` output.

### MIME detection
Signatures (`filetype`), text decodability and encoding (`charset-normalizer`), and the
extension only as a tie-breaker between plain text, Markdown and HTML. Accepted text encodings:
UTF-8, UTF-16/32 with a byte-order mark, cp1251, koi8-r, cp866, ISO 8859-5, Mac Cyrillic and the
western single-byte family. The encoding is recorded with the job. A **short** non-UTF-8 file
can be assigned a similar single-byte encoding (detection needs a few dozen characters), so
parsers decode defensively.

### Logging
Ingestion emits `stage.start/done/failed` for `ingest.scan`, `ingest.apply`, `ingest.gc`, plus
`ingest.done`/`ingest.planned` with counts, all under one `request_id`. Events carry numbers and
codes only: no paths, names or text (a test checks this).

## Consequences
Retry and lease parameters can be tuned without code changes; the safety rules are
intentionally not configurable. The rule that tripped is visible in the `rule` context of
`E_MASS_DELETE_GUARD`.
