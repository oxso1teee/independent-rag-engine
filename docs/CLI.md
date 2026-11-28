# Command line

`rag [--config FILE] [--db FILE] COMMAND`. Every command that prints data accepts `--json` and then
prints exactly one JSON document (errors included, as `{"error": {...}}`). The text of errors goes
to standard error.

| Command | Purpose |
|---|---|
| `rag init [--dir DIR] [--force]` | create a starter `rag.toml` and the data directory |
| `rag config show` / `rag config check` | print the effective settings / validate them (exit 3 when invalid) |
| `rag source add NAME DIR`, `rag source list`, `rag source rebase NAME DIR` | register and maintain document sources |
| `rag ingest [NAME] [--dry-run] [--accept-mass-delete]` | scan sources, queue changes and index them |
| `rag status` | sources, document states, the job queue |
| `rag doctor --integrity` | read-only integrity inspection, with warnings (for example a stale gate calibration) |
| `rag chunks DOC` | show how one document is parsed and chunked; nothing is stored |
| `rag search QUERY` | search the published index |
| `rag context QUERY` | the context a question would get: evidence, token budget, what was left out |
| `rag ask QUESTION` | answer with sources, or refuse (`--stream`, `--live`, `--sources`, `--show-unverified`) |
| `rag index gc`, `rag index rebuild`, `rag index switch-model` | maintain the published indexes |
| `rag eval validate DATASET`, `rag eval run`, `rag eval report RUN`, `rag eval compare A B`, `rag eval gate RUN BASELINE` | measure retrieval, compare runs pairwise, fail on a regression (see [EVALUATION.md](EVALUATION.md)) |
| `rag calibrate run QUESTIONS`, `rag calibrate show`, `rag calibrate clear` | fit, inspect or remove the sufficiency gate's calibration on this index (see [CALIBRATION.md](CALIBRATION.md)); `run` takes `--holdout FILE`, `--max-false-refusal`, `--folds`, `--dry-run`, `--require-target` (exit 12 when the estimate misses the targets) |
| `rag backup DEST` | a verified, consistent copy of the database; refuses to overwrite |
| `rag verify BACKUP` | check a backup without opening it for writing (exit 9 when damaged) |
| `rag restore BACKUP [--replace]` | restore a verified backup; the current database is kept aside as `.before-restore` |
| `rag serve [--host H] [--port P]` | run the HTTP API (see [API.md](API.md)) |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success (for `ask`: an answer) |
| 1 | unexpected failure |
| 2 | usage or user error |
| 3 | configuration (`E_CONFIG_INVALID`) |
| 4 | source or data not found |
| 5 | parse error |
| 6 | provider (model server) error |
| 7 | resource limit (memory gate, context budget, timeout) |
| 8 | storage (locked, migration required) |
| 9 | integrity |
| 10 | `ask`: insufficient evidence (a valid outcome, not a failure) |
| 11 | `ask`: unverified answer (the citations did not hold up) |
| 12 | `eval gate`: a metric is worse than its baseline |

Codes 3 to 9 follow the error category of the engine's stable error codes; the exact code is in the
text and in the JSON document. A script can therefore tell an answer (0) from a refusal (10) from a
failure (3 to 9).
