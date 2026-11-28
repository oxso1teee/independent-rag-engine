# Operations

How to run the engine, look after its data and watch it. Everything here is a local, single-machine
service; there is no cluster, no hosted component and no telemetry.

## Running

| How | Command |
|---|---|
| Command line | `rag ingest`, `rag search`, `rag ask`: no daemon, the process ends with the command |
| HTTP API | `RAG_API_TOKEN=... rag serve` (see [API.md](API.md)); binds `127.0.0.1` by default |
| Docker | `docker build -t rag-engine .` and `docker run -p 127.0.0.1:8765:8765 -e RAG_API_TOKEN=... -v rag-data:/data rag-engine` |

The image is multi-stage, pinned to a base-image digest and to hashed dependencies from `uv.lock`,
runs as the unprivileged user `rag` (uid 10001), keeps its data in `/data`, has a health check, and is
about 80 MB. It contains the API and no embedding runtime: it reaches Ollama over HTTP, or build with
`--build-arg EXTRAS="--extra api --extra onnx"` for in-process ONNX embeddings (about 250 MB more).
It refuses to start without `RAG_API_TOKEN` because it listens on `0.0.0.0` inside the container;
publish the port on `127.0.0.1` unless something in front of it terminates TLS. `scripts/docker_smoke.sh`
checks all of this and runs in CI.

## Memory

The target machine is small and shared. Measured or file-size figures: the ONNX embedder 1.2 GiB,
the cross-encoder (opt-in) 2.4 GiB, a 4B chat model about 3.5 GiB (not measured). Rules that keep
them from meeting: Ollama's `keep_alive` is `0`; a configuration that enables the reranker with a
resident chat model is refused; `rag ask` hands the in-process models back before it calls a local
model. A server keeps its models loaded, so size the machine for the sum. See
[ARCHITECTURE.md](ARCHITECTURE.md) (resource budget).

## Backup and restore

Everything the engine stores (documents' versions, chunks, vectors, the full-text index, jobs,
calibration) is one SQLite file. `rag backup` takes a consistent copy while the server runs, and
checks it before it reports success.

```bash
rag backup backups/rag-$(date +%F).db      # refuses to overwrite; prints the size
rag verify backups/rag-2026-11-28.db       # integrity check of a backup; exit 9 when damaged
```

Restore drill (do it before you need it):

```bash
rag verify backups/rag-2026-11-28.db                 # 1. the backup is sound
# stop `rag serve` and anything else using the database
rag restore backups/rag-2026-11-28.db --replace      # 2. the old database is kept as rag.db.before-restore
rag doctor --integrity                               # 3. clean, or do not go on
rag status                                           # 4. sources and documents are where you left them
```

`restore` verifies the backup first, copies it under a temporary name and moves it in only after the
copy verifies; a damaged backup is never restored and the current database is never overwritten
without `--replace`. The test suite runs this drill (`tests/e2e/test_admin_cli.py`).
Sources are directories on disk: a backup holds the index, not your documents; back those up as usual.

## Logs

JSON lines on standard error. `[logging] file = "logs/rag.log"` writes them to a file rotated by
size (`rotate_mb`, `backups`; the file is readable by its owner only). Document text, questions,
answers, prompts and secrets are not logged; events carry the `request_id` and error codes.

## Metrics and health

`GET /v1/health` (liveness, no token), `GET /v1/ready` (200 when a store, a published index and an
embedding model exist, else 503 with the failing checks) and `GET /v1/stats`. With `[api] metrics =
true`, `GET /v1/metrics` serves Prometheus text: request counts by method, route template and status
class, request time, answers by status, uptime. Nothing in a label identifies a document or a
question. There is no push, no trace export and no third-party service.

## Soak

`eval/scripts/soak.py` starts a real server, sends one search per minute for an hour and reads its
resident memory from `/proc`. Pass criteria: growth under 10 %, fewer than 2 % errors, no death.
It soaks search (the real embedding model); no language model has run on this machine. The one-hour
run of 2026-11-28 passed (60 requests, no error, resident memory 1162 down to 1101 MiB); see
[TESTING.md](TESTING.md). The script writes its report only at the end; to watch a run, read the
server's resident memory from `/proc/<pid>/status`.

## Upgrading

The schema migrates forward only, in one transaction per step, and a store that needs a migration
refuses to run until it is applied (`E_MIGRATION_REQUIRED`); `rag backup` first. Changing the
embedding model is `rag index switch-model` (staged, then activated atomically); changing the
chunker re-processes only the documents it affects.
