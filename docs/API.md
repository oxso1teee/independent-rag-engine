# HTTP API

`rag serve` runs the API (extra `api`: `pip install "independent-rag-engine[api]"`). It is a thin
transport over the same services as the command line: the same retrieval, the same query pipeline,
the same error codes. The machine-readable contract is [`openapi.json`](openapi.json) (also served
at `/v1/openapi.json`, behind the token); a test fails when the API changes without it.

## Starting the server

```bash
RAG_API_TOKEN="$(openssl rand -hex 24)" rag serve            # 127.0.0.1:8765
```

| Setting (`[api]`) | Default | Meaning |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `8765` | where to listen (`--host`, `--port` override) |
| `token_env` | `RAG_API_TOKEN` | the environment variable that holds the bearer token |
| `max_body_bytes` | 8 MiB | larger request bodies are refused with 413 |
| `max_question_chars` | 4000 | longer questions are refused with 400 |
| `idempotency_ttl_s` | 86400 | how long an `Idempotency-Key` is remembered |
| `cors_origins` | none | browser origins allowed by CORS; empty keeps CORS off |
| `metrics` | off | serve Prometheus-format metrics at `GET /v1/metrics` (needs the token like everything else) |

## Security

- **Loopback by default.** With the token variable unset the server is open on `127.0.0.1` only.
  Listening on any other address **refuses to start** without a token (`E_CONFIG_INVALID`), and a
  token shorter than 16 characters is refused everywhere.
- **Bearer token.** Every endpoint except `GET /v1/health` needs `Authorization: Bearer <token>`;
  comparison is constant-time and the token never appears in a response or a log.
- **No server paths.** Responses never contain filesystem paths. Sources created through the API are
  *managed*: the engine owns their directory (`<database directory>/inline/<name>`), and documents
  can be written only there. Filesystem sources registered with `rag source add` can be read and
  searched through the API but never written, so the API is not a way to read or write arbitrary files.
- **Limits.** Body size, question length and list page size are bounded; document keys are
  validated (no absolute paths, no `..`, supported extensions only) and written atomically.
- **CORS is off** unless origins are listed explicitly.

## Errors

Every error is `application/problem+json` (RFC 9457):

```json
{"type": "urn:rag:problem:e-source-not-found", "title": "Not Found", "status": 404,
 "detail": "no such document", "code": "E_SOURCE_NOT_FOUND", "category": "source",
 "retryable": false, "instance": "/v1/documents/...", "request_id": "..."}
```

`code` is the engine's stable error code; `retryable` tells a client whether to try again (a
`Retry-After` header accompanies a retryable 503). API-only codes: `E_UNAUTHORIZED`, `E_NOT_FOUND`,
`E_METHOD_NOT_ALLOWED`, `E_BODY_TOO_LARGE`, `E_IDEMPOTENCY_CONFLICT`, `E_INTERNAL`. Every response
carries `X-Request-ID`, the same id the logs use.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /v1/health` | liveness, no authentication |
| `GET /v1/ready` | 200 when the store exists, an index is published and an embedding model is configured; else 503 with the failing checks |
| `GET /v1/stats` | sources, document counts, jobs, index generation |
| `GET /v1/sources`, `POST /v1/sources`, `GET /v1/sources/{name}` | list, create a managed source, read one |
| `POST /v1/sources/{name}/ingest` | scan a source and queue what changed (202); indexing continues in the background |
| `GET /v1/documents` | list, `?source=&status=&limit=&cursor=` (keyset pagination, `next_cursor`) |
| `GET /v1/documents/{id}` | one document with its versions and chunk count |
| `POST /v1/documents` | add or replace a document in a managed source (202), `Idempotency-Key` supported |
| `DELETE /v1/documents/{id}` | remove a document of a managed source (202) |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | indexing jobs, cursor-paginated |
| `POST /v1/search` | retrieval only: ranked passages with provenance |
| `POST /v1/answers` | an answer with citations, or an honest refusal; JSON or server-sent events |

### Idempotency

`POST /v1/documents` accepts `Idempotency-Key` (1 to 128 characters). A retry with the same key and
body within the TTL replays the first response (header `Idempotent-Replay: true`) and writes nothing;
the same key with a different body is `409 E_IDEMPOTENCY_CONFLICT`. The memory is the server
process's: a restart forgets keys.

### Answers

```bash
curl -s localhost:8765/v1/answers -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question": "When do the backups run?", "include_sources": true}'
```

The result has one of four `status` values. HTTP 200 covers the first three; a failed stage is a
problem response (502, 503, 504 by code).

| `status` | Meaning | `text` | `citations` |
|---|---|---|---|
| `answered` | answered, every citation resolves to evidence of this request | yes | at least one |
| `insufficient_evidence` | nothing found, the evidence was too weak (the model was not called), or the model said the documents do not contain the answer | empty | none |
| `unverified` | the model's citations did not hold up; its text is withheld (`include_unverified_text` returns it as `unverified_text`, which must not be trusted) | empty | none |
| `error` | only inside a stream: a failed stage, with `error.code` and `error.stage` | empty | none |

### Streaming

`"stream": true` returns `text/event-stream`. Events, in order: `evidence`, `delta`\*, `retract`?,
`citations`, `final`. By default the text arrives once, as a single `delta` with
`"provisional": false`, **after** its citations were checked. With `"live": true` deltas stream as
the model writes them and are `"provisional": true`: if validation fails a `retract` event follows
and the draft must be discarded. Only the `final` event, which has the same shape as the JSON
answer, is authoritative. A stream always ends with `final`; failures are reported in it (status
`error`) rather than by closing the connection. Closing the connection cancels the request and
frees its slot.

### Concurrency

Management calls open the store per request. The query services (embedding model, vector index,
chat provider) are opened on the first query and kept; queries are served one at a time on the
`low` profile (`limits.llm_concurrency`), the others wait. A cancelled request does not affect the
others.
