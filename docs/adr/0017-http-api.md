# 0017. The HTTP API: a thin transport with a narrow write surface

Status: Accepted (2026-11-28).

## Context

The specification asks for a REST API with OpenAPI, stable error codes and a safe default. The
engine's services are synchronous and its query services are heavy to open; the machine is small.

## Decisions

1. **FastAPI behind an extra.** The core and the command line work without it; the dev group
   includes it so the API is tested in CI.
2. **Same services, same code path.** The API calls `RagPipeline`, `RetrievalService` and the
   ingestion `Services`; presentation helpers shared with the CLI moved to `retrieval.presentation`
   and `pipeline.presentation`. Neither transport re-implements a rule.
3. **A narrow write surface.** Over HTTP documents can only be written into *managed* sources whose
   directory the engine owns. Registering an arbitrary directory, or reading and writing files by
   path, is left to the local command line; responses never include server paths. This keeps a
   leaked token from becoming a file-system primitive.
4. **Token by environment variable, safe by default.** The default variable name is `RAG_API_TOKEN`;
   when it is set the API requires it, when it is not the API is open on the loopback address only,
   and any other bind refuses to start without it. CORS is off. Health is open.
5. **Problem+json everywhere**, one status table, and API-only codes for API-only conditions
   (`E_BODY_TOO_LARGE`, `E_IDEMPOTENCY_CONFLICT`, ...).
6. **Idempotency in memory.** `Idempotency-Key` is remembered per process for a TTL; durable
   idempotency would need a schema change and was judged out of proportion for a personal service.
   Documented.
7. **Lazy, long-lived query services.** Opened on the first query and kept, because loading the
   embedding model and the vector matrix per request would cost seconds and over a GiB each time.
   Indexing is scheduled in the background after a 202.
8. **Snapshot as contract.** `docs/openapi.json` is committed and compared in the test suite.

## Consequences

An API client can search, ask, stream, upload and manage documents, and cannot touch the file
system outside managed sources. Not done: authentication beyond one shared token, per-client rate
limits, TLS (put a reverse proxy in front for a non-loopback bind), and persistent job-status
streaming. Real-model behaviour through the API is as unmeasured as through the command line.
