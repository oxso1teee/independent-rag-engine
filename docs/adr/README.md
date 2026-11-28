# Architecture decision records

Format: context, decision, consequences. A changed decision is a new record that
supersedes the old one.

| # | Title | Status |
|---|---|---|
| [0001](0001-error-taxonomy-and-categories.md) | Error taxonomy, categories and `E_FILTER_INVALID` | Accepted |
| [0002](0002-single-resource-profile.md) | A single `low` resource profile | Accepted |
| [0003](0003-async-provider-ports.md) | Asynchronous provider ports, synchronous storage ports | Accepted |
| [0004](0004-store-boundaries-for-phase-2.md) | Store boundaries and the domain additions needed for Phase 2 | Accepted (amended by 0005, 0007) |
| [0005](0005-documents-versions-and-schema-v2.md) | Documents, versions and schema v2 | Accepted |
| [0006](0006-ingestion-safety-and-queue.md) | Ingestion safety rules, retry policy, exit codes and MIME detection | Accepted |
| [0007](0007-layout-and-composition-root.md) | Layout, composition root and boundaries | Accepted |
| [0008](0008-parsing-and-chunking-contracts.md) | Parsing and chunking contracts | Accepted |
| [0009](0009-transactional-index-publication.md) | Transactional index publication and schema v3 | Accepted |
| [0010](0010-embedding-input-identity.md) | Embedding input and model identity | Accepted |
| [0011](0011-read-only-integrity-inspection.md) | Read-only integrity inspection | Accepted |
| [0012](0012-embedding-default-evaluation.md) | Default embedding model: multilingual-e5-small through ONNX (D-03) | Accepted |
| [0013](0013-default-retrieval-mode.md) | Default retrieval mode stays dense; hybrid is an option | Accepted (provisional) |
| [0014](0014-reranker-off-by-default.md) | Reranking stays off by default | Accepted (provisional) |
| [0015](0015-generation-contracts.md) | Generation contracts: refusal sentinel, provisional streaming, lexical overlap | Accepted |
| [0016](0016-query-pipeline-contract.md) | Query pipeline: four statuses, one time budget, verified streaming | Accepted |
| [0017](0017-http-api.md) | The HTTP API: a thin transport with a narrow write surface | Accepted |
