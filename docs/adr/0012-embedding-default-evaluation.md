# 0012. Embedding default requires a resource-gated bake-off

Status: Proposed (2026-11-26). Decision D-03 is open.

## Context

The default embedding model determines RU/EN retrieval quality, disk, RAM, latency and license
obligations. The project supports pinned Ollama models and provisioned local ONNX files. A
hashing test double cannot establish retrieval quality.

## Evidence so far

The reproducible local probe and 10k/100k exact-search measurements are recorded in
[`EMBEDDING_BAKEOFF.md`](../EMBEDDING_BAKEOFF.md). Two installed Ollama models were measured on
the same 16-passage, 10-question RU/EN probe. Both reached Recall@5 of 1.0. The 4B model
improved nDCG@10 from 0.984 to 1.0, while requiring about 2.6 times the runner RSS and about
four times the query embedding latency. The corpus is too small and saturated to establish a
general quality difference. No local embedding ONNX weights were available for comparison.

## Proposed decision rule

Do not change the default configuration until an ONNX candidate and the installed Ollama models
are measured on the same reviewed public RU/EN retrieval set. Compare Recall@5 and nDCG@10
with paired uncertainty intervals and record ingestion throughput, p95 embedding latency, peak
RSS, model/environment size, license and exact dense-search p50/p95. Keep model identity pinned
to a digest or a local-file fingerprint. The operator can already configure any supported model
explicitly.
