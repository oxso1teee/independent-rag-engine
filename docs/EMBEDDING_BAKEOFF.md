# Phase 4 embedding bake-off: local diagnostic run

Status: **incomplete**. The two installed Ollama models were measured on the same small RU/EN
probe. The ONNX candidate and a reviewed public retrieval set are still required for D-03. The
probe has only five questions per language; its scores are descriptive, not acceptance evidence
for a default model.

## Reproduction and environment

```bash
uv run python eval/scripts/bakeoff_embedders.py --model qwen3-embedding:0.6b
uv run python eval/scripts/bakeoff_embedders.py --model qwen3-embedding:4b
OPENBLAS_NUM_THREADS=4 uv run python eval/scripts/benchmark_dense.py
```

Measured on 2026-11-26 with four CPU cores, Python 3.12.3, 7.8 GiB RAM and 4 GiB swap.
Before loading models, 16 GiB disk was free. The development environment `.venv` occupied
368,353,031 bytes; this includes tests, lint and type-checking tools. ONNX Runtime and FastEmbed
were installed, but no provisioned embedding ONNX weights were present. No model was downloaded.
The probe dataset contains 16 short passages, five RU and five EN queries, and two relevant
passages per query. Its SHA-256 is
`f0cd3d18c02fc41134e591a1b175e0ead76fa523b87e251312da50531e1c78ce`.

## Measured model results

| Installed model | Digest prefix | Model bytes | Dimensions | RU Recall@5 / nDCG@10 | EN Recall@5 / nDCG@10 | Query embedding p50 / p95 | Passage batch throughput | Runner peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3-embedding:0.6b` | `ac6da0dfba84` | 639,150,858 | 1024 | 1.000 / 0.984 | 1.000 / 0.984 | 286 / 453 ms | 6.41 passages/s | 1,159 MiB |
| `qwen3-embedding:4b` | `df5bd2e3c74c` | 2,496,704,041 | 2560 | 1.000 / 1.000 | 1.000 / 1.000 | 1,079 / 1,623 ms | 0.59 passages/s | 3,069 MiB |

Throughput is for one 16-passage request. The 0.6B value is a warm run; a cold run took
6.15 s (2.60 passages/s). The 4B run included loading and took 27.29 s. Query times were
measured after the passage batch. Runner RSS was sampled from the local `llama-server`
process; Python peak RSS was 46 MiB. These timings include HTTP and provider inference but
exclude full parsing, chunking and transactional publication.

Both upstream Qwen3 embedding model cards declare Apache-2.0:
[0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B). Local Ollama blobs were identified by
their installed digests; the upstream license declaration does not independently verify the
provenance of those local blobs. No model weights are distributed with this repository.

## Exact dense search

Synthetic unit-normalized float32 vectors were inserted into the existing fp16 flat index.
Forty top-10 searches were measured at each size with `OPENBLAS_NUM_THREADS=4`; query
embedding time is excluded.

| Vectors × dimensions | p50 | p95 | Search process RSS | Construction peak RSS |
|---|---:|---:|---:|---:|
| 10,000 × 1024 | 20.55 ms | 25.42 ms | 70.7 MiB | 129.1 MiB |
| 100,000 × 1024 | 203.04 ms | 216.10 ms | 267.2 MiB | 947.4 MiB |

The 10k and 100k search latencies meet NFR-03 (100 ms and 400 ms p95 respectively), and
search RSS meets NFR-01 (300 MiB and 600 MiB). Synthetic index construction peaks above the
100k search RSS budget because it holds the float32 source array and fp16 index concurrently;
this is a separate ingestion-memory concern.

## Decision still needed

The probe is too small and saturated to establish a quality difference. The 0.6B model has
lower latency, memory and disk use on this machine; the 4B model has a small nDCG advantage on
these ten queries. A default cannot be justified from this dataset alone. Complete the ONNX
comparison on the same corpus, then evaluate all candidates on a reviewed public RU/EN set
with enough queries for paired uncertainty intervals before accepting D-03.
