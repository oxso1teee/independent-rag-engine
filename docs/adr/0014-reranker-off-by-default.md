# 0014. Reranking stays off by default

Status: Accepted (2026-11-27). Provisional: it is revisited when the measurement below can be made.

## Context

A cross-encoder can reorder a pool of candidates more precisely than the retrieval channels do, at
the cost of memory and latency on every query. The specification allows reranking as an optional
component and permits it to be enabled by default only after it has shown a gain on the evaluation
set (paired nDCG@10 interval above zero on dev, confirmed on test, with a bounded p95 and memory).
The target machine has 4 vCPU and often 1.5-3 GiB of free memory (ADR 0002), and the corpus is
Russian and English, so the model must be multilingual and its licence must allow use.

## What exists

- The `Reranker` port, a deterministic fake, a stage that bounds the pool and the time and can
  never fail a search, and `OnnxCrossEncoder` (extra `onnx`, offline, files pinned by fingerprint,
  pairs longer than the window refused). `[rerank]` in the configuration turns it on; `rag search
  --rerank` and `rag context` use it when it is configured.
- Of the cross-encoders FastEmbed can run, one is multilingual with a permissive licence:
  `BAAI/bge-reranker-v2-m3-int8` (Apache-2.0, 0.57 GB). The other multilingual one is licensed
  CC-BY-NC and is excluded.

## What was not measured

No cross-encoder weights are provisioned on the development machine and none was downloaded, so
there is no measured gain, latency or memory figure for any reranker. Downloading the 0.57 GB model
needs the owner's approval. Even with the weights, the current evaluation set could not show a
gain: dense and hybrid already return the gold document first for every question (the set is
saturated, see ADR 0013), so there is nothing left for a reranker to improve.

## Decision

1. `[rerank] enabled` is `false` by default. Reranking is an explicit opt-in; the `low` profile
   keeps it off.
2. Enabling it by default needs all of: a reviewed dev set of at least 30 answerable questions that
   is not saturated; a paired nDCG@10 interval above zero on dev and the same direction on test; p95
   rerank latency of at most 1.5 s for a pool of 30 on 4 vCPU; resident memory within the budget
   left after the embedding model.
3. The bake-off to run then: `BAAI/bge-reranker-v2-m3-int8` against no reranking, dense and hybrid
   as first stage, with the pipeline of `eval/scripts/retrieval_eval.py`.

## Consequences

- Operators who enable it get the safety of the fallback: a failing, slow or busy reranker keeps
  the fused order and is reported as `rerank_skipped:<reason>`.
- Evaluating the reranker is blocked on two things the owner provides: approval to download the
  model and reviewed, harder evaluation questions.
