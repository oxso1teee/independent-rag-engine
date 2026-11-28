# 0014. Reranking stays off by default

Status: Accepted (2026-11-27), confirmed by a measurement on 2026-11-28 (below). It is revisited
when reviewed labels and a harder evaluation set exist.

## Context

A cross-encoder can reorder a pool of candidates more precisely than the retrieval channels do, at
the cost of memory and latency on every query. The requirements allow reranking as an optional
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

## Measurement (2026-11-28)

With the owner's approval the pinned `BAAI/bge-reranker-v2-m3-int8` snapshot (revision `6f5ff65`,
570,727,094 bytes, fingerprint `01a3f06c…`) was provisioned and measured with
`eval/scripts/rerank_bakeoff.py`: the real ingest and search of the mini corpus, 36 answerable
questions, a pool of 30, four threads, on a machine shared with a desktop session. The labels are
unreviewed, so the result is exploratory.

| | dense | hybrid |
|---|---:|---:|
| nDCG@10 without / with the reranker | 1.000 / 0.979 | 1.000 / 0.969 |
| paired difference nDCG@10 (95 % interval) | −0.021 [−0.051, 0.000] | −0.031 [−0.072, 0.000] |
| paired difference hit@1 | −0.056 [−0.139, 0.000] | −0.083 [−0.194, 0.000] |
| rerank stage p50 / p95 | 3.2 s / 4.1 s | 3.1 s / 4.1 s |
| whole search p95 without / with | 21 ms / 4.2 s | 24 ms / 4.1 s |

- **No gain, a small loss.** Without the reranker the set is already saturated (1.000), so there
  was no room to improve; the reranker moved the gold passage down on 2 to 3 of 36 questions.
  No interval excludes zero.
- **Latency is two to three times the limit.** The rerank stage takes about 3 s at p50 and 4.1 s at
  p95 for the questions of this tiny corpus, against a limit of 1.5 s; longer chunks would be slower.
- **Memory.** The process peaked at 2.5 GiB (embedding model, reranker and the evaluation), against
  1.5-3 GiB usually available; loading the reranker alone took 4 s and 1.3 GiB.
- **Quality of the model itself is fine:** on the opt-in test with the real weights it puts the
  relevant passage first in English and in Russian and refuses a pair over the window.

## What was not measured

The measurement above could not show a gain even in principle: dense and hybrid already return
the gold document first for every question (the set is saturated, see ADR 0013). Whether the
reranker helps on a harder, reviewed set (near-duplicates, contradicting versions, longer
passages) is still open. Its latency of about 3-4 s per query for a pool of 30 is not.

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
- A smaller pool (10 to 15), a faster model or a quantised English-only one would be the way to
  meet the latency limit; each needs its own measurement under the same rule.
- Evaluating whether the reranker helps at all is blocked on reviewed, harder evaluation questions.

## Amendment (2026-11-28)

On the harder, multilingual set the cross-encoder raised nDCG@10 by +0.16 (dense) and +0.37 (hybrid)
on dev with intervals above zero ([ADR 0018](0018-hard-set-gate-and-reranker.md)); the mini set had
been saturated. The labels are unreviewed, the test split was not read for it, p95 is still about
4 s against the 1.5 s limit and the peak is about 2.4 GiB. The decision is therefore unchanged:
off by default, recommended as an opt-in. It is revisited when reviewed labels and a test read exist.
