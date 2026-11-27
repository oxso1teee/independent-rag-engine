# Phase 4 embedding bake-off (decision D-03)

Status: **complete except one measurement** — the quality of `qwen3-embedding:4b` could not be
measured on the development machine (see [Qwen3 4B](#qwen3-4b-not-measured)). The decision and its
rule are in [ADR 0012](adr/0012-embedding-default-evaluation.md).

Three candidates were compared on the same data:

| Model | Identity pinned by | License | Dimensions | Window | Size on disk | Runs as |
|---|---|---|---:|---|---:|---|
| `intfloat/multilingual-e5-small` | upstream revision `614241f622f5…`, ONNX file `onnx/model.onnx` | MIT | 384 | 512 tokens | 492,919,681 B (all files) | in-process ONNX Runtime (extra `onnx`) |
| `qwen3-embedding:0.6b` | Ollama digest `ac6da0dfba84…` | Apache-2.0 | 1024 | Ollama context, 4096 by default here | 639,150,858 B | Ollama server (separate process) |
| `qwen3-embedding:4b` | Ollama digest `df5bd2e3c74c…` | Apache-2.0 | 2560 | same | 2,496,704,041 B | Ollama server |

Licences were read from the model cards at the pinned revisions on 2026-11-27. The weights are
not part of this repository.

## Method and its limits

**Data.** NoMIRACL `dev.relevant`, English and Russian, dataset revision
`ecd08778d0426a5ca28ac99763b0c9ddc2c78e68` (Apache-2.0, files checked against pinned SHA-256
values). Queries that have both relevant and non-relevant judgments are sorted by id and 40
evenly spaced ones are taken per language: 80 queries, 827 judged passages (410 English,
417 Russian). A passage is `title + "\n" + text`. Every model is used as its card prescribes:
`query: ` / `passage: ` prefixes for E5; for Qwen3 the instruction
`Instruct: Given a web search question, retrieve relevant passages.\nQuery: ` on queries only.
Ollama is called with `truncate: false`, as the product adapter does.

**What is ranked.** Each query ranks only its own judged passages (mean 10.5 in English, 10.4 in
Russian). This is *judged-pool re-ranking*, not retrieval over the full corpus, and it has
consequences:

- Scores are **not comparable** with published MIRACL or NoMIRACL retrieval numbers.
- Recall@10 is saturated (0.98–1.00 for every model) and says nothing; it is shown only because
  the pool is about ten passages.
- Recall@1 is capped at `1 / (number of relevant passages)`; with several relevant passages per
  query its mean cannot approach 1. Compare it between models, not with 1.
- Every distractor is a passage that earlier retrieval systems surfaced and annotators judged,
  so distractors are hard, but a model that finds an *unjudged* relevant passage gets no credit.
  This is how the pool is built according to the dataset documentation; it was not re-derived here.
- 80 queries give wide intervals (about ±0.04–0.05 on nDCG@10), so small differences are
  indistinguishable.

**Statistics.** Per-query metrics (Recall@1/5/10, MRR@10, nDCG@10 with binary gains) are averaged;
intervals are 95 % percentile bootstrap over queries (10,000 resamples, seed 20261126). Differences
between models are paired: every model uses the same resampled queries.

**E5 window.** 15 of the 827 passages (2 English, 13 Russian) exceed 512 tokens. FastEmbed cut them
silently during this evaluation. The product adapter refuses such input and the default chunker
produces chunks of at most 384 tokens, so the product never embeds a truncated chunk; the
evaluation, however, gave E5 slightly less text than Qwen3 on those 15 passages.

**Machine and conditions.** 4 vCPU, 7.8 GiB RAM shared with a desktop session (1.3–4 GiB available
during the runs), swap full, Python 3.12.3, ONNX Runtime 1.32.0, FastEmbed 0.9.1, Ollama 0.30.10.
Timings come from `bakeoff_nomiracl.py timing`: an uncached, evenly spaced sample of 64 passages
(batches of 8) and 40 queries sent one at a time, after one warm-up call; the Ollama model is
unloaded first so the first query shows a cold start. Quality comes from cached vectors
(`bakeoff_nomiracl.py quality`), which makes interrupted runs resumable; quality runs carry no timing.

Reproduce (needs the data under `data/eval/nomiracl`, the E5 snapshot under `data/models`, and the
Ollama models):

```bash
uv run python eval/scripts/bakeoff_nomiracl.py quality e5-small
uv run python eval/scripts/bakeoff_nomiracl.py quality qwen3-embedding:0.6b
uv run python eval/scripts/bakeoff_nomiracl.py quality qwen3-embedding:4b   # needs about 3.5 GiB free
uv run python eval/scripts/bakeoff_nomiracl.py timing e5-small              # and each Qwen3 model
uv run python eval/scripts/bakeoff_report.py                                # tables below
```

## Ranking quality

Mean and 95 % interval; 40 queries per language, 80 pooled.

| Scope | Model | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| English | E5-small | 0.345 [0.231, 0.470] | 0.785 [0.690, 0.873] | 0.746 [0.646, 0.843] | 0.765 [0.696, 0.830] |
| English | Qwen3 0.6B | 0.400 [0.287, 0.522] | 0.804 [0.712, 0.887] | 0.819 [0.724, 0.910] | 0.805 [0.737, 0.870] |
| Russian | E5-small | 0.263 [0.194, 0.338] | 0.771 [0.688, 0.850] | 0.839 [0.750, 0.919] | 0.827 [0.770, 0.879] |
| Russian | Qwen3 0.6B | 0.225 [0.148, 0.311] | 0.779 [0.706, 0.849] | 0.755 [0.665, 0.843] | 0.803 [0.746, 0.859] |
| Pooled | E5-small | 0.304 [0.235, 0.376] | 0.778 [0.715, 0.839] | 0.792 [0.724, 0.856] | 0.796 [0.752, 0.838] |
| Pooled | Qwen3 0.6B | 0.313 [0.241, 0.389] | 0.792 [0.733, 0.846] | 0.787 [0.719, 0.851] | 0.804 [0.760, 0.847] |

Paired differences, E5-small minus Qwen3 0.6B (`*`: the interval excludes zero):

| Scope | nDCG@10 | Recall@5 |
|---|---:|---:|
| English | −0.040 [−0.081, −0.006] `*` | −0.019 [−0.100, +0.060] |
| Russian | +0.024 [−0.022, +0.068] | −0.008 [−0.091, +0.068] |
| Pooled | −0.008 [−0.039, +0.022] | −0.013 [−0.071, +0.043] |

Reading: on this sample the two models are indistinguishable overall. Qwen3 0.6B is better in English
by about 0.04 nDCG@10 (the only interval that excludes zero); E5-small is nominally better in
Russian, within noise. E5-small prefers a passage in the query's language over a better one in
another language (a Russian question about France scored a Russian passage about Japan higher than
the English passage about Paris), so mixed-language corpora can behave differently from these
single-language pools.

## Cost on this machine

Uncached timing, two runs each (E5 and Qwen3 0.6B), one sample of 64 passages and 40 queries.

| Model | Model load / cold first query | Query p50 / p95 | Passages per second (batch 8) | Peak memory |
|---|---:|---:|---:|---:|
| E5-small, ONNX | 3.0 s load, then 0.01 s | 5.8–6.2 / 14.9–18.7 ms | 6.7–7.3 | 1.23 GiB (the Python process) |
| Qwen3 0.6B, Ollama | 3.4–4.0 s (first query after unload) | 272–364 / 518–564 ms | 0.29–0.39 | 2.77 GiB (the runner) + 54 MiB client |
| Qwen3 4B, Ollama | not measured | not measured | 0.59 on the 16-passage probe | 2.95 GiB right after load; 3.50 GiB while embedding |

E5-small embeds 17–25 times as many passages per second as Qwen3 0.6B and answers a query 45–63
times faster. The 0.6B runner holds 2.77 GiB resident on a 7.8 GiB machine that often has 1.5–3
GiB available, and `mem_available` was 1.3–1.4 GiB at the end of both 0.6B runs, so those timings
were taken under memory pressure; the compute gap is large enough that this does not change the
order. The Ollama adapter passes `keep_alive: "0"` by default, which unloads the model after every
request: with it each query pays the cold-start time above. The measurements here used a 30-minute
keep-alive; the product default is a separate decision (see ADR 0012).

Index size follows the dimension: 100,000 vectors take 73 MiB (384 dimensions) or 195 MiB
(1024 dimensions) in the fp16 index, and exact search cost grows with the dimension (not measured at 384).

Runtime dependencies: the ONNX path adds `fastembed`, `onnxruntime` (74 MiB installed), and
`tokenizers` (12 MiB) to the environment and runs in-process. The Ollama path adds nothing to the
Python environment but needs a running Ollama server (a separate install with its own weights
store) and an HTTP round trip per call.

## Qwen3 4B (not measured)

The quality run for `qwen3-embedding:4b` was started twice and did not finish:

1. The first attempt ended when the kernel's OOM killer terminated the Ollama service while it
   loaded the model (swap was 99 % full; the unit log records `oom-kill`).
2. The second attempt loaded the model (runner 2.95–3.50 GiB resident) but left 0.6 GiB available
   with swap full; 16 vectors were written in several minutes and the run was stopped by hand to
   avoid another OOM kill.

So the 4B model does not run usefully alongside a desktop session on this machine. This is a
finding in itself: the specification describes the target as 4 vCPU and about 7.8 GiB with often
only 1.5–3 GiB available, and a model that needs about 3.5 GiB resident is outside it. The cached vectors written so
far stay in `data/eval/nomiracl`, and the command above resumes the run once about 4 GiB is free.
On the earlier 16-passage probe 4B reached nDCG@10 1.000 against 0.984 for 0.6B, a difference that
probe cannot support (see below). No claim about 4B quality is made.

## Exact dense search

Synthetic unit-normalized float32 vectors inserted into the fp16 flat index; 40 top-10 searches
per size with `OPENBLAS_NUM_THREADS=4`; query embedding time excluded. Two runs on a loaded machine:

| Vectors × dimensions | p50 | p95 | Search process RSS |
|---|---:|---:|---:|
| 10,000 × 1024 | 20.6 / 41.9 ms | 25.4 / 53.8 ms | 56.9–70.7 MiB |
| 100,000 × 1024 | 203.0 / 221.7 ms | 216.1 / 255.4 ms | 248.1–267.2 MiB |

NFR-03 (p95 ≤ 100 ms at 10⁴, ≤ 400 ms at 10⁵) holds in both runs.

## Loading a real index (100,000 vectors)

The synthetic build above peaks at 942–947 MiB because it holds the float32 source array and the
fp16 index together; that is an artefact of the benchmark. The product path is measured by
`eval/scripts/benchmark_refresh.py`: a real schema-v3 SQLite database with published chunks and
1024-dimensional fp16 embeddings, loaded by `FlatVectorIndex.refresh` in a fresh process.

| Chunks | Database | Load time | RSS before / after load | Peak RSS | Search p50 |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 50.9 MiB | 0.53 s | 37.6 / 62.4 MiB | 80.7 MiB | 20.4 ms |
| 100,000 | 508.7 MiB | 7.84 s | 37.4 / 257.8 MiB | 275.9 MiB | 214.2 ms |

Loading streams rows and fills the fp16 matrix in place, so the peak is the matrix (195 MiB) plus
about 80 MiB, within NFR-01 (600 MiB). One run per size; the chunk texts are one character, so
the database is smaller than a real one, which does not affect the vector memory measured here.

## Diagnostic probe (16 passages, 10 questions)

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
