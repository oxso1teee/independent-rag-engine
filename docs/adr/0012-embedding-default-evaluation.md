# 0012. Default embedding model: multilingual-e5-small through ONNX

Status: Accepted (2026-11-27). Closes decision D-03. One measurement is missing (Qwen3 4B
quality); the consequence is stated under [Limits](#limits-and-when-to-revisit).

## Context

The default embedding model sets RU/EN retrieval quality, indexing speed, memory, disk and
licence obligations. The product supports Ollama models pinned by digest and local ONNX files
pinned by a directory fingerprint. The specification names the target machine: 4 vCPU, about
7.8 GiB RAM of which often only 1.5–3 GiB is available, no GPU. Candidates, as the specification
framed D-03 plus the larger sibling: `intfloat/multilingual-e5-small` (ONNX), `qwen3-embedding:0.6b`
and `qwen3-embedding:4b` (Ollama). A hashing test double cannot establish quality.

## Decision rule

The proposal of 2026-11-26 fixed the comparison before any model was ranked: Recall@5 and nDCG@10
with paired uncertainty intervals, plus throughput, latency, memory, size and licence. The order
of the criteria below, in particular the resource gate, was written down after the quality of
E5-small and Qwen3 0.6B had been seen; it follows the environment stated in the specification
rather than any score.

1. **Resource gate.** A default must fit the target machine: its resident memory should stay
   within the lower end of the memory that is usually available (1.5 GiB), because retrieval,
   the index and, in later phases, generation share it.
2. **Quality.** Among models that pass, a challenger replaces the cheaper candidate only if the
   paired 95 % interval of the pooled nDCG@10 difference excludes zero in its favour.
3. **Ties.** Prefer higher indexing throughput, lower query latency, fewer runtime parts and a
   permissive licence.

## Evidence

Full method, limits and tables: [`EMBEDDING_BAKEOFF.md`](../EMBEDDING_BAKEOFF.md). NoMIRACL
`dev.relevant` at a pinned revision, 40 queries per language, ranking among each query's judged
passages (judged-pool re-ranking, not full-corpus retrieval).

| | E5-small (ONNX) | Qwen3 0.6B (Ollama) | Qwen3 4B (Ollama) |
|---|---:|---:|---:|
| Pooled nDCG@10 (80 queries) | 0.796 [0.752, 0.838] | 0.804 [0.760, 0.847] | not measured |
| English / Russian nDCG@10 | 0.765 / 0.827 | 0.805 / 0.803 | not measured |
| Resident memory | 1.23 GiB | 2.77 GiB | 2.95–3.50 GiB |
| Passages per second | 6.7–7.3 | 0.29–0.39 | not measured |
| Query p50 | 6 ms | 272–364 ms | not measured |
| Licence | MIT | Apache-2.0 | Apache-2.0 |
| Vector size (100k, fp16) | 73 MiB | 195 MiB | 488 MiB |

Paired difference E5 − Qwen3 0.6B, nDCG@10: pooled −0.008 [−0.039, +0.022]; English −0.040
[−0.081, −0.006]; Russian +0.024 [−0.022, +0.068].

## Decision

**The default embedding model is `intfloat/multilingual-e5-small`, run offline through ONNX
Runtime** (extra `onnx`), pinned by the fingerprint of its model directory, with the prefixes
`query: ` and `passage: `, 384 dimensions and a 512-token window.

- Gate: E5-small is the only candidate that fits 1.5 GiB (1.23 GiB). The 0.6B runner (2.77 GiB)
  and the 4B runner (up to 3.50 GiB) do not, and the 4B model could not even be run to completion
  next to a desktop session (OOM kill, then memory exhaustion).
- Quality: no pooled difference with the 0.6B model (the interval spans zero), so the rule keeps
  the cheaper model. English favours 0.6B by about 0.04 nDCG@10; Russian does not.
- Tie-breaks agree: 17–25 times the indexing throughput, 45–63 times faster queries, an
  in-process model without a server, a 2.7-times smaller index, MIT licence.

Embedding stays disabled in the shipped configuration, and nothing is downloaded: the operator
provisions the files and pins the fingerprint (see `rag.example.toml`). Ollama models remain
supported for machines that can host them; any supported model can be configured explicitly.

## Consequences

- `rag.example.toml` and the README describe the E5-small ONNX setup. The adapter refuses any
  input over 512 tokens counted with the model's own tokenizer, special tokens included.
- Chunks must stay within the window: the default maximum of 384 tokens leaves a margin; the
  heuristic counter was at least as large as the real count for every passage of 256 tokens or
  more in the measured corpus, and the adapter checks exactly anyway.
- E5-small ranks a passage in the query's language above a better one in another language.
  Mixed-language corpora must be evaluated in Phase 5 with hybrid retrieval, not assumed.
- The Ollama adapter defaults to `keep_alive: "0"`, which unloads the model after every call and
  makes each query pay a 3–4 s cold start on 0.6B. Making `keep_alive` configurable is a follow-up
  for the retrieval phase.

## Limits and when to revisit

- The sample is 80 queries over judged pools; intervals are about ±0.04–0.05, so differences
  below that are not detectable. The scores do not predict full-corpus recall.
- **Qwen3 4B quality was not measured, and the owner decided not to measure it (2026-11-27).** The decision does not depend on it: the model fails the
  resource gate on the target machine whatever its quality. If a machine with at least 4 GiB of
  free memory becomes the target, resume the cached run
  (`bakeoff_nomiracl.py quality qwen3-embedding:4b`) and apply rule 2 again.
- Revisit when Phase 5 end-to-end retrieval evaluation (hybrid search on a full corpus) shows
  recall at fixed K below the target with E5-small, or when a multilingual ONNX model with a
  larger window and similar footprint is available to compare under the same rule.
