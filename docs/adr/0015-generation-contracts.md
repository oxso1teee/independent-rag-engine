# 0015. Generation contracts: refusal sentinel, provisional streaming, lexical overlap

Status: Accepted (2026-11-27).

## Context

Generation sits after retrieval and context building and must never present a source that was not
supplied, an empty or looping reply, or an answer cut by a silently truncated prompt. It has to
work with local models of unknown tokenizers on a small shared machine, and with servers that
stream.

## Decisions

1. **The refusal is a sentinel, not a sentence.** The model answers exactly `<<INSUFFICIENT>>` when
   the evidence does not contain the answer. A natural-language refusal would have to be matched in
   every language and would be confused with answers that merely mention missing information.
   A reply that mixes the sentinel with prose is degenerate and retried.
2. **Citations are checked against the context, in two levels.** Level 1 is structural and fatal
   after one retry: a marker that names no evidence of this context, or no citation at all
   (`E_CITATION_INVALID`). Level 2 is a lexical overlap between a cited sentence and its evidence.
   It is deliberately **not** called grounding or faithfulness: it detects a marker pasted on an
   unrelated passage, not a paraphrase that distorts the meaning, and it is blind across scripts.
   It is reported by default and fatal only with `strict_overlap`. Faithfulness is measured in the
   evaluation phase, with labelled data.
3. **Streaming is provisional unless asked otherwise.** Citations can only be validated on the
   finished text, so live deltas are drafts. A failed validation or a provider failure after text
   was sent emits a `RetractEvent` before the error; `DoneEvent.answer` is the only validated text.
   `verified_only=True` trades first-token latency for the guarantee that unvalidated text is never
   sent. The alternative of always buffering would remove streaming's reason to exist for
   interactive use; the alternative of silently streaming and failing afterwards was rejected.
4. **Failures are classified, not retried blindly.** Only retryable "unreachable, 5xx, slow down"
   errors are retried; a timeout is not (its deadline is spent); a 4xx is not; an out-of-memory
   failure is `E_RESOURCE_GATE` and is never retried, so a retry cannot repeat a crash. A breaker
   fails fast after three consecutive failures for 60 s.
5. **Reasoning is off by default** (`think: false`) and a `thinking` field is never returned: it
   would consume the answer budget and is not evidence.
6. **Real-model runs are gated.** `eval/scripts/llm_smoke.py` and the opt-in external test refuse
   when available memory is below the model's size plus 1 GiB or free swap is below 0.5 GiB.
7. **A chat model is not kept resident by default.** `keep_alive` is `0`: 1.2 GiB (E5) plus about
   3.5 GiB (a 4B model) is already more than the 1.5 to 3.7 GiB usually available, and with the
   reranker (2.5 GiB) it is not possible at all. Configuration with the reranker enabled and a
   resident Ollama model is rejected. Longer values stay available for machines with room.

## Consequences

The generation layer is testable without a model (`ScriptedLLM`) and without a network
(`httpx.MockTransport`); the standard CI needs neither weights nor sockets. A prompt layout or
system prompt change must update the pinned hash and the snapshot in the same commit.
Token counts for streams are estimates. Entailment-level faithfulness remains open until Phase 10.
