# 0016. The query pipeline: four statuses, one time budget, verified streaming by default

Status: Accepted (2026-11-27).

## Context

The pipeline joins retrieval, context assembly, the sufficiency gate and generation. Callers (the
command line now, an HTTP API next) need one answer type they can trust without re-deriving what
happened, and the machine is small enough that memory and time have to be managed by the pipeline.

## Decisions

1. **Four statuses and no exception for stage failures.** `answered`, `insufficient_evidence`,
   `unverified` and `error`. A failed stage is an `error` result naming its stage and stable code;
   a stream always ends with a final event. Cancellation is not a status: it propagates.
2. **The result type enforces the table.** Text and citations exist only on `answered`, which needs
   a real answer (NFR-08) and citations that point at its sources. A caller cannot be handed text
   that failed validation: the rejected text is kept apart as `raw_text` and shown only on request.
3. **The gate answers before the model.** An insufficient gate decision, no hits or an empty context
   end the request without a model call. An uncalibrated gate lets the question through and says
   so in the diagnostics; calibration quality is an open criterion (83.3 % against 90 %).
4. **One budget for the request.** Early stages get a share of the total, generation the rest;
   running out is `E_TIMEOUT` with the stage. It is cooperative and cannot interrupt synchronous
   code, which is stated rather than hidden.
5. **Streaming is verified by default.** The text of an answer is sent once, after its citations
   were checked. `--live` / `verified_only=False` streams provisional drafts and retracts them on
   failure; the final event is the only authoritative result.
6. **Memory hand-over is measured.** Before a local model is called, a one-shot process drops its
   in-process models and trims the heap. `release()` alone returned about a third of the resident
   memory in the one measurement made, so the trim is part of the hand-over; the effect is
   reported, not assumed. A service that stays up does not release.
7. **Unverified is not a soft answer.** `unverified_on_invalid` makes a rejected reply a status
   instead of an exception; it still never becomes `answered`. `strict_overlap` stays off by default
   because the overlap check is a lexical screen with false positives.

## Consequences

The command line and the future API share one code path and one result type. MVP-1 is
functionally in place on a scripted model; its quality acceptance depends on closing the gate,
real-model, faithfulness and dataset-review criteria, none of which this record claims.
