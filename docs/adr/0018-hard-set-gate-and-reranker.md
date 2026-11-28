# 0018. What the harder set showed about the gate, the reranker and hybrid retrieval

Status: Accepted (2026-11-28). Provisional: every number below is on **unreviewed** labels.

## Context

The mini set is saturated (dense and hybrid both score 1.0), so it could not separate retrieval
variants, decide the reranker, or calibrate the sufficiency gate (83.3 % correct refusals against a
90 % target). Phase 10 asks to return to these questions on data that can discriminate.

## Evidence

`retrieval-hard@v1` (26 documents, 141 questions, hard negatives, contradictory versions, questions
across languages, near-miss unanswerable questions), drafted with an AI language model and unreviewed. Results
and method are in [EVALUATION.md](../EVALUATION.md). In short:

- No gate rule reached 90 % correct refusals on the held-out split; the best simple rule (dense
  cosine) gave 73.3 % at 8.6 % false refusals; a fitted combination overfit.
- The reranker raised nDCG@10 by +0.16 (dense) and +0.37 (hybrid) on dev, intervals above zero.
- Hybrid fusion was worse than dense on this multilingual set.

## Decisions

1. **The gate criterion stays open.** It is not closed, relaxed or re-defined. The stored default
   (no calibration: the question passes) is unchanged, and a calibration is still only a per-index
   artifact. The "dense OR coverage" combination is rejected as a policy.
2. **The reranker stays off by default** (ADR 0014 amended, not replaced). Its conditions were a
   gain confirmed on dev **and** test, p95 within 1.5 s and memory within budget. The gain is shown on
   dev, but on unreviewed labels; test was not read for it; p95 is 4.0 s and the peak
   2.43 GiB. It is recommended as an opt-in for quality-critical use on a machine with memory to spare.
3. **Dense stays the default retrieval mode** (ADR 0013 confirmed); hybrid is worse on a multilingual
   corpus with cross-language questions and is not recommended there.
4. **The test split of the hard set has been read once** (for the gate rules). A second read ends
   its held-out status, so further decisions need new, reviewed data.

## What would change these decisions

A reviewed dataset (the owner's review of the 141 questions, or a larger one), a test read of the
reranker on it with latency inside the limit, and a gate signal evaluated on at least 30
unanswerable questions per split.
