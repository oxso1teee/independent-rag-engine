# 0013. Default retrieval mode stays dense; hybrid is an option

Status: Accepted (2026-11-27), provisional until the labels are reviewed.

## Context

Search offers three modes: dense (exact cosine over fp16 vectors), lexical (FTS5 with BM25 and
Russian/English stemming) and hybrid (reciprocal rank fusion of both, `k = 60`, weights 1.0 and
0.5). The retrieval design makes hybrid the default only if, on a human-reviewed dev set of at
least 30 questions, the lower bound of the paired 95 % interval of the Recall@5 and nDCG@10
difference exceeds -0.02 and the point difference is not negative. It also records an earlier
observation that hybrid can be worse than dense.

## Evidence

[`RETRIEVAL_EVALUATION.md`](../RETRIEVAL_EVALUATION.md): the committed mini corpus (18 documents,
36 answerable questions, English and Russian) with `intfloat/multilingual-e5-small`.

- dense and hybrid score 1.000 on every metric; the paired difference is exactly 0.000 with a
  zero-width interval. The set is saturated, so it cannot separate them.
- lexical is lower: nDCG@10 0.935 [0.88, 0.98], and 0.868 on paraphrases; hybrid − lexical is
  +0.065 [+0.020, +0.121].
- None of the 44 questions has been reviewed by a person, so the requirement of at least 30
  accepted dev questions is not met.

## Decision

1. The default mode in `[retrieval]` remains `dense`. Hybrid and lexical stay available through
   `--mode` and the configuration.
2. Fusion parameters stay at the starting values (RRF `k = 60`, dense 1.0, lexical 0.5). They were
   not tuned: with no measurable difference there is nothing to tune against.
3. The decision is provisional. It is revisited when the owner has reviewed at least 30 dev
   questions and the corpus is hard enough for the evaluation not to be saturated.

## Why not hybrid now

The rule is formally satisfied by a zero difference, but a zero difference on a saturated,
unreviewed set is the absence of evidence, not non-inferiority. Choosing hybrid would add a
second index lookup and a fusion step to every query without a measured benefit, and the design
notes an earlier case where it hurt. Dense costs nothing extra: the lexical index is maintained
regardless.

## Consequences

- Lexical-only queries (identifiers, rare terms) need `--mode hybrid` or `--mode lexical`; in the
  test of the search command a rare serial number is found by hybrid.
- A harder, reviewed evaluation set is the next evaluation task; the saturation flag in the
  report is the signal that a set has stopped being informative.
- Mixed-language corpora, where E5-small prefers a passage in the query's language
  (ADR 0012), are not covered by this set and may favour hybrid.
