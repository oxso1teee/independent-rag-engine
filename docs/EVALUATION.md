# Evaluation protocol

What is measured, how a run is recorded, and what a result is allowed to claim. The numbers that
exist today are in [RETRIEVAL_EVALUATION.md](RETRIEVAL_EVALUATION.md), [EMBEDDING_BAKEOFF.md](EMBEDDING_BAKEOFF.md)
and [TESTING.md](TESTING.md); this page is the method.

## Rules

1. **A label is not true until a person accepted it.** Metrics count only `accepted` and `edited`
   records. A run that uses unreviewed records needs `--include-unreviewed` and is marked
   *exploratory* everywhere it is shown. No tool here marks a record as reviewed.
2. **Every number has an interval.** 95 % percentile bootstrap over questions (seed from the dataset
   hash); questions that share a gold document can be resampled as clusters
   (`stats.cluster_bootstrap`). Comparisons are paired: the bootstrap of the difference, the sign
   test, McNemar's exact test for binary metrics, Holm's correction across metrics.
3. **Warnings travel with the numbers.** Fewer than 30 questions is *anecdotal*. A metric at 0.98
   or above whose interval reaches 1.0 is *saturated*: it cannot show an improvement.
4. **The test split is held out.** `rag eval run --split test` is refused without `--final`; each
   read is counted in `eval_runs/test_access.json`, and a run whose split was read more than once
   says *not held out any more*.
5. **A run is reproducible.** Its manifest holds the dataset and corpus hashes, the configuration
   hash, the embedding and language models, the index generation, the seed, the engine version, the
   git commit and the resources at the time.

## Commands

`rag eval validate DATASET` checks the schema, the anchors against the corpus, the splits and the
reviewers. `rag eval run --dataset D --split dev|test [--final] --mode dense|lexical|hybrid
--output RUN.json` scores retrieval against the configured index. `rag eval report RUN` prints a run
as Markdown. `rag eval compare A B` prints the paired comparison. `rag eval gate RUN BASELINE`
exits 12 when a metric is worse than the baseline by more than `--delta` (0.03): the mean must stay
above the baseline mean minus delta, and, when the same questions were scored, the lower bound of
the paired difference must not fall below minus delta.

## Baselines

`baselines/mini_hashing_dev_v1.json` is the retrieval run of the public mini corpus with the
deterministic hashing embedder. It is checked in the test suite (`tests/e2e/test_eval_cli.py`): the
current code must pass the gate, and a deliberately shallower search must fail it. It tests the code
and the pipeline; it says nothing about the quality of a real model, and its labels are unreviewed.

## Generation metrics

`evaluation/generation_metrics.py` and `evaluation/judge.py` define the metrics of an answer:
faithfulness (a deterministic proxy of numbers and names, and a claim judge), hallucination rate
(share of answered replies with an unsupported claim), citation precision and recall, correctness
and completeness against the reference, and the answerable-by-answered matrix with refusal
precision and recall, false-refusal rate and false-answer rate. A judge is **calibrated** only when
it agrees with a human-labelled sample at 90 % or more with Cohen's kappa of 0.8 or more
(`calibrate_judge`); until then its verdicts are not evidence. The lexical judge is a baseline, and
no language-model judge has been calibrated: that needs a human-labelled sample that does not
exist yet and a judge model that fits in memory beside the generator.

## What is not established

- No metric over a *reviewed* dataset exists; the owner has not reviewed any label.
- No real language model has been run through the pipeline (the memory gate refused it), so no
  faithfulness, hallucination rate or generation latency is measured.
- The sufficiency gate does not meet its target (see the Phase 10 notes in [TESTING.md](TESTING.md)).
