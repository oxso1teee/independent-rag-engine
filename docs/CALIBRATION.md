# Calibrating the sufficiency gate

Before the model is called, the engine can refuse a question whose retrieved evidence is plainly too
weak. It does so by comparing one retrieval score with a threshold. That threshold depends on your
documents, your embedding model and the way your users ask, so it is **not shipped**: you fit it on
your own index with `rag calibrate`. Until you do, the gate is open: every question reaches the
model, and every answer says so in its warnings.

## The workflow

```bash
rag source add notes ./notes          # 1. index your documents
rag ingest notes
rag calibrate run questions.jsonl     # 2. fit the gate on questions with and without an answer
rag calibrate show                    # 3. see what was stored and whether it still applies
rag ask "How do the nightly backups work?"      # 4. weak evidence is now refused before the model
rag doctor --integrity                # 5. warns when the calibration is missing or stale
```

### The questions file

JSON Lines, one object per line:

```json
{"question": "When do the nightly backups run?", "answerable": true}
{"question": "What is the guest wifi password?", "answerable": false}
```

- `answerable: true`: the indexed documents contain the answer.
- `answerable: false`: they do not. Write **near misses**: questions about topics your documents
  touch but do not answer. Obviously unrelated questions are refused by any threshold and make the
  estimate look better than it is.
- At least 20 of each kind are needed; fewer than 30 of a kind gives a result marked anecdotal.
  A person should write or review the questions; a language model's draft is a starting point.
- A repository evaluation dataset (`eval/datasets/*.json`) also works; its `dev` split is used, and
  only questions a person accepted unless `--include-unreviewed` is given (the result is then
  exploratory).

### What the command does

It runs every question through the same retrieval as `rag ask` (the configured mode, depth and
reranker), takes the score that will decide at query time (the reranker's logit if the reranker is
enabled, otherwise the dense cosine) and picks the threshold that refuses as many unanswerable
questions as possible while refusing at most `--max-false-refusal` (default 10 %) of the answerable
ones. No model is called.

It then reports an **estimate by cross-validation** (`--folds`, default 5): each question is judged by
a threshold fitted without it. The rates on the questions used for the fit are shown too, labelled
optimistic, and are never the quality. With `--holdout FILE` a second file, which must not share
questions with the first (exact or near duplicates are refused), is scored once with the fitted
threshold.

```text
signal dense, threshold 0.8123
  cross-validated estimate: correct refusals 83.3% (24 unanswerable), false refusals 4.2% (24 answerable)
  on the calibration questions (optimistic): correct refusals 91.7% ...
targets: correct refusals at least 90%, false refusals at most 10%
warning: estimated correct refusals 83.3% (target 90%), ...: the gate is written but below its targets
calibration written
```

The threshold is written even when the estimate is below the targets, because a gate that refuses
most weak questions is better than none, and the result says plainly that it is below. Use
`--require-target` to write nothing and exit with code 12 in that case, and `--dry-run` to see the
result without writing.

## What a calibration is valid for

The stored record is bound to everything that changes the scores. It is ignored, and the gate stays
open with a warning, when any of these differ from the calibration:

| What | Warning reason |
|---|---|
| the active embedding model | `calibration_invalid` |
| the decision rule's version (the engine changed how it decides) | `calibration_outdated` |
| the retrieval of the query: mode, `top_k`, `candidate_k`, whether the reranker ran; and the chunking, the lexical analyser and the reranker model | `calibration_stale` |
| the score it was calibrated on is missing for this query (for example the reranker was off) | `no_signal` |

So `rag ask --top-k 3` against a calibration made with the default depth is let through and says why,
rather than being judged by a threshold fitted for other scores. Calibrate with the settings you run.
Lexical-only retrieval has no score to calibrate and cannot be calibrated.

`rag doctor --integrity` also warns when there is no calibration, when it was made for another
model or rule, when the number of active documents has changed by more than half since, and when the
stored estimate was below the targets. After re-indexing with a different model, chunking or
reranker, run `rag calibrate run` again.

## What to expect

A gate is not a guarantee. In the project's own checks the same recipe refused 95 % of the
unanswerable questions on the data it was fitted on and 72.5 % on a fresh set of near misses (see
[ADR 0020](adr/0020-gate-signal-precedence.md)). The cross-validated estimate is the figure to
believe, and a larger and more varied question file makes it more trustworthy. The gate works before
the model; the model's own refusal and the citation checks after it remain.

Nothing is stored about your questions except the file's identity (a name or a digest) and the
counts; the question texts never enter the database or the log. Writing is one transaction that
replaces the whole record or none of it, and is refused if the index was published again or the
embedding model switched while the questions were being measured (run the command again).

## Commands

| Command | Does |
|---|---|
| `rag calibrate run QUESTIONS` | fit and store; `--holdout FILE`, `--max-false-refusal X`, `--folds N`, `--dry-run`, `--require-target`, `--include-unreviewed`, `--json` |
| `rag calibrate show` | the stored record, whether it applies, and why not |
| `rag calibrate clear` | remove the record; the gate is open again |
