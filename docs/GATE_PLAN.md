# Plan: reaching 90 % correct refusals at 10 % false refusals

Status: proposed (2026-11-28). Nothing here is implemented; no target is changed, no criterion is
softened. The targets stay as written: **at least 90 % of the unanswerable questions refused before
generation and at most 10 % of the answerable ones refused**, measured on questions a person has
reviewed.

## Where it stands

| Evidence | Correct refusals | False refusals | What it is |
|---|---|---|---|
| reranker logit, 5-fold CV on the calibration sets (21 unanswerable) | 95.2 % | 6.7 % | in-sample data, small |
| reranker logit on `gate-heldout@v2` (40 unanswerable, 96 answerable), read once | **72.5 %** | 4.2 % | independent, unreviewed, **spent** |
| best dense cosine, same recipe, same set | 32.5 % | 6.3 % | spent |
| dense cosine on the earlier hard test split | 73.3 % | 8.6 % | spent |

Reading: the false-refusal target is met everywhere; the correct-refusal target fails by about 18
points on independent data, and a threshold fitted on the calibration sets lost about 23 points on
the next set. The probable causes ([AUDIT.md](AUDIT.md), G-1) were not isolated: few unanswerable
calibration questions, thresholds sitting at the edge of the allowed false-refusal rate, unanswerable
questions that are near misses of real topics (which no topical-similarity score separates), and
labels no person has reviewed. Dense alone cannot be the gate: its scores for unrelated text are
compressed around 0.8. The reranker is the only signal that has separated near misses, and it costs
2.4 GiB of memory at peak and about 3 s per query on the reference CPU, so it is off by default.

## Rules that do not bend

1. **Three roles, three disjoint sets of questions.**
   - *Development*: anything may be looked at and tried here.
   - *Calibration*: used only to fit thresholds of an already chosen method.
   - *Final*: written after the method and its thresholds are frozen, read **once**, never used for
     a choice of any kind.
2. **Spent sets stay spent.** `gate-heldout@v2` and the test splits already read for gate research
   (`retrieval-hard@v1`) are excluded from all three roles: not for tuning, not for calibration, and
   not for error analysis, because an analysis is a design input. They remain a historical result.
3. **The criteria do not move.** If the final evaluation fails, the answer is "not met" and a new
   final set is needed for the next attempt; the failed set is not reused, the target is not
   relaxed. Restating a target is an owner decision recorded in an ADR, with the evidence.
4. **Labels are reviewed by a person** before a set counts as development, calibration or final.
   Unreviewed data can guide, never decide ([DATASET_REVIEW.md](DATASET_REVIEW.md)).
5. **Resource limits stay**: no work package assumes more memory than the machine has, and any model
   run goes through the shared gate (model peak plus 1 GiB of headroom).

## What "met" means, and how big the sets must be

The criterion is the point estimate on the final set: correct refusals ≥ 90 % and false refusals
≤ 10 %. A point estimate on a small set proves little, so the final set is sized for confidence and
the result is reported with its 95 % Wilson interval. The table gives the smallest observed result
that also puts the lower (upper) bound on the right side of the target:

| unanswerable questions | refused at least (lower bound ≥ 90 %) | answerable questions | refused at most (upper bound ≤ 10 %) |
|---|---|---|---|
| 100 | 96 (96 %) | 100 | 4 (4 %) |
| 200 | 189 (94.5 %) | 200 | 11 (5.5 %) |
| 300 | 281 (93.7 %) | 300 | 19 (6.3 %) |

Minimum for the final set: **200 unanswerable and 300 answerable** questions (a half-width of about
4 points at 90 %). "Met" is the point estimate at or above the targets on at least that many
reviewed questions; "demonstrated" is the interval on the right side of them, and the release notes
say which one was reached.

## Work packages

### WP0. Data (the longest pole, and the owner's time)

- **A new corpus**, not the 26-document synthetic one: at least 150 documents in English and
  Russian, realistic length and structure (policies, runbooks, release notes, tables), including
  versions of the same document and overlapping topics. Public or synthetic text only.
- **Questions of four kinds** for each role: answerable in one passage; answerable across passages;
  unanswerable *near misses* (the topic is in the corpus, the asked fact is not; about 60 % of the
  unanswerable ones); unanswerable out of scope (about 25 %); partially answerable (about 15 %,
  labelled by what the corpus really supports).
- **Sizes**: development 150 answerable + 100 unanswerable; calibration 150 + 100 (no overlap with
  development, checked by the near-duplicate test of `rag calibrate`); final 300 + 200.
- **Drafting** by a model other than the one that will ever judge answers, then **human review of
  every question** with the review page ([DATASET_REVIEW.md](DATASET_REVIEW.md)); the automatic audit
  runs first so the person looks at the doubtful ones. Budget for the owner: about 700 questions in
  batches of 100.
- **Sealing**: the final set is committed as a hash (a file digest in the repository, the file kept
  outside the development environment) before WP1 starts; it is opened only in WP4.

### WP1. Diagnosis on the development set (no thresholds yet)

- Run the current signals (dense, rerank logit, lexical coverage) and record, per unanswerable
  question, which kind it is and what the top passages were.
- Answer one question with data: which kinds of unanswerable questions slip through, and why. Without
  this, any new signal is a guess.
- Output: a short report and a ranked list of hypotheses. Exit criterion: a stated reason for the
  72.5 % that survives a second look.

### WP2. Candidate methods, chosen by a rule fixed in advance

Each candidate is a hypothesis with a cost; none is adopted without its numbers on the development
set. Order by expected value for the cost:

1. **Reranker logit as the only gate signal, with a pool-limited reranker.** Score only the top 5 to
   8 candidates instead of the pool of 30, measure the latency and memory, and keep the reranker out of
   the default path (an opt-in `gate = "rerank"`). Cost: 2.4 GiB peak; may be unusable on 8 GiB with a
   resident chat model, which the existing validator already forbids.
2. **A smaller cross-encoder** that fits the machine (licence and size checked first; no new
   dependency without an ADR). Replaces item 1 if it separates as well.
3. **Answer-support features from the retrieved passages**: the share of the question's rare terms
   found in the best passage, and the margin between the best and the fifth passage. Cheap and
   explainable; it did not beat the reranker logit on the old calibration data, but the old data was
   small, so it is retried on the new development set, with at most three features and a monotonic,
   strongly regularised model fitted on the calibration set only.
4. **A second stage after generation**: count a refusal by the model (`<<INSUFFICIENT>>`) and an
   unverified citation as part of the refusal, and report the pipeline-level correct-refusal rate next
   to the gate's. This does not change the gate criterion; it measures what a user experiences, and
   needs a real model (see [LLM_VERIFICATION.md](LLM_VERIFICATION.md)).
5. **An entailment-style check** (question and passage in, "supported or not" out) with a small
   multilingual model. Highest cost (a model, a licence, memory); only if 1 to 3 fail on development.

**Selection rule, written down before WP2 starts:** the simplest method whose paired-bootstrap
interval on the development set shows correct refusals ≥ 90 % at false refusals ≤ 10 %, with a
latency and memory the product can carry, wins; complexity is a cost, not a goal. A method that only
reaches the target with a threshold tuned to the development set is not eligible.

### WP3. Calibration

- Fit the chosen method's thresholds on the calibration set only, with the existing procedure
  (`calibrate`, cross-validation, `rag calibrate`), at the allowed false-refusal rate of 10 %; record
  the cross-validated estimate next to the threshold.
- Freeze: method, thresholds, code revision, the model files' digests. Write them into a record
  (`docs/adr`) *before* the final set is opened.
- A second look is allowed only on the development set, and anything learned there restarts WP3.

### WP4. Independent final evaluation

- Open the sealed final set (digest verified), run once, report correct and false refusals with
  Wilson intervals, by question kind, with the counted read.
- **Pass:** the point estimates meet the targets on at least 200 / 300 reviewed questions.
  **Fail:** record the numbers in an ADR, keep the criterion open, and start again from WP0 with a
  *new* final set; do not adjust anything on the failed one.

### WP5. Product wiring

Only after a pass: ship the method as an option of the gate (the record already carries the
algorithm version), keep `rag calibrate` for the user's own index, document the reranker's resource
cost next to it, and re-run the soak with it enabled.

## Risks and stop conditions

- **The target may not be reachable on this hardware** with the signals that fit in 8 GiB. WP2's
  first two candidates answer that early, on development data. If neither reaches the target, the
  honest outcome is an ADR proposing options (a bigger memory budget, a different gate, a different
  target) for the owner to choose; the engine ships with the gate calibrated per installation and the
  limitation stated.
- **Label quality.** Near-miss questions are the hard cases and the easiest to label wrongly; every one
  is reviewed by a person, and disagreements are resolved before the set is used.
- **Overfitting by repetition.** Each pass over the development set is a degree of freedom; the plan
  limits WP2 to the listed candidates and one selection rule.
- **Cost.** WP0 is mostly the owner's review time; WP1 to WP4 are a few days of engineering once the
  data exists.

## Order and deliverables

1. WP0 data (corpus, sets, reviews, sealed digest) → *deliverable:* three reviewed sets and the
   committed digest.
2. WP1 diagnosis → a report.
3. WP2 candidates → a table of development-set results and the chosen method.
4. WP3 calibration → the frozen record.
5. WP4 final → an ADR with the result, pass or fail.
6. WP5 wiring (only on a pass).
