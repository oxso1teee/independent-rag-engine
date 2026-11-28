# Reviewing an evaluation dataset

An evaluation dataset is a set of questions with the passages that answer them. Until a person has
checked a question, it is a draft: metrics are computed only from questions marked `accepted` or
`edited`, and a reviewer name that looks like a program is refused. This page describes how to
review `eval/datasets/retrieval_mini_v1.json` (44 questions over `eval/corpus/mini_v1`).

## Batch review with the automatic audit (a table, not 185 cards)

`audit_questions.py` screens every question against its own documents and sorts them into three
groups. `review_kit.py build --audit` puts the result in the review page as a compact table.

```bash
uv run python eval/scripts/audit_questions.py                      # lexical layer: nothing to load
uv run python eval/scripts/audit_questions.py --embed-dir <e5 dir> \
    --output eval_runs/question_audit_semantic.json                # plus the semantic layer
uv run python eval/scripts/review_kit.py build --audit eval_runs/question_audit_semantic.json
```

| Group | Meaning |
|---|---|
| **correct** | no objection found and the evidence is linked to the question |
| **doubtful** | look at it: another passage fits as well, the answer has words the evidence lacks, the answer leaves out part of the evidence, or the link between question and evidence could not be shown |
| **incorrect** | wrong on its face: a figure in the expected answer that no gold document contains, an evidence quote that has nothing to do with the question or the answer, a "missing" term that is in the corpus, gold evidence on an unanswerable question, a label contradicted by the text |

What it can and cannot do. The **lexical layer** compares stems, figures and identifiers. It is blind
across languages and for paraphrases by design, so on its own it leaves those as "doubtful: the link
could not be shown". The **semantic layer** ranks every passage of the corpus for each question with
the E5 model and says whether the gold passage is the best match, or how far down it is (ranked among the passages written in the evidence's language, because the model prefers the question's own language; an unanswerable question is flagged when its best passage scores like the lower quartile of the answerable ones); it refuses
to start below 2.3 GiB of free memory and a watchdog stops it if free memory falls under 0.45 GiB.
Neither layer understands the question. "correct" means no automatic objection, **not** reviewed,
and nothing here writes to a dataset, fills in a decision or marks a question as reviewed.

The table (default view of the page) is sorted incorrect, doubtful, correct. Each row shows the
question, the expected answer, the findings and your decision; the arrow opens the evidence with its
surrounding text, the audit's reasons and, for an unanswerable question, the closest passages.

**Group confirmation.** Filter or select rows (all shown, only the shown "correct", or by hand), open
their evidence (one by one, or "Open evidence of all shown"), then *Accept selected* or *Reject
selected…* (one reason for all). A group action applies only to rows that are both selected and
opened; it tells you how many it skipped, mentions how many of the accepted rows carry an audit
objection, and asks you to type the number of questions before it records anything. Decisions stay in
the page (and the browser's storage) until you press *Export decisions (CSV)*; nothing changes in a
dataset until `review_kit.py apply` records them under your name. Edits (a corrected question or
answer) are made in the card view; a wrong gold quote cannot be edited, so reject the question.

## Reviewing all questions in one page (review kit)

`review_kit.py` builds one self-contained HTML page for every question of the hard set (141) and
the mini set (44): 185 cards, each with the question, the source document with the gold quote
highlighted, the reference answer and facts, and for questions without an answer the terms that
must not occur. Nothing is sent anywhere and nothing in a dataset changes while you work.

1. Build the page (`--dataset` may be repeated; the 136 questions of
   `eval/datasets/gate_heldout_v2.json` can be added the same way):

   ```bash
   uv run python eval/scripts/review_kit.py build
   ```

2. Open `eval_runs/review_kit/review_kit.html` in a browser. For each card choose *accept*,
   *edit* (with the corrected question and/or reference answer) or *reject* (with a reason).
   Progress is kept in the browser; leave a card undecided to leave the question a draft.
3. Export the decisions as CSV with the button on the page.
4. Apply them under your name. One new file per dataset is written with the version raised by one;
   an existing file is never overwritten, and a reviewer name that looks like a program is refused:

   ```bash
   uv run python eval/scripts/review_kit.py apply --sheet <exported.csv> \
       --reviewer "Your Name" --outdir eval/datasets
   ```

5. Point the evaluation scripts at the new versions and re-run them
   (`docs/EVALUATION.md`). The re-run, not this page, decides what the reviewed numbers are.

The kit never marks a question reviewed by itself; that is only possible through step 4.

## Steps (CSV sheet, one dataset)

1. Export a sheet (nothing in the dataset changes):

   ```bash
   uv run python eval/scripts/review_sheet.py export --output eval_runs/review.csv
   ```

   Open it in a spreadsheet. Each row is a question with its type, split, the gold quote and the
   text around it, the reference answer and, for questions without an answer, the terms that must
   not occur in the corpus.

2. Fill in the `decision` column for every row you have judged:

   | decision | meaning | also fill in |
   |---|---|---|
   | `accept` | the question and its labels are correct as they are | optional `notes` |
   | `edit` | correct after your change | `edited_question` and/or `edited_reference_answer` |
   | `reject` | the question should not be used | `notes` with the reason |
   | (empty) | not reviewed yet; the question stays a draft | |

3. Record the decisions under your name. This writes a new file and never overwrites the one it
   read:

   ```bash
   uv run python eval/scripts/review_sheet.py apply --sheet eval_runs/review.csv \
       --reviewer "Your Name" --output eval/datasets/retrieval_mini_v2.json
   ```

   The result is validated (anchors, absence terms, split policy, reviewer). Labels are immutable
   per version, so the reviewed file is a new version: set `"version": 2` in the output before it
   replaces the old one, and point the evaluation scripts at it.

## What to check

- **Is the gold quote really the answer?** The anchor must be a verbatim quote that answers the
  question, and the document it is in must be the one that should be found. For multi-document
  questions both documents must be needed.
- **Is the question unambiguous and answerable from the corpus alone?** Another document must not
  answer it equally well; if one does, add it or reject the question.
- **Paraphrases:** the question should not repeat the anchor's words (a lexical overlap above 30 %
  is flagged by the validator), or it tests keyword search, not meaning.
- **Questions without an answer:** the topic must really be absent from the corpus, not only the
  listed terms.
- **Language:** a Russian question needs Russian wording that a native speaker would use.
- **The reference answer** must be supported by the quote and short enough to compare with.

## What the review changes

With at least 30 accepted dev questions the evaluation reports stop being labelled exploratory;
ADR 0013 (default retrieval mode), the sufficiency gate calibration and the reranker decision
(ADR 0014) are then re-run on reviewed labels. Fewer than 30 stays labelled anecdotal.
