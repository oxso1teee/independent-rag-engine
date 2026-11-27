# Retrieval evaluation

How retrieval quality is measured, what the committed question set can and cannot show, and the
first exploratory results. Decision record: [ADR 0013](adr/0013-default-retrieval-mode.md).

## Data

- **Corpus** `eval/corpus/mini_v1`: 18 short original documents about a fictional company: nine in
  English, eight in Russian and one mixed. Several are deliberate confusables (two backup policies
  in two languages, key management against passwords, incident response against on-call duty).
- **Dataset** `eval/datasets/retrieval_mini_v1.json`, schema v2 (`retrieval-mini@v1`): 44 questions,
  22 per language: 13 keyword, 17 paraphrase, 6 multi-document, 8 without an answer. Each answerable
  question carries gold documents with a verbatim anchor quote; each unanswerable one carries terms
  that must not occur in the corpus. Seven documents (16 questions) form the `test` split; questions
  that share a gold document always share a split.
- **Review status.** Every record is `unreviewed`. Labels are not truth until a person accepts
  them, so metrics are counted only from `accepted` and `edited` records, and a reviewer whose name
  looks like a program is refused. Until the owner reviews at least 30 dev questions, any run is
  labelled exploratory and cannot decide a default.

Validate the dataset against the corpus (anchors, absence terms, split policy, reviewers):

```python
from pathlib import Path
from rag_engine.evaluation.dataset import load_dataset, validate_dataset

dataset = load_dataset(Path("eval/datasets/retrieval_mini_v1.json"))
corpus = {p.name: p.read_text() for p in Path(dataset.corpus).glob("*.md")}
print(validate_dataset(dataset, corpus))
```

## Metrics

Scored per answerable question and summarised with 95 % percentile bootstrap intervals over
questions (2000 resamples, seed taken from the dataset hash), overall and by language and type:
`hit@k` (a gold document in the top *k*; its mean is coverage), `recall@5` (share of gold documents
found), `precision@5`, `mrr@10`, `ndcg@10` (chunk-level grades: 2 for a chunk that contains a gold
anchor, once per anchor; 1 for another chunk of a gold document), `evidence@5` (share of gold
anchors covered by the top five chunks), `all_gold@5` and `diversity@5`. Differences between modes
are paired over the same questions. A metric at 0.98 or above whose interval reaches 1.0 is flagged
saturated, and fewer than 30 questions is flagged anecdotal.

## Running it

```bash
uv run python eval/scripts/retrieval_eval.py \
    --model-dir data/models/multilingual-e5-small/<revision> --split all \
    --include-unreviewed --output eval_runs/retrieval_mini_v1.json
```

The script ingests the corpus with the real `rag ingest`, runs every question through the same
search service as `rag search` in dense, lexical and hybrid mode (top 10 of 40 candidates), and
prints the tables below plus per-stage latency. The JSON report records the dataset and corpus
hashes, the embedding model identity, the index generation and the seed. Without
`--include-unreviewed` it refuses to run until some questions are accepted.

## Exploratory result (unreviewed labels, 36 answerable questions, all splits)

Measured on 4 vCPU with `intfloat/multilingual-e5-small` through ONNX; mean and 95 % interval.

| Mode | hit@1 | hit@5 | mrr@10 | ndcg@10 | evidence@5 |
|---|---:|---:|---:|---:|---:|
| dense | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 [1.00, 1.00] | 1.000 |
| lexical | 0.861 [0.75, 0.97] | 0.972 [0.92, 1.00] | 0.917 [0.84, 0.98] | 0.935 [0.88, 0.98] | 0.972 |
| hybrid (RRF) | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 [1.00, 1.00] | 1.000 |

Paired differences of nDCG@10: hybrid − dense +0.000 [+0.000, +0.000]; hybrid − lexical +0.065
[+0.020, +0.121]; dense − lexical +0.065 [+0.020, +0.121]. Lexical on paraphrases reaches hit@1
0.706 [0.47, 0.88] (nDCG@10 0.868) against 1.000 for dense and hybrid.

Per-stage latency, p50 / p95 in ms: dense channel including query embedding 12.7 / 15.0, lexical
channel 0.9 / 1.2, fusion 0.1 / 0.2, diversity 0.8 / 1.1.

## What this does and does not show

- **The set is saturated for dense and hybrid.** Both retrieve the gold document first for every
  question, so the interval has no width and a difference between them cannot be measured. The
  only measurable result is that lexical alone is worse, significantly on paraphrases.
- 18 short documents with one answer each are easy; two-language confusables did not trip the
  multilingual model. A harder corpus (near-duplicate documents, versions that contradict each
  other, longer texts) is needed before hybrid and dense can be told apart.
- Labels are unreviewed drafts, so this is a check of the pipeline, not a quality claim.
- No-answer questions (8) are validated but not scored here; they belong to the sufficiency gate.
- Latency was measured on a machine shared with a desktop session; treat it as indicative.
