# 0021. `rag calibrate`: the gate is calibrated on the user's own index

Status: Proposed (2026-11-28). Not implemented. The audit ([AUDIT.md](../AUDIT.md), F-1) found that
nothing writes `index_state.calibration`, so the sufficiency gate of a real installation is always
open and reports `uncalibrated`. This record designs the missing command.

## Context

- Thresholds depend on the embedding model, the reranker, the corpus and the questions people ask.
  The held-out check ([ADR 0020](0020-gate-signal-precedence.md)) showed that a threshold fitted on
  one set can fall from 95 % to 72.5 % correct refusals on another, so shipped defaults would be
  claims nobody has verified. The gate must therefore be calibrated on data of the installation
  itself, and its quality must be reported as an estimate with its uncertainty, not as a promise.
- The pieces exist: `calibrate()` picks a threshold, `Thresholds` and the metadata codec store it,
  `SufficiencyGate` reads it at each query, `doctor` warns when it is stale, and
  `update_index_metadata` is a writer operation. The research scripts show how to collect the
  signals. Missing: a command, a place for the estimate, and staleness for the reranker.

## Decision

### Command

```
rag calibrate run QUESTIONS [--max-false-refusal 0.10] [--folds 5] [--dry-run]
                            [--holdout FILE] [--include-unreviewed] [--json]
rag calibrate show  [--json]
rag calibrate clear
```

- `QUESTIONS` is either an evaluation dataset (the existing JSON; reviewed questions only unless
  `--include-unreviewed`, which marks the result exploratory) or a JSON Lines file of
  `{"question": "...", "answerable": true|false}`. Gold references are not needed and not checked:
  calibration only needs to know whether the index should be able to answer. At most 5 000 rows,
  each question at most 2 000 characters; anything else is `E_QUERY_INVALID`.
- The questions run through the same retrieval path as `rag ask` (configured mode, `top_k`,
  `candidate_k`, reranker as configured), so the signal at calibration time is the signal at query
  time. Nothing is generated and no LLM is loaded.
- Read-only until the final step; `--dry-run` never writes. The write is one writer transaction on
  `index_state` (a second writer gets the usual retryable `E_STORE_LOCKED`).

### What is calibrated

Only the signal that will decide at query time (ADR 0020): the reranker logit when the reranker is
enabled in the configuration, otherwise the dense cosine. If neither exists (lexical-only mode),
the lexical coverage of the top passage. One threshold is stored, together with the signal's name.
At query time a stored threshold applies only when its signal is present; a gate calibrated for the
reranker is not applied to dense scores (it says `no_signal` and lets the question through).

### Minimum data and honesty about the result

- At least 20 answerable and 20 unanswerable questions, otherwise `E_QUERY_INVALID` with the counts
  missing. Below 30 per class the result carries the flag `anecdotal`, as in the evaluation reports.
- The threshold maximises correct refusals within `--max-false-refusal` (the existing `calibrate`).
- Reported next to the in-sample rates: **k-fold cross-validated** rates, and, with `--holdout FILE`,
  the rates on a second file that was not used for fitting, read once. The CV rates are what is
  stored as the estimate. The in-sample figure is never presented as the quality.
- The result is written even when the targets (≥ 90 % correct refusals, ≤ 10 % false refusals) are
  not met, because a gate that refuses 70 % of the unanswerable questions at 5 % false refusals is
  better than none. The output then says plainly: `estimated correct refusals 71 % (target 90 %)`.
  `--require-target` makes the command write nothing and exit 12, the existing code for an
  evaluation outcome that is worse than required (documented for this command too).

### Stored record

`index_state.calibration["sufficiency"]` gains optional fields (old records still load):

```
signal, dense|rerank|lexical (threshold), model_id, rerank_model (fingerprint, if used),
dataset (identifier or the file's SHA-256), questions {answerable, unanswerable},
estimate {cv_false_refusal, cv_correct_refusal, folds, holdout?}, exploratory,
index_generation, engine_version, created_at
```

No question text is stored. The record is replaced as a whole.

### Staleness

`SufficiencyGate` already ignores a calibration made for another embedding model. Added: the same
for a different reranker fingerprint. `rag doctor` warns, without blocking, when the record is
older than the last model switch, when the reranker differs, or when the number of active chunks
has changed by more than 50 % since calibration. `rag ask --json` already reports `uncalibrated`,
`calibration_invalid` or `no_signal` as the reason the gate was open.

### Placement

- fitting, cross-validation and the record codec: `retrieval/sufficiency.py` (pure, no I/O);
- orchestration (read questions, collect signals through the search services, write the record):
  `evaluation/gate_calibration.py`, which sits above retrieval and the store ports in the layer
  order and needs no new import contract;
- CLI group in `cli/calibratecmd.py`, registered like `register_eval`;
- no HTTP endpoint in this iteration: the write surface of the API stays narrow (ADR 0017).

### Resources

With the reranker the command loads the cross-encoder (about 2.4 GiB at peak). It goes through the
existing resource gate and the rule that a reranker and a resident LLM never coexist; models are
released when the command ends. On a machine that cannot hold the reranker the command refuses with
`E_RESOURCE_GATE` and the user can calibrate the dense signal with the reranker disabled.

## Alternatives rejected

- **Thresholds shipped per embedding model.** Unverifiable claims: the held-out result differed
  from the calibration result by 23 points, and thresholds depend on the corpus.
- **Calibrating automatically at ingest time.** There are no unanswerable questions to learn from;
  it would invent them or fit on answerable questions alone, which can only loosen the gate.
- **Unanswerable questions taken from an unrelated corpus.** Easy negatives overstate correct
  refusals; the numbers would look good and mean little.
- **A calibration over several signals (a fitted logistic model).** It overfit on the
  calibration data in the research and was no better than the reranker logit alone.

## Tests (to be written with the implementation)

- fitting: threshold within the allowed false refusals, cross-validation folds are disjoint and
  seeded, tiny classes are refused, `anecdotal` below 30;
- record: round trip, older records without the new fields still load, a stale model or reranker is
  ignored with the right reason, a signal that is absent lets the question through;
- command: end to end with the fake embedder (dry run writes nothing, a real run changes only
  `calibration_json`, `clear` restores `uncalibrated`, a locked store gives `E_STORE_LOCKED`,
  oversized or malformed input is refused, the output never contains question text);
- `doctor` warnings for the three stale cases;
- a documented, deterministic example (a committed small question file and a stub corpus) used by
  the quickstart.

## Consequences

- The gate becomes usable in a real installation, with a stated and cross-validated expectation.
  It still may not reach the 90 % target; the command reports that instead of hiding it.
- The user has to bring unanswerable questions. The documentation says how to write good ones
  (near misses of real topics, not nonsense) and that they must be reviewed by a person.
- Estimated work: one pull request, about 400 lines of code and 300 of tests.
