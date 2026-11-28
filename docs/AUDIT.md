# Audit of the first release candidate

Audit of main at `d010b89` (Phases 0–11), carried out on 2026-11-28 on the reference machine
(4 vCPU, 7.8 GiB, Linux, Python 3.12.3). Verdicts: **PASS** (met, with evidence), **FAIL** (measured
and not met), **NOT VERIFIED** (nothing was measured that could show it), **DEFERRED** (out of the
first release by decision). Requirement IDs are defined in [REQUIREMENTS.md](REQUIREMENTS.md);
per-phase evidence tables are in [TESTING.md](TESTING.md). No target was changed to fit a result.

## Summary

| | |
|---|---|
| Functional requirements FR-01…19 | all PASS except FR-14 (the gate stage), see F-1 and G-1 |
| FR-20…22 (MCP, ANN, agentic search) | DEFERRED, outside the first release |
| NFR-01, 03–12 | PASS (NFR-12: Linux verified, macOS NOT VERIFIED) |
| NFR-02 | environment PASS, database-size clause FAIL (F-2) |
| Quality targets | **FAIL / open**: sufficiency gate; **NOT VERIFIED**: real model, reviewed labels, judge |
| Tests | 2006 passed, 11 skipped (opt-in external), 14 perf tests run separately; coverage 95 % of `src/` |
| Static checks | ruff, mypy strict (src, tests, eval, scripts), 11 import contracts, notices/SBOM check, gitleaks, pip-audit |

## Functional requirements

| ID | Verdict | Evidence |
|---|---|---|
| FR-01 | PASS | filesystem source and managed API sources: `tests/integration` (scan, globs), `tests/api` (upload, managed-source rule) |
| FR-02 | PASS | identity and explicit rebase: `tests/integration/test_identity*`, `tests/integrity` |
| FR-03 | PASS | change detection and rescan with zero parser calls: `tests/perf/test_nfr04_rescan.py` (1000 files, 0.15 s) |
| FR-04 | PASS | parser suites per format incl. isolated worker; limits on archives and nesting |
| FR-05 | PASS | structure-aware, recursive, fixed chunkers; offsets and `heading_path` property tests |
| FR-06 | PASS | normalisation and in-document de-duplication tests |
| FR-07 | PASS | contract suite per provider; cache, batches, retries; real E5 ONNX `tests/external` (opt-in) |
| FR-08 | PASS | migrations incl. real v1 upgrade, integrity check, model-based ingestion test |
| FR-09 | PASS | dense, BM25, RRF, filters DSL; retrieval metrics in [RETRIEVAL_EVALUATION.md](RETRIEVAL_EVALUATION.md) |
| FR-10 | PASS | optional cross-encoder with fallback, off by default (ADR 0014, 0018); memory guard forbids reranker plus resident LLM |
| FR-11 | PASS | packer and builder tests, truncation report in diagnostics |
| FR-12 | PASS (stub and scripted) | generation, streaming, retries, resilience: `tests/unit`, `tests/e2e`; **no real model run, see L-1** |
| FR-13 | PASS | citation parsing and validation levels 1–2 (lexical overlap is a screen, not entailment: documented) |
| FR-14 | PARTIAL | statuses, post-checks and refusal marker PASS; the pre-generation gate is available but never calibrated for a user (F-1) and misses its quality target (G-1) |
| FR-15 | PASS | versions, soft delete, GC: `tests/integrity` |
| FR-16 | PASS | CLI exit codes and RFC 9457 problems; committed OpenAPI snapshot `docs/openapi.json` checked in tests |
| FR-17 | PASS | protocol with split guard, paired statistics, baselines, regression gate |
| FR-18 | PASS | `doctor` resource gates and integrity |
| FR-19 | PASS | backup, verify, restore drill: `tests/e2e/test_admin_cli.py` |

## Non-functional requirements

| ID | Verdict | Measurement |
|---|---|---|
| NFR-01 | PASS | idle server 88 MiB (limit 150); 10⁴ chunks peak 81 MiB, 10⁵ chunks 276 MiB (limits 300 / 600) |
| NFR-02 | environment PASS; database FAIL | environment without models: 82 MB core, 150 MB with every optional extra (limit 300 MB, no torch). Database: about 7–9× the source text against the target of about 3× (F-2) |
| NFR-03 | PASS | dense p95 25–54 ms at 10⁴, 216–255 ms at 10⁵ (limits 100 / 400 ms) |
| NFR-04 | PASS | rescan of 1000 unchanged files 0.15 s (limit 5 s), 0 parser calls |
| NFR-05 | PASS | incremental re-embedding test: a one-paragraph edit re-embeds at most 3 chunks |
| NFR-06 | PASS | `tests/chaos`: `SIGKILL` at every ingestion stage, database opens, integrity passes, re-run finishes |
| NFR-07 | PASS | truncation and drops are in diagnostics and log; tests per stage |
| NFR-08 | PASS | `min_answer_chars` 20 enforced by `PipelineAnswer` |
| NFR-09 | PASS | unit suite offline (`pytest-socket`), deterministic Hypothesis profile |
| NFR-10 | PASS | non-loopback bind refuses without token; the OpenAPI document was found open and is now behind the token (D-1); typed filter DSL; secrets from environment; logs without text |
| NFR-11 | PASS | `uv sync --frozen`, lock checked in CI, hashed Docker install, actions pinned by SHA (D-4) |
| NFR-12 | PASS on Linux; macOS NOT VERIFIED | Python 3.12 and 3.13 jobs in CI |

Other measured facts: one-hour soak, 60 searches, 0 errors, resident memory -3.3 % ([TESTING.md](TESTING.md));
Docker image 80 MB, non-root, healthy, refuses to start without a token; `make perf` 14 of 14 within budget.

## Defects found by this audit

| # | Finding | Resolution |
|---|---|---|
| D-1 | The OpenAPI document was served without the token while every other route (but `/health`) required it | moved behind the token; test added |
| D-2 | `rag.example.toml` lacked a dozen settings that exist | all documented; a test fails when a setting is missing from the example |
| D-3 | Public text spoke of "a specification" and named an assistant as author; two pull-request descriptions did too | reworded to "requirements" and "an AI language model"; the remote descriptions edited; history untouched |
| D-4 | CI actions were referenced by moving tags | pinned by commit SHA; Dependabot for actions, Docker and uv |
| D-5 | The gate's policy "dense OR rerank" can never refuse more than the weaker signal; on held-out data dense alone refused 32.5 % of unanswerable questions, the reranker 72.5 % | the reranker logit now decides alone when it ran; dense, then lexical, otherwise (tests updated) |
| D-6 | The generation evaluation reported only mechanical metrics, and "not measured" for the rest | `evaluation/llm_run.py` adds correctness, completeness, faithfulness, hallucination rate, citation precision and recall, the answered/refused matrix, latency and tokens; wired into `generation_eval.py` (also for an OpenAI-compatible endpoint) |
| D-7 | No way for the owner to review 185 questions in one sitting | review kit (HTML page, CSV export, applying under a person's name), tested and tried in a browser |

## Findings not fixed

| # | Finding | Why not, and what is needed |
|---|---|---|
| F-1 | **No command writes the gate calibration.** `index_state.calibration` is read but only the research scripts compute thresholds, so a user's gate is always open and says `uncalibrated` | needs a design decision: a `rag calibrate` over the user's own questions, or shipped thresholds per embedding model. Recommended before any claim about refusing weak evidence |
| F-2 | Database about 7–9× the source text, target about 3× | a schema change (vector encoding or full-text options) is too risky for a hardening release; owner decision whether to relax the clause or schedule the work |
| F-3 | Coverage gaps: the parser worker subprocess (0 % in-process, covered by e2e through the worker), `cli/ask_render.py` 72 % | acceptable for the release, listed here |
| F-4 | The memory gate refuses the 4B model on this machine | by design; see L-1 |

## Quality targets

### G-1 Sufficiency gate: FAIL (open)

Targets: at least 90 % correct refusals and at most 10 % false refusals.

| Data | Rule | Correct refusals | False refusals |
|---|---|---|---|
| calibration sets (hard dev + mini dev, 21 unanswerable, 5-fold CV) | reranker logit | 95.2 % | 6.7 % |
| **independent held-out** `gate-heldout@v2` (40 unanswerable, 96 answerable), opened once | reranker logit, chosen before opening | **72.5 %** (about 57–84 % at 95 %) | 4.2 % |
| same | best dense cosine, same recipe | 32.5 % | 6.3 % |
| earlier hard test split (already viewed) | dense | 73.3 % | 8.6 % |

The rule was fixed on the calibration sets before the held-out set was read; its access counter is
1 and the set must not be used for tuning again. The false-refusal target is met; the correct-refusal
target is not, and the reranker only works when it is switched on (about 2.4 GiB at peak). Probable
causes, **not isolated**: only 21 unanswerable calibration questions; thresholds placed at the edge of the
allowed false-refusal rate; held-out unanswerable questions built as near misses of real topics,
which no score of topical similarity separates; labels unreviewed, so some may be wrong. A decision
needs reviewed labels and a new untouched held-out set; until then the criterion is open and not
weakened. The existing thresholds still apply only to an index that someone has calibrated (F-1).

### L-1 Real language model: NOT VERIFIED

No chat model ran through the pipeline. The memory gate refused `qwen3.5:4b` (needs 4.1 GiB, 3.23 GiB
available, no free swap); it was not weakened and no process was stopped. The measurement is
prepared and reproducible: [LLM_VERIFICATION.md](LLM_VERIFICATION.md).

### H-1 Human review and judge calibration: NOT VERIFIED

All 321 questions (141 hard, 44 mini, 136 held-out) are drafts written with an AI language model.
None is reviewed, so every retrieval and generation number is exploratory, the 100-reviewed-questions
criterion is unmet, and no judge is calibrated. The review kit is in [DATASET_REVIEW.md](DATASET_REVIEW.md).

## Phases 0–11

| Phase | Verdict | Notes |
|---|---|---|
| 0 Foundations | PASS | layout, tooling, CI, 11 import contracts |
| 1 Domain and storage | PASS | models, migrations, repository, integrity, chaos tests |
| 2 Sources and change detection | PASS | NFR-04 |
| 3 Parsing and chunking | PASS | budgets 14 of 14, isolated worker |
| 4 Embeddings | PASS | E5 ONNX default; measurements in EMBEDDING_BAKEOFF.md |
| 5 Retrieval | PASS | hybrid by default (ADR 0013); NFR-01, 03 |
| 6 Reranking | PASS | optional, off by default; gain on dev, memory-guarded |
| 7 Generation | PASS with stub; real model NOT VERIFIED | |
| 8 Pipeline | PASS | gate quality FAIL (G-1) |
| 9 Evaluation | PASS | protocol; labels NOT VERIFIED |
| 10 API | PASS | D-1 fixed |
| 11 Operations and release | PASS for packaging; release itself is the owner's | licence, version, tag, publication not chosen |

## Public hygiene and licences

Tracked files, comments, documents, history subjects and both pull-request texts were scanned for
internal-research, other-project and preparation-history wording; findings (D-3) are fixed, history
is unchanged. Attribution and provenance are kept ([PROVENANCE.md](../PROVENANCE.md)). Dependencies:
54 components, none under a forbidden licence (MPL-2.0 components are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)); SBOM is regenerated and checked in CI. The project's
own licence is not chosen and so none is granted.
