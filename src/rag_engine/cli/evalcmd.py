"""``rag eval``: validate a dataset, run it, compare runs, report and gate against a baseline."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Protocol

import typer

from rag_engine import __version__
from rag_engine.cli.context import Container
from rag_engine.cli.exit_codes import ExitCode
from rag_engine.config import load_settings
from rag_engine.domain import (
    ErrorCode,
    RagError,
    RerankMode,
    RetrievalMode,
    RetrievalQuery,
)
from rag_engine.evaluation.dataset import (
    Dataset,
    Question,
    accepted_questions,
    load_dataset,
    validate_dataset,
)
from rag_engine.evaluation.protocol import (
    Manifest,
    TestSplitGuard,
    build_run,
    check_regression,
    compare_runs,
    load_run,
    now_iso,
    render_comparison,
    render_report,
    seed_for,
    sha256_text,
    write_run,
)
from rag_engine.evaluation.runner import METRICS, QuestionScore, RankedChunk, score_question
from rag_engine.retrieval.presentation import load_details
from rag_engine.storage.sqlite import StoreNotInitialised

ACCESS_FILE = "eval_runs/test_access.json"


class _State(Protocol):
    @property
    def container(self) -> Container: ...
    @property
    def config(self) -> Path | None: ...
    @property
    def db(self) -> Path | None: ...


def _git_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, timeout=5, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _corpus(dataset: Dataset, dataset_path: Path, override: Path | None) -> dict[str, str]:
    candidates = [override] if override else []
    candidates += [dataset_path.parent / dataset.corpus, dataset_path.parents[1] / dataset.corpus]
    candidates += [Path.cwd() / dataset.corpus, dataset_path.parents[2] / dataset.corpus]
    for folder in candidates:
        if folder is not None and folder.is_dir():
            return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}
    raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, f"the corpus {dataset.corpus} was not found")


def _corpus_hash(corpus: dict[str, str]) -> str:
    return sha256_text(json.dumps(sorted(corpus.items()), ensure_ascii=False))


def register_eval(
    app: typer.Typer,
    *,
    state_of: Callable[[typer.Context], _State],
    fail: Callable[..., typer.Exit],
    emit: Callable[[object], None],
) -> None:
    evaluation = typer.Typer(help="Measure retrieval and compare runs.", rich_markup_mode=None)
    app.add_typer(evaluation, name="eval")

    @evaluation.command("validate")
    def validate(
        dataset_path: Annotated[Path, typer.Argument(help="The dataset JSON file.")],
        corpus: Annotated[Path | None, typer.Option("--corpus")] = None,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Check a dataset against its corpus: schema, anchors, splits, reviewers."""
        try:
            dataset = load_dataset(dataset_path)
            report = validate_dataset(dataset, _corpus(dataset, dataset_path, corpus))
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        counted = len(accepted_questions(dataset))
        data = {
            "dataset": dataset.identifier, "ok": report.ok, "questions": len(dataset.questions),
            "reviewed": counted, "errors": list(report.errors), "warnings": list(report.warnings),
        }  # fmt: skip
        if json_output:
            emit(data)
        else:
            typer.echo(
                f"{dataset.identifier}: {len(dataset.questions)} questions, {counted} reviewed, "
                f"{len(report.errors)} errors, {len(report.warnings)} warnings"
            )
            for message in (*report.errors, *report.warnings):
                typer.echo(f"  {message}")
        if not report.ok:
            raise typer.Exit(int(ExitCode.INTEGRITY))

    @evaluation.command("run")
    def run(
        ctx: typer.Context,
        dataset_path: Annotated[Path, typer.Option("--dataset")],
        output: Annotated[Path, typer.Option("--output", "-o")],
        split: Annotated[str, typer.Option("--split")] = "dev",
        final: Annotated[
            bool, typer.Option("--final", help="Read the held-out test split.")
        ] = False,
        mode: Annotated[str, typer.Option("--mode", help="dense, lexical or hybrid.")] = "dense",
        top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 10,
        include_unreviewed: Annotated[bool, typer.Option("--include-unreviewed")] = False,
        corpus: Annotated[Path | None, typer.Option("--corpus")] = None,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Run the retrieval evaluation against the configured index and write the run file."""
        state = state_of(ctx)
        try:
            if split not in {"dev", "test"}:
                raise RagError(ErrorCode.E_QUERY_INVALID, "--split must be dev or test")
            try:
                retrieval_mode = RetrievalMode(mode)
            except ValueError:
                raise RagError(
                    ErrorCode.E_QUERY_INVALID, "--mode must be dense, lexical or hybrid"
                ) from None
            dataset = load_dataset(dataset_path)
            texts = _corpus(dataset, dataset_path, corpus)
            if not validate_dataset(dataset, texts).ok:
                raise RagError(ErrorCode.E_INTEGRITY, "the dataset does not validate")
            reads: int | None = None
            if split == "test":
                reads = TestSplitGuard(Path(ACCESS_FILE)).read(dataset.identifier, final=final)
            pool = (
                [q for q in dataset.questions if q.review.status != "rejected"]
                if include_unreviewed
                else accepted_questions(dataset)
            )
            questions = [q for q in pool if q.split == split and q.answerable]
            if not questions:
                raise RagError(
                    ErrorCode.E_QUERY_INVALID,
                    "no reviewed answerable questions in this split; review the dataset or "
                    "use --include-unreviewed (an exploratory run)",
                )
            settings = load_settings(state.config)
            with state.container.search(state.config, state.db) as searcher:
                scores = asyncio.run(_score_all(searcher, questions, retrieval_mode, top_k))
                metadata = searcher.store.reader().get_index_metadata()
                run_data = build_run(
                    Manifest(
                        mode="retrieval", split=split, final=final, dataset=dataset.identifier,
                        dataset_hash=dataset.manifest_hash, corpus_hash=_corpus_hash(texts),
                        config_hash=sha256_text(settings.model_dump_json()),
                        embedding_model=None if metadata is None
                        or metadata.active_embedding_model_id is None
                        else str(metadata.active_embedding_model_id),
                        llm=None, judge=None, context_window=None,
                        index_generation=None if metadata is None else metadata.index_generation,
                        seed=seed_for(dataset.manifest_hash), engine_version=__version__,
                        git_commit=_git_commit(), resources={}, exploratory=include_unreviewed,
                        created_at=now_iso(),
                    ),
                    scores,
                    METRICS,
                    test_reads=reads,
                )  # fmt: skip
            write_run(run_data, output)
        except StoreNotInitialised:
            raise fail(
                RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no index yet; ingest the corpus first"),
                as_json=json_output,
            ) from None
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"run": str(output), "questions": len(scores), "flags": run_data["flags"]})
        else:
            typer.echo(f"wrote {output} ({len(scores)} questions, {run_data['flags']['sample']})")

    @evaluation.command("report")
    def report(
        run_file: Annotated[Path, typer.Argument()],
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Print a run as Markdown (or as JSON)."""
        try:
            data = load_run(run_file)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"manifest": data["manifest"], "flags": data["flags"], "summary": data["summary"]})
        else:
            typer.echo(render_report(data), nl=False)

    @evaluation.command("compare")
    def compare(
        first: Annotated[Path, typer.Argument()],
        second: Annotated[Path, typer.Argument()],
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Paired comparison of two runs: first minus second."""
        try:
            rows = compare_runs(load_run(first), load_run(second))
        except (RagError, ValueError) as error:
            wrapped = (
                error
                if isinstance(error, RagError)
                else RagError(ErrorCode.E_QUERY_INVALID, str(error))
            )
            raise fail(wrapped, as_json=json_output) from error
        if json_output:
            emit([asdict(row) for row in rows])
        else:
            typer.echo(render_comparison(rows, first.name, second.name), nl=False)

    @evaluation.command("gate")
    def gate(
        run_file: Annotated[Path, typer.Argument()],
        baseline_file: Annotated[Path, typer.Argument()],
        delta: Annotated[float, typer.Option("--delta", min=0.0, max=1.0)] = 0.03,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Fail (exit 12) when a metric is worse than the baseline by more than --delta."""
        try:
            violations = check_regression(load_run(run_file), load_run(baseline_file), delta=delta)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"passed": not violations, "violations": [asdict(v) for v in violations]})
        else:
            typer.echo("no regression" if not violations else "REGRESSION")
            for violation in violations:
                typer.echo(f"  {violation.metric}: {violation.reason}")
        if violations:
            raise typer.Exit(int(ExitCode.REGRESSION))


async def _score_all(
    searcher: object, questions: list[Question], mode: RetrievalMode, top_k: int
) -> list[QuestionScore]:
    scores: list[QuestionScore] = []
    for question in questions:
        request = RetrievalQuery(
            request_id=searcher.ids.request_id(),  # type: ignore[attr-defined]
            text=question.question,
            mode=mode,
            top_k=top_k,
            candidate_k=max(40, top_k),
            rerank=RerankMode.OFF,
        )
        outcome = await searcher.retrieval.search(request)  # type: ignore[attr-defined]
        ranked = [
            RankedChunk(detail.path, detail.text)
            for detail in load_details(searcher.store, outcome.result)  # type: ignore[attr-defined]
        ]
        scores.append(score_question(question, ranked))
    return scores
