"""``rag calibrate``: fit, show or clear the sufficiency gate's calibration (ADR 0021)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Protocol

import typer

from rag_engine import __version__
from rag_engine.cli.context import Container
from rag_engine.cli.exit_codes import ExitCode
from rag_engine.domain import DocumentStatus, ErrorCode, RagError
from rag_engine.evaluation.gate_calibration import (
    CalibrationResult,
    Observation,
    QuestionSet,
    clear_calibration,
    describe_calibration,
    fit_calibration,
    load_question_set,
    observe,
    overlap,
    save_calibration,
)
from rag_engine.ports import MetadataReader
from rag_engine.retrieval.sufficiency import TARGET_CORRECT_REFUSAL, TARGET_FALSE_REFUSAL
from rag_engine.storage.sqlite import StoreNotInitialised
from rag_engine.system import SystemClock

JsonOption = Annotated[bool, typer.Option("--json", help="Print one JSON document.")]


class _State(Protocol):
    @property
    def container(self) -> Container: ...
    @property
    def config(self) -> Path | None: ...
    @property
    def db(self) -> Path | None: ...


def _documents(reader: MetadataReader) -> int:
    return sum(reader.count_documents(s.id, DocumentStatus.ACTIVE) for s in reader.list_sources())


def _rates_line(label: str, rates: Mapping[str, object]) -> str:
    return (
        f"  {label}: correct refusals {float(str(rates['correct_refusal'])):.1%} "
        f"({rates['unanswerable']} unanswerable), false refusals "
        f"{float(str(rates['false_refusal'])):.1%} ({rates['answerable']} answerable)"
    )


def _result_dict(result: CalibrationResult, *, written: bool) -> dict[str, object]:
    thresholds = result.thresholds
    return {
        "written": written,
        "signal": result.signal,
        "threshold": thresholds.rerank if thresholds.rerank is not None else thresholds.dense,
        "meets_target": result.meets_target,
        "anecdotal": result.anecdotal,
        "exploratory": result.exploratory,
        "estimate": result.estimate.as_context(),
        "in_sample": result.in_sample.as_context(),
        "holdout": None if result.holdout is None else result.holdout.as_context(),
        "warnings": list(result.warnings),
    }


def register_calibrate(
    app: typer.Typer,
    *,
    state_of: Callable[[typer.Context], _State],
    fail: Callable[..., typer.Exit],
    emit: Callable[[object], None],
) -> None:
    group = typer.Typer(help="Calibrate the sufficiency gate on this index.", no_args_is_help=True)
    app.add_typer(group, name="calibrate")

    @group.command("run")
    def run(
        ctx: typer.Context,
        questions: Annotated[
            Path, typer.Argument(help="An evaluation dataset or JSON Lines of question/answerable.")
        ],
        holdout: Annotated[
            Path | None,
            typer.Option(
                "--holdout", help="A second file, read once, that took no part in the fit."
            ),
        ] = None,
        max_false_refusal: Annotated[
            float, typer.Option("--max-false-refusal", min=0.0, max=0.99)
        ] = TARGET_FALSE_REFUSAL,
        folds: Annotated[int, typer.Option("--folds", min=2, max=20)] = 5,
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Compute and print; write nothing.")
        ] = False,
        require_target: Annotated[
            bool,
            typer.Option(
                "--require-target",
                help="Write nothing and exit 12 if the estimate misses the targets.",
            ),
        ] = False,
        include_unreviewed: Annotated[
            bool, typer.Option("--include-unreviewed", help="Use questions no person accepted.")
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Fit the gate's threshold on questions with and without an answer.

        The result is an estimate (cross-validated), not a promise; it is written even when it is
        below the targets, and says so.
        """
        state = state_of(ctx)
        try:
            fit_set = load_question_set(questions, include_unreviewed=include_unreviewed)
            held_set: QuestionSet | None = None
            if holdout is not None:
                held_set = load_question_set(holdout, include_unreviewed=include_unreviewed)
                shared = overlap(fit_set, held_set)
                if shared:
                    raise RagError(
                        ErrorCode.E_QUERY_INVALID,
                        f"{shared} holdout question(s) also appear in the calibration set; "
                        "a holdout must be independent",
                    )
            with state.container.search(state.config, state.db) as searcher:
                reader = searcher.store.reader()
                metadata = reader.get_index_metadata()
                if metadata is None or metadata.active_embedding_model_id is None:
                    raise RagError(
                        ErrorCode.E_SOURCE_NOT_FOUND, "no published index yet; run rag ingest first"
                    )
                model_id = str(metadata.active_embedding_model_id)
                generation = metadata.index_generation
                documents = _documents(reader)
                mode, depth, candidates = (
                    searcher.defaults.mode,
                    searcher.defaults.top_k,
                    searcher.defaults.candidate_k,
                )

                async def measure() -> tuple[list[Observation], list[Observation] | None]:
                    fit_obs = await observe(
                        searcher.retrieval, searcher.ids, fit_set.questions,
                        mode=mode, top_k=depth, candidate_k=candidates,
                    )  # fmt: skip
                    held_obs = (
                        None
                        if held_set is None
                        else await observe(
                            searcher.retrieval, searcher.ids, held_set.questions,
                            mode=mode, top_k=depth, candidate_k=candidates,
                        )
                    )  # fmt: skip
                    return fit_obs, held_obs

                fit_obs, held_obs = asyncio.run(measure())
                environment = searcher.gate_environment()
                searcher.release_models()
            result = fit_calibration(
                fit_obs,
                model_id=model_id,
                environment=environment,
                dataset=fit_set.identifier,
                exploratory=fit_set.exploratory or bool(held_set and held_set.exploratory),
                created_at=SystemClock().now().isoformat(),
                engine_version=__version__,
                index_generation=generation,
                documents=documents,
                max_false=max_false_refusal,
                folds=folds,
                holdout=held_obs,
                holdout_dataset=None if held_set is None else held_set.identifier,
            )
            write = not dry_run and not (require_target and not result.meets_target)
            if write:
                with state.container.open(state.config, state.db, read_only=False) as services:
                    save_calibration(
                        services.store, result, model_id=model_id, index_generation=generation
                    )
        except StoreNotInitialised:
            raise fail(
                RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no database yet; ingest documents first"),
                as_json=json_output,
            ) from None
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        _print_run(result, written=write, json_output=json_output, emit=emit)
        if require_target and not result.meets_target:
            raise typer.Exit(int(ExitCode.REGRESSION))

    @group.command("show")
    def show(ctx: typer.Context, json_output: JsonOption = False) -> None:
        """Show the stored calibration and whether it still applies."""
        state = state_of(ctx)
        try:
            with state.container.search(state.config, state.db) as searcher:
                reader = searcher.store.reader()
                metadata = reader.get_index_metadata()
                stored = {} if metadata is None else metadata.calibration
                active = (
                    None
                    if metadata is None or metadata.active_embedding_model_id is None
                    else str(metadata.active_embedding_model_id)
                )
                info = describe_calibration(
                    stored, active_model=active, documents=_documents(reader)
                )
        except StoreNotInitialised:
            info = {"state": "uncalibrated", "problems": ["there is no database yet"]}
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit(info)
            return
        typer.echo(f"gate calibration: {info['state']}")
        problems = info["problems"]
        if isinstance(problems, list):
            for problem in problems:
                typer.echo(f"  - {problem}")
        details = info.get("details")
        if isinstance(details, dict):
            typer.echo(f"  signal {details.get('signal')}, threshold {info.get('threshold')}")
            estimate = details.get("estimate")
            if isinstance(estimate, dict):
                typer.echo(_rates_line("estimate", estimate))
        if info["state"] != "calibrated":
            typer.echo("run: rag calibrate run QUESTIONS")

    @group.command("clear")
    def clear(ctx: typer.Context, json_output: JsonOption = False) -> None:
        """Remove the stored calibration; the gate then lets every question through."""
        state = state_of(ctx)
        try:
            with state.container.open(state.config, state.db, read_only=False) as services:
                removed = clear_calibration(services.store)
        except StoreNotInitialised:
            removed = False
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"cleared": removed})
        else:
            typer.echo("calibration removed" if removed else "there was no calibration")


def _print_run(
    result: CalibrationResult, *, written: bool, json_output: bool, emit: Callable[[object], None]
) -> None:
    if json_output:
        emit(_result_dict(result, written=written))
        return
    threshold = result.thresholds.rerank
    if threshold is None:
        threshold = result.thresholds.dense
    typer.echo(
        f"signal {result.signal}, threshold {threshold:.4f}" if threshold is not None else ""
    )
    typer.echo(_rates_line("cross-validated estimate", result.estimate.as_context()))
    typer.echo(
        _rates_line("on the calibration questions (optimistic)", result.in_sample.as_context())
    )
    if result.holdout is not None:
        typer.echo(_rates_line("separate holdout", result.holdout.as_context()))
    typer.echo(
        f"targets: correct refusals at least {TARGET_CORRECT_REFUSAL:.0%}, "
        f"false refusals at most {TARGET_FALSE_REFUSAL:.0%}"
    )
    for note in result.warnings:
        typer.echo(f"warning: {note}")
    typer.echo("calibration written" if written else "nothing written")
