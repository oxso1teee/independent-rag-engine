"""``rag ask``: question in, a cited answer (or an honest refusal) out."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer

from rag_engine.cli.ask_render import answer_to_dict, describe_answer, event_to_dict
from rag_engine.cli.context import AskServices, Container
from rag_engine.cli.exit_codes import ExitCode, exit_code_for_error_code
from rag_engine.cli.search import parse_filters
from rag_engine.domain import (
    AskStatus,
    ErrorCode,
    RagError,
    RerankMode,
    RetrievalMode,
    RetrievalQuery,
)
from rag_engine.generation.generator import DeltaEvent, RetractEvent
from rag_engine.pipeline import FinalEvent, PipelineAnswer
from rag_engine.storage.sqlite import StoreNotInitialised


class _State(Protocol):
    @property
    def container(self) -> Container: ...
    @property
    def config(self) -> Path | None: ...
    @property
    def db(self) -> Path | None: ...


def exit_code_for_answer(answer: PipelineAnswer) -> ExitCode:
    match answer.status:
        case AskStatus.ANSWERED:
            return ExitCode.OK
        case AskStatus.INSUFFICIENT_EVIDENCE:
            return ExitCode.INSUFFICIENT
        case AskStatus.UNVERIFIED:
            return ExitCode.UNVERIFIED
        case AskStatus.ERROR:
            assert answer.error is not None  # noqa: S101 - an error result always has one
            return exit_code_for_error_code(answer.error.code)


def register_ask(
    app: typer.Typer,
    *,
    state_of: Callable[[typer.Context], _State],
    fail: Callable[..., typer.Exit],
    emit: Callable[[object], None],
) -> None:
    @app.command("ask")
    def ask_command(
        ctx: typer.Context,
        question: Annotated[str, typer.Argument(help="What to ask the documents.")],
        mode: Annotated[
            str | None, typer.Option("--mode", "-m", help="dense, lexical or hybrid.")
        ] = None,
        top_k: Annotated[int | None, typer.Option("--top-k", "-k", min=1, max=100)] = None,
        source: Annotated[list[str] | None, typer.Option("--source", "-s")] = None,
        path_prefix: Annotated[str | None, typer.Option("--path-prefix")] = None,
        path_glob: Annotated[str | None, typer.Option("--path-glob")] = None,
        rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
        stream: Annotated[
            bool, typer.Option("--stream", help="Print the answer as it is produced.")
        ] = False,
        live: Annotated[
            bool,
            typer.Option(
                "--live",
                help="With --stream: show the draft while it is written, before its citations "
                "are checked (it is withdrawn if they fail). Default: only the checked text.",
            ),
        ] = False,
        sources: Annotated[
            bool, typer.Option("--sources", help="Also list the evidence the model saw.")
        ] = False,
        show_unverified: Annotated[
            bool,
            typer.Option(
                "--show-unverified", help="Show the text of an unverified answer (do not trust it)."
            ),
        ] = False,
        json_output: Annotated[
            bool,
            typer.Option("--json", help="One JSON document (with --stream: one event per line)."),
        ] = False,
    ) -> None:
        """Answer a question from the indexed documents, with sources, or say they do not know."""
        state = state_of(ctx)
        try:
            if not question.strip():
                raise RagError(ErrorCode.E_QUERY_INVALID, "the question is empty")
            if live and not stream:
                raise RagError(ErrorCode.E_QUERY_INVALID, "--live needs --stream")
            filters = parse_filters(
                sources=source or (),
                path_prefix=path_prefix,
                path_glob=path_glob,
                tags_any=(),
                tags_all=(),
                language=None,
                kinds=(),
                modified_after=None,
                modified_before=None,
                doc_ids=(),
                exclude_doc_ids=(),
            )
            try:
                chosen = None if mode is None else RetrievalMode(mode)
            except ValueError:
                raise RagError(
                    ErrorCode.E_QUERY_INVALID, "--mode must be dense, lexical or hybrid"
                ) from None
            with state.container.ask(state.config, state.db) as services:
                depth = top_k or services.defaults.top_k
                query = RetrievalQuery(
                    request_id=services.ids.request_id(),
                    text=question,
                    mode=chosen or services.defaults.mode,
                    top_k=depth,
                    candidate_k=max(services.defaults.candidate_k, depth),
                    filters=filters,
                    rerank=(
                        RerankMode.AUTO
                        if rerank is None
                        else RerankMode.ON
                        if rerank
                        else RerankMode.OFF
                    ),
                )
                answer = _run(services, query, stream, live, json_output, sources, show_unverified)
        except StoreNotInitialised:
            raise fail(
                RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no database yet; add a source first"),
                as_json=json_output,
            ) from None
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if not stream:
            if json_output:
                emit(answer_to_dict(answer, sources=sources, show_unverified=show_unverified))
            else:
                text = describe_answer(answer, sources=sources, show_unverified=show_unverified)
                typer.echo("\n".join(text), err=answer.status is AskStatus.ERROR)
        code = exit_code_for_answer(answer)
        if code is not ExitCode.OK:
            raise typer.Exit(int(code))


def _run(
    services: AskServices,
    query: RetrievalQuery,
    stream: bool,
    live: bool,
    json_output: bool,
    sources: bool,
    show_unverified: bool,
) -> PipelineAnswer:
    if not stream:
        return asyncio.run(services.pipeline.ask(query))

    async def consume() -> PipelineAnswer:
        final: PipelineAnswer | None = None
        async for event in services.pipeline.stream(query, verified_only=not live):
            if json_output:
                sys.stdout.write(json.dumps(event_to_dict(event), ensure_ascii=False) + "\n")
                sys.stdout.flush()
            elif isinstance(event, DeltaEvent):
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif isinstance(event, RetractEvent):
                sys.stderr.write(f"\n[withdrawn: {event.reason}]\n")
            if isinstance(event, FinalEvent):
                final = event.answer
        assert final is not None  # noqa: S101 - a stream always ends with its final event
        return final

    answer = asyncio.run(consume())
    if not json_output:
        text = describe_answer(answer, sources=sources, show_unverified=show_unverified)
        # The text of an answer was already streamed; print what follows it.
        body = text[1:] if answer.status is AskStatus.ANSWERED else text
        typer.echo("\n" + "\n".join(body))
    return answer
