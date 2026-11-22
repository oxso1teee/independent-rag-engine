"""The ``rag`` command: argument parsing, calling services, rendering results.

No business rule lives here. Commands turn arguments into service calls, services come from
a :class:`~rag_engine.cli.container.Container` (dependency injection), and every expected
failure becomes a stable exit code (see :mod:`rag_engine.cli.exit_codes`).

With ``--json`` a command prints exactly one JSON document to stdout, errors included
(``{"error": {...}}``); in text mode errors go to stderr.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import typer

from rag_engine import __version__
from rag_engine.cli.context import Container, Processing, Services
from rag_engine.cli.exit_codes import ExitCode, exit_code_for
from rag_engine.cli.render import chunks_to_dict, describe_chunks
from rag_engine.domain import ErrorCode, RagError, SourceKey, SourcePolicy
from rag_engine.ingestion.identity import RebasePlan
from rag_engine.ingestion.processing import read_limited
from rag_engine.ingestion.service import IngestReport, StoreStatus, collect_status
from rag_engine.ports import ParseLimits
from rag_engine.storage.sqlite import StoreNotInitialised

JsonOption = Annotated[bool, typer.Option("--json", help="Print one JSON document.")]


@dataclass(frozen=True, slots=True)
class _State:
    container: Container
    config: Path | None
    db: Path | None


def _emit(data: object) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _fail(error: RagError, *, as_json: bool) -> typer.Exit:
    if as_json:
        _emit({"error": error.to_dict()})
    else:
        typer.echo(f"error [{error.code.value}]: {error.message}", err=True)
        for key, value in error.context.items():
            typer.echo(f"  {key}: {value}", err=True)
    return typer.Exit(int(exit_code_for(error)))


def _state(ctx: typer.Context) -> _State:
    state = ctx.obj
    if not isinstance(state, _State):  # pragma: no cover - set by the root callback
        raise RuntimeError("the command line was not initialised")
    return state


def _execute[T](
    ctx: typer.Context,
    action: Callable[[Services], T],
    *,
    read_only: bool,
    as_json: bool,
    if_missing: T | None = None,
) -> T:
    """Open the services, run ``action`` and convert engine errors into exit codes."""
    state = _state(ctx)
    try:
        with state.container.open(state.config, state.db, read_only=read_only) as services:
            return action(services)
    except StoreNotInitialised:
        if if_missing is not None:
            return if_missing
        raise _fail(
            RagError(
                ErrorCode.E_SOURCE_NOT_FOUND,
                "no database yet; add a source first (rag source add)",
            ),
            as_json=as_json,
        ) from None
    except RagError as error:
        raise _fail(error, as_json=as_json) from error


def _version(value: bool) -> None:
    if value:
        typer.echo(f"rag {__version__}")
        raise typer.Exit(0)


def build_app(container: Container) -> typer.Typer:
    app = typer.Typer(
        name="rag",
        help="Modular, local-first RAG engine: manage document sources and ingestion.",
        add_completion=False,
        pretty_exceptions_enable=False,
        rich_markup_mode=None,
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    sources = typer.Typer(help="Register and maintain document sources.", rich_markup_mode=None)
    app.add_typer(sources, name="source")

    @app.callback()
    def root(
        ctx: typer.Context,
        config: Annotated[
            Path | None, typer.Option("--config", "-c", help="Path to a rag.toml file.")
        ] = None,
        db: Annotated[
            Path | None, typer.Option("--db", help="Database file (overrides the configuration).")
        ] = None,
        version: Annotated[
            bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
        ] = False,
    ) -> None:
        del version
        ctx.obj = _State(container, config, db)
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())

    # ------------------------------------------------------------------ source add

    @sources.command("add")
    def source_add(
        ctx: typer.Context,
        name: Annotated[str, typer.Argument(help="Short name: lowercase letters, digits, '-'.")],
        root: Annotated[Path, typer.Argument(help="Directory that contains the documents.")],
        include: Annotated[
            list[str] | None, typer.Option("--include", "-i", help="Glob to include (repeatable).")
        ] = None,
        exclude: Annotated[
            list[str] | None, typer.Option("--exclude", "-x", help="Glob to exclude (repeatable).")
        ] = None,
        follow_symlinks: Annotated[
            bool, typer.Option("--follow-symlinks", help="Follow links that stay inside the root.")
        ] = False,
        max_file_mb: Annotated[
            float, typer.Option("--max-file-mb", min=0.001, help="Skip files larger than this.")
        ] = 50.0,
        grace_days: Annotated[
            int,
            typer.Option("--grace-days", min=0, help="Days a vanished document stays restorable."),
        ] = 30,
        cleaner: Annotated[
            list[str] | None,
            typer.Option(
                "--cleaner", "-C", help="Regex of whole lines to drop from every text (repeatable)."
            ),
        ] = None,
        json_output: JsonOption = False,
    ) -> None:
        """Register a directory as a document source (idempotent for the same name and root)."""
        try:
            policy = SourcePolicy(
                include=tuple(include) if include else ("**/*",),
                exclude=tuple(exclude or ()),
                max_file_bytes=int(max_file_mb * 1024 * 1024),
                cleaners=tuple(cleaner or ()),
                follow_symlinks=follow_symlinks,
                delete_grace_days=grace_days,
            )
        except RagError as error:
            raise _fail(error, as_json=json_output) from error
        source = _execute(
            ctx,
            lambda s: s.registry.add(name, root, policy),
            read_only=False,
            as_json=json_output,
        )
        if json_output:
            _emit(
                {"source": {"name": source.name, "id": str(source.id), "locator": source.locator}}
            )
        else:
            typer.echo(f"Source '{source.name}' -> {source.locator}")

    # ----------------------------------------------------------------- source list

    @sources.command("list")
    def source_list(ctx: typer.Context, json_output: JsonOption = False) -> None:
        """List registered sources."""
        found: list[tuple[str, str, int]] = _execute(
            ctx,
            lambda s: [
                (x.name, x.locator or "", x.policy.delete_grace_days) for x in s.registry.list()
            ],
            read_only=True,
            as_json=json_output,
            if_missing=[],
        )
        if json_output:
            _emit(
                {
                    "sources": [
                        {"name": n, "locator": loc, "delete_grace_days": days}
                        for n, loc, days in found
                    ]
                }
            )
        elif not found:
            typer.echo("No sources registered.")
        else:
            width = max(len(n) for n, _, _ in found)
            for name, locator, _ in found:
                typer.echo(f"{name.ljust(width)}  {locator}")

    # --------------------------------------------------------------- source rebase

    @sources.command("rebase")
    def source_rebase(
        ctx: typer.Context,
        name: Annotated[str, typer.Argument(help="Registered source name.")],
        new_root: Annotated[Path, typer.Argument(help="The directory the source moved to.")],
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Show what would be kept and re-attached.")
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Move a source to a new root, keeping document ids wherever the match is provable."""
        plan: RebasePlan = _execute(
            ctx,
            lambda s: s.registry.rebase(name, new_root, apply=not dry_run),
            read_only=dry_run,
            as_json=json_output,
        )
        if json_output:
            _emit({"dry_run": dry_run, "rebase": plan.to_dict()})
            return
        prefix = "Dry run: " if dry_run else ""
        if plan.unchanged_locator:
            typer.echo(f"{prefix}'{name}' already points to this root; nothing to do.")
            return
        typer.echo(
            f"{prefix}'{name}': {plan.kept} kept, {len(plan.moves)} re-attached by content, "
            f"{len(plan.unmatched_old)} existing not found under the new root, "
            f"{len(plan.unmatched_new)} new, {plan.ambiguous} ambiguous (not merged)."
        )

    # ---------------------------------------------------------------------- ingest

    @app.command("ingest")
    def ingest(
        ctx: typer.Context,
        source: Annotated[
            str | None, typer.Argument(help="Source name; all sources when omitted.")
        ] = None,
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Report what would change; write nothing.")
        ] = False,
        accept_mass_delete: Annotated[
            bool,
            typer.Option(
                "--accept-mass-delete", help="Confirm a disappearance the delete guard stopped."
            ),
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Scan sources, record changes and queue the documents that need processing."""

        def work(services: Services) -> tuple[list[IngestReport], list[RagError]]:
            names = [source] if source else [s.name for s in services.registry.list()]
            reports: list[IngestReport] = []
            errors: list[RagError] = []
            for name in names:
                try:
                    run = services.ingest.plan if dry_run else services.ingest.run
                    reports.append(run(name, accept_mass_delete=accept_mass_delete))
                except RagError as error:
                    if source:
                        raise
                    errors.append(error)
            return reports, errors

        reports, errors = _execute(ctx, work, read_only=dry_run, as_json=json_output)
        if json_output:
            payload: dict[str, object] = {
                "dry_run": dry_run,
                "results": [r.to_dict() for r in reports],
            }
            if errors:
                payload["errors"] = [e.to_dict() for e in errors]
            _emit(payload)
        else:
            if not reports and not errors:
                typer.echo("No sources registered.")
            for report in reports:
                typer.echo(_describe(report))
            for error in errors:
                typer.echo(f"error [{error.code.value}]: {error.message}", err=True)
        if errors:
            raise typer.Exit(int(exit_code_for(errors[0])))

    # ---------------------------------------------------------------------- chunks

    @app.command("chunks")
    def chunks(
        ctx: typer.Context,
        doc: Annotated[
            str, typer.Argument(help="A file; with --source, the key of a document in that source.")
        ],
        source: Annotated[
            str | None,
            typer.Option("--source", "-s", help="Registered source; its cleaners are applied."),
        ] = None,
        cleaner: Annotated[
            list[str] | None,
            typer.Option("--cleaner", "-C", help="Extra regex of lines to drop (repeatable)."),
        ] = None,
        strategy: Annotated[
            str | None,
            typer.Option(
                "--strategy", help="auto, structure_aware, recursive or fixed (default: config)."
            ),
        ] = None,
        full: Annotated[bool, typer.Option("--full", help="Print whole chunk texts.")] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Parse a document and show the chunks it is split into; nothing is stored."""
        state = _state(ctx)
        extra = tuple(cleaner or ())
        try:
            processing = state.container.processing(state.config)
            if source is None:
                path = Path(doc)
                name, patterns = unicodedata.normalize("NFC", path.name), extra
                limits = processing.limits
                data = _read_file(path, limits.max_input_bytes)
            else:
                name, found, limits, data = _execute(
                    ctx,
                    lambda s: _read_entry(s, source, doc, processing),
                    read_only=True,
                    as_json=json_output,
                )
                patterns = (*found, *extra)
            done = processing.build(patterns, strategy).process(
                data,
                SourceKey(name),
                limits,
                processing.policy,
                processing.ids.document_version_id(),
            )
        except RagError as error:
            raise _fail(error, as_json=json_output) from error
        if json_output:
            _emit(chunks_to_dict(done, processing.policy, name))
        else:
            typer.echo("\n".join(describe_chunks(done, name, full=full)))

    # ---------------------------------------------------------------------- status

    @app.command("status")
    def status(ctx: typer.Context, json_output: JsonOption = False) -> None:
        """Show sources, document states and the job queue."""
        empty = StoreStatus(sources=(), jobs={}, schema_version=None, index_generation=None)
        result = _execute(
            ctx,
            lambda s: collect_status(s.store.reader()),
            read_only=True,
            as_json=json_output,
            if_missing=empty,
        )
        if json_output:
            _emit(result.to_dict())
            return
        if result.schema_version is None:
            typer.echo("No database yet.")
            return
        typer.echo(f"schema v{result.schema_version}, index generation {result.index_generation}")
        if not result.sources:
            typer.echo("No sources registered.")
        for s in result.sources:
            docs = ", ".join(f"{n} {k}" for k, n in s.documents.items()) or "no documents"
            if s.awaiting_first_ingest:
                docs += f" ({s.awaiting_first_ingest} awaiting first ingestion)"
            typer.echo(f"{s.name}: {docs}")
        jobs = ", ".join(f"{n} {k}" for k, n in result.jobs.items()) or "empty"
        typer.echo(f"jobs: {jobs}")

    return app


def _key(value: str) -> SourceKey:
    try:
        return SourceKey(value)
    except RagError:
        raise RagError(
            ErrorCode.E_SOURCE_NOT_FOUND,
            "not a valid document key (relative, '/'-separated, no '..')",
            context={"key": value},
        ) from None


def _read_file(path: Path, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as stream:
            return read_limited(stream, max_bytes)
    except OSError:
        raise RagError(
            ErrorCode.E_SOURCE_NOT_FOUND,
            "the file does not exist or cannot be read",
            context={"path": str(path)},
        ) from None


def _read_entry(
    services: Services, name: str, key: str, processing: Processing
) -> tuple[str, tuple[str, ...], ParseLimits, bytes]:
    source, stream = services.registry.open(name, _key(key))
    limits = replace(processing.limits, max_input_bytes=source.policy.max_file_bytes)
    with stream:
        data = read_limited(stream, limits.max_input_bytes)
    return key, source.policy.cleaners, limits, data


def _describe(report: IngestReport) -> str:
    verb = "would" if report.dry_run else "did"
    parts = [
        f"{report.new} new",
        f"{report.modified} modified",
        f"{report.unchanged} unchanged",
        f"{report.deleted} missing",
        f"{report.moved} moved",
    ]
    for label, value in (
        ("restored", report.restored),
        ("inaccessible", report.inaccessible),
        ("unsupported", report.unsupported),
        ("skipped", report.skipped),
        ("collected", report.gc_removed),
        ("duplicate link(s) updated", report.duplicates),
    ):
        if value:
            parts.append(f"{value} {label}")
    suffix = "" if report.dry_run else f"; {report.jobs_enqueued} job(s) queued"
    return f"{report.source} ({verb}): " + ", ".join(parts) + suffix


def main(argv: Sequence[str] | None, *, container: Container) -> int:
    """Entry point; returns the process exit code."""
    app = build_app(container)
    try:
        app(args=None if argv is None else list(argv), prog_name="rag")
    except SystemExit as exit_:  # the parser reports --help, --version and usage errors this way
        code = exit_.code
        if code is None:
            return int(ExitCode.OK)
        return code if isinstance(code, int) else int(ExitCode.FAILURE)
    except RagError as error:
        typer.echo(f"error [{error.code.value}]: {error.message}", err=True)
        return int(exit_code_for(error))
    except Exception as exc:
        typer.echo(f"error: unexpected failure ({type(exc).__name__})", err=True)
        return int(ExitCode.FAILURE)
    return int(ExitCode.OK)
