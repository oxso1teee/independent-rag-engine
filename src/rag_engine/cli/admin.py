"""``rag init``, ``rag config``, ``rag backup`` and ``rag serve``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer

from rag_engine.cli.context import Container
from rag_engine.cli.exit_codes import ExitCode
from rag_engine.config import load_settings
from rag_engine.domain import ErrorCode, RagError
from rag_engine.storage.backup import backup_database, restore_backup, verify_backup

STARTER_CONFIG = """\
# rag.toml: settings of the Independent RAG Engine. Every value is optional; environment variables
# prefixed with RAG_ override this file (for example RAG_LOGGING__LEVEL=DEBUG). Secrets never go
# here: name the environment variable that holds them (api_key_env, api.token_env).
# See rag.example.toml in the repository for every setting.

[storage]
db_path = "data/rag.db"

# Search needs an embedding model. Pin the installed model's digest, dimension and input limit.
# [embedding]
# enabled = true
# provider = "onnx"            # or "ollama"
# model = "intfloat/multilingual-e5-small"
# revision = "<digest of the model directory>"
# dimension = 384
# max_input_tokens = 512
# model_dir = "data/models/multilingual-e5-small/<revision>"
# query_prefix = "query: "
# document_prefix = "passage: "

# Answers need a language model and its context window.
# [llm]
# enabled = true
# provider = "ollama"
# model = "qwen3.5:4b"
# context_window = 4096
# max_new_tokens = 512
# keep_alive = "0"             # unload the model after every answer (the safe default here)

# The HTTP API (`rag serve`). Listening beyond the loopback address requires a token.
# [api]
# host = "127.0.0.1"
# port = 8765
# token_env = "RAG_API_TOKEN"
"""


class _State(Protocol):
    @property
    def container(self) -> Container: ...
    @property
    def config(self) -> Path | None: ...
    @property
    def db(self) -> Path | None: ...


def register_admin(
    app: typer.Typer,
    *,
    state_of: Callable[[typer.Context], _State],
    fail: Callable[..., typer.Exit],
    emit: Callable[[object], None],
) -> None:
    @app.command("init")
    def init(
        directory: Annotated[Path, typer.Option("--dir", help="Where to create rag.toml.")] = Path(
            "."
        ),
        force: Annotated[
            bool, typer.Option("--force", help="Overwrite an existing rag.toml.")
        ] = False,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Create a starter rag.toml and the data directory."""
        target = directory / "rag.toml"
        try:
            if target.exists() and not force:
                raise RagError(
                    ErrorCode.E_CONFIG_INVALID,
                    "rag.toml already exists; use --force to overwrite it",
                    context={"path": str(target)},
                )
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "data").mkdir(exist_ok=True)
            target.write_text(STARTER_CONFIG, encoding="utf-8")
        except OSError as error:
            raise fail(
                RagError(
                    ErrorCode.E_CONFIG_INVALID, f"cannot write {target}: {type(error).__name__}"
                ),
                as_json=json_output,
            ) from error
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"created": str(target)})
        else:
            typer.echo(f"created {target}")
            typer.echo("next: edit it, then `rag source add NAME DIR` and `rag ingest NAME`")

    config = typer.Typer(help="Inspect the configuration.", rich_markup_mode=None)
    app.add_typer(config, name="config")

    @config.command("show")
    def config_show(
        ctx: typer.Context, json_output: Annotated[bool, typer.Option("--json")] = False
    ) -> None:
        """Print the effective settings (file, environment and defaults merged)."""
        try:
            settings = load_settings(state_of(ctx).config)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        data = settings.model_dump(mode="json")
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))

    @config.command("check")
    def config_check(
        ctx: typer.Context, json_output: Annotated[bool, typer.Option("--json")] = False
    ) -> None:
        """Validate the configuration; exit code 3 when it is invalid."""
        try:
            settings = load_settings(state_of(ctx).config)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        notes = [
            name
            for name, enabled in (
                ("embedding", settings.embedding.enabled),
                ("llm", settings.llm.enabled),
                ("rerank", settings.rerank.enabled),
            )
            if enabled
        ]
        if json_output:
            emit({"valid": True, "enabled": notes})
        else:
            typer.echo("configuration is valid; enabled: " + (", ".join(notes) or "nothing yet"))

    @app.command("backup")
    def backup(
        ctx: typer.Context,
        destination: Annotated[Path, typer.Argument(help="New backup file (must not exist).")],
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Write a verified, consistent copy of the database; safe while the server runs."""
        state = state_of(ctx)
        try:
            source = state.db or load_settings(state.config).storage.db_path
            made = backup_database(source, destination)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"backup": str(made), "bytes": made.stat().st_size})
        else:
            typer.echo(f"backup written to {made} ({made.stat().st_size} bytes)")

    @app.command("verify")
    def verify(
        backup_file: Annotated[Path, typer.Argument(help="A backup made by `rag backup`.")],
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Check a backup without opening it for writing; exit 9 when it is damaged."""
        try:
            report = verify_backup(backup_file)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        data = {
            "ok": report.ok, "errors": list(report.errors), "warnings": list(report.warnings),
            "chunks": report.published_chunks, "embeddings": report.embeddings,
            "index_generation": report.index_generation,
        }  # fmt: skip
        if json_output:
            emit(data)
        else:
            typer.echo(
                f"{'ok' if report.ok else 'DAMAGED'}: {report.published_chunks} chunks, "
                f"{report.embeddings} embeddings, index generation {report.index_generation}"
            )
            for message in (*report.errors, *report.warnings):
                typer.echo(f"  {message}")
        if not report.ok:
            raise typer.Exit(int(ExitCode.INTEGRITY))

    @app.command("restore")
    def restore(
        ctx: typer.Context,
        backup_file: Annotated[Path, typer.Argument(help="A backup made by `rag backup`.")],
        replace: Annotated[
            bool, typer.Option("--replace", help="Keep the current database aside and restore.")
        ] = False,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Restore a verified backup as the database; stop the server first."""
        state = state_of(ctx)
        try:
            destination = state.db or load_settings(state.config).storage.db_path
            kept = restore_backup(backup_file, destination, replace=replace)
        except RagError as error:
            raise fail(error, as_json=json_output) from error
        if json_output:
            emit({"restored": str(destination), "previous": None if kept is None else str(kept)})
        else:
            typer.echo(f"restored {destination}")
            if kept is not None:
                typer.echo(f"the previous database is kept as {kept}")
            typer.echo("next: `rag doctor --integrity` to check it")

    @app.command("serve")
    def serve(
        ctx: typer.Context,
        host: Annotated[str | None, typer.Option("--host", help="Address to listen on.")] = None,
        port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    ) -> None:
        """Run the HTTP API. Listening beyond the loopback address needs api.token_env."""
        state = state_of(ctx)
        try:
            code = state.container.serve(state.config, state.db, host=host, port=port)
        except RagError as error:
            raise fail(error, as_json=False) from error
        raise typer.Exit(code)
