"""Ingestion emits structured stage events with a request id and never leaks document data."""

from __future__ import annotations

import io
import json

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.observability import configure_logging
from tests.workspace import Workspace

SECRET_TEXT = "launch codes are 0000-SECRET-BODY"
SECRET_NAME = "top-secret-plan"


def events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_a_run_logs_its_stages_with_one_request_id_and_only_numbers(workspace: Workspace) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    workspace.write(f"{SECRET_NAME}.md", SECRET_TEXT)
    workspace.write("other.md", "plain")
    workspace.run()

    recorded = events(stream)
    names = [e["event"] for e in recorded]
    assert names == [
        "stage.start",
        "stage.done",
        "stage.start",
        "stage.done",
        "stage.start",
        "stage.done",
        "ingest.done",
    ]
    assert [e.get("stage") for e in recorded if e["event"] == "stage.start"] == [
        "ingest.scan",
        "ingest.apply",
        "ingest.gc",
    ]
    assert len({e["request_id"] for e in recorded}) == 1  # one request id on every event
    summary = recorded[-1]
    assert summary["new"] == 2 and summary["jobs_enqueued"] == 2

    raw = stream.getvalue()
    assert SECRET_TEXT not in raw and SECRET_NAME not in raw and "0000" not in raw
    assert str(workspace.root) not in raw  # no paths either


def test_a_rescan_without_changes_skips_the_apply_stage(workspace: Workspace) -> None:
    workspace.write("a.md", "x")
    workspace.run()
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    workspace.run()
    stages = [e["stage"] for e in events(stream) if e["event"] == "stage.start"]
    assert stages == ["ingest.scan", "ingest.gc"]


def test_a_dry_run_is_logged_as_planned(workspace: Workspace) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    workspace.write("a.md", "x")
    workspace.ingest.plan(workspace.source.name)
    recorded = events(stream)
    assert recorded[-1]["event"] == "ingest.planned"
    assert recorded[0]["dry_run"] is True


def test_the_guard_failure_is_logged_with_its_code_but_without_details(
    workspace: Workspace,
) -> None:
    for i in range(3):
        workspace.write(f"{SECRET_NAME}-{i}.md", SECRET_TEXT)
    workspace.run()
    for i in range(3):
        (workspace.root / f"{SECRET_NAME}-{i}.md").unlink()
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    with pytest.raises(RagError):
        workspace.run()
    failed = [e for e in events(stream) if e["event"] == "stage.failed"]
    assert len(failed) == 1
    assert failed[0]["error_code"] == ErrorCode.E_MASS_DELETE_GUARD.value
    assert failed[0]["stage"] == "ingest.scan"
    assert SECRET_NAME not in stream.getvalue()
