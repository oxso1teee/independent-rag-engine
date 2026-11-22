"""P2.9: the ``rag`` command end to end against a temporary workspace."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rag_engine.cli import main
from rag_engine.cli.container import Services, build_services, filesystem_adapter
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.testing import FakeClock
from tests.factories import make_ids


class ContainerForTests:
    """Dependency injection for tests: fake clock and deterministic ids."""

    def __init__(self, default_db: Path, clock: FakeClock) -> None:
        self.default_db = default_db
        self.clock = clock
        self.ids = make_ids()

    @contextmanager
    def open(self, config: Path | None, db: Path | None, *, read_only: bool) -> Iterator[Services]:
        path = db or self.default_db
        store = open_store_read_only(path) if read_only else open_store(path, self.clock)
        try:
            yield build_services(store, path, self.clock, self.ids, filesystem_adapter)
        finally:
            store.close()


@dataclass
class Cli:
    container: ContainerForTests
    capsys: pytest.CaptureFixture[str]

    def run(self, *args: str) -> tuple[int, str, str]:
        code = main(list(args), container=self.container)
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def json(self, *args: str) -> tuple[int, dict[str, Any]]:
        code, out, err = self.run(*args, "--json")
        assert err == ""
        return code, json.loads(out)


@pytest.fixture
def cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Cli:
    return Cli(ContainerForTests(tmp_path / "state" / "rag.db", FakeClock()), capsys)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "notes").mkdir(parents=True)
    for i in range(10):
        (root / "notes" / f"n{i}.md").write_text(f"# note {i}\n", encoding="utf-8")
    return root


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_help_lists_every_command(cli: Cli) -> None:
    code, out, _ = cli.run("--help")
    assert code == 0
    for word in ("source", "ingest", "status"):
        assert word in out
    code, out, _ = cli.run("source", "--help")
    assert code == 0
    assert all(word in out for word in ("add", "list", "rebase"))


def test_source_add_and_list(cli: Cli, docs: Path) -> None:
    code, out, _ = cli.run("source", "add", "notes", str(docs))
    assert code == 0 and "Source 'notes'" in out
    code, out, _ = cli.run("source", "list")
    assert code == 0 and out.split() == ["notes", str(docs.resolve())]
    code, payload = cli.json("source", "list")
    assert code == 0
    assert payload == {
        "sources": [{"name": "notes", "locator": str(docs.resolve()), "delete_grace_days": 30}]
    }
    assert cli.run("source", "add", "notes", str(docs))[0] == 0  # idempotent
    assert len(cli.json("source", "list")[1]["sources"]) == 1


def test_list_and_status_without_a_database_say_so_and_create_nothing(cli: Cli) -> None:
    code, out, _ = cli.run("source", "list")
    assert code == 0 and "No sources registered." in out
    code, out, _ = cli.run("status")
    assert code == 0 and "No database yet." in out
    assert not cli.container.default_db.exists()


def test_ingest_status_and_rescan(cli: Cli, docs: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    code, payload = cli.json("ingest", "notes")
    assert code == 0
    (result,) = payload["results"]
    assert result["new"] == 10 and result["jobs_enqueued"] == 10 and result["unchanged"] == 0

    code, payload = cli.json("ingest")  # all sources, nothing changed
    (again,) = payload["results"]
    assert again["unchanged"] == 10 and again["new"] == 0 and again["jobs_enqueued"] == 0

    code, payload = cli.json("status")
    assert code == 0
    assert payload["jobs"] == {"queued": 10}
    (source,) = payload["sources"]
    assert source["documents"] == {"pending": 10} and source["sync"] == {"changed": 10}
    code, out, _ = cli.run("status")
    assert "jobs: 10 queued" in out and "notes: 10 pending" in out

    code, out, _ = cli.run("ingest")
    assert code == 0 and "10 unchanged" in out and "0 job(s) queued" in out


def test_dry_run_writes_nothing_anywhere(cli: Cli, docs: Path, tmp_path: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    db = cli.container.default_db
    before_state = tree_digest(db.parent)
    before_docs = tree_digest(docs)

    code, out, _ = cli.run("ingest", "--dry-run")
    assert code == 0 and "10 new" in out and "(would)" in out
    code, payload = cli.json("ingest", "notes", "--dry-run")
    assert code == 0 and payload["dry_run"] is True

    assert tree_digest(docs) == before_docs
    assert file_digest(db) == before_state[str(db.relative_to(db.parent))]
    assert cli.json("status")[1]["jobs"] == {}  # nothing was queued


def test_dry_run_without_a_database_fails_without_creating_one(cli: Cli, docs: Path) -> None:
    code, _, err = cli.run("ingest", "notes", "--dry-run")
    assert code == 4 and "no database yet" in err
    assert not cli.container.default_db.exists()


def test_mass_delete_guard_has_its_own_exit_code(cli: Cli, docs: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    cli.run("ingest")
    for i in range(5):
        (docs / "notes" / f"n{i}.md").unlink()
    code, _, err = cli.run("ingest", "notes")
    assert code == 6 and "E_MASS_DELETE_GUARD" in err
    code, payload = cli.json("ingest", "notes")
    assert code == 6 and payload["error"]["code"] == "E_MASS_DELETE_GUARD"
    assert cli.run("ingest", "notes", "--dry-run")[0] == 6
    code, out, _ = cli.run("ingest", "notes", "--allow-mass-delete")
    assert code == 0 and "5 missing" in out


def test_ingesting_all_sources_reports_each_failure_and_keeps_going(
    cli: Cli, docs: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    for i in range(3):
        (other / f"f{i}.md").write_text(f"{i}", encoding="utf-8")
    cli.run("source", "add", "a-notes", str(docs))
    cli.run("source", "add", "b-other", str(other))
    cli.run("ingest")
    for i in range(5):
        (docs / "notes" / f"n{i}.md").unlink()
    (other / "f0.md").write_text("changed", encoding="utf-8")
    code, payload = cli.json("ingest")
    assert code == 6
    (good,) = payload["results"]
    assert good["source"] == "b-other" and good["modified"] == 1
    assert payload["errors"][0]["code"] == "E_MASS_DELETE_GUARD"


def test_rebase_preview_and_apply(cli: Cli, docs: Path, tmp_path: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    cli.run("ingest")
    moved = tmp_path / "moved"
    docs.rename(moved)
    before = cli.container.default_db.read_bytes()
    code, out, _ = cli.run("source", "rebase", "notes", str(moved), "--dry-run")
    assert code == 0 and "Dry run" in out and "10 kept" in out
    assert cli.container.default_db.read_bytes() == before
    code, payload = cli.json("source", "rebase", "notes", str(moved))
    assert code == 0 and payload["rebase"]["kept"] == 10
    code, out, _ = cli.run("ingest")
    assert code == 0 and "10 unchanged" in out
    assert "nothing to do" in cli.run("source", "rebase", "notes", str(moved))[1]


def test_error_codes_are_stable(cli: Cli, docs: Path, tmp_path: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    assert cli.run("ingest", "unknown")[0] == 4  # E_SOURCE_NOT_FOUND
    assert cli.run("source", "add", "missing", str(tmp_path / "nope"))[0] == 4
    assert cli.run("source", "add", "notes", str(tmp_path))[0] == 5  # E_SOURCE_LOCATOR_CHANGED
    assert cli.run("source", "add", "bad", str(docs), "--include", "[x")[0] == 3  # E_CONFIG_INVALID
    assert cli.run("source", "add", "Bad Name", str(docs))[0] == 9  # E_INTEGRITY
    assert cli.run("source", "add")[0] == 2  # usage
    assert cli.run("ingest", "--bogus")[0] == 2


def test_json_errors_are_one_document_on_stdout(cli: Cli) -> None:
    code, out, err = cli.run("ingest", "ghost", "--json")
    assert code == 4 and err == ""
    assert json.loads(out)["error"]["code"] == "E_SOURCE_NOT_FOUND"


def test_include_exclude_and_policy_options(cli: Cli, docs: Path) -> None:
    (docs / "skip").mkdir()
    (docs / "skip" / "x.md").write_text("x", encoding="utf-8")
    (docs / "readme.txt").write_text("t", encoding="utf-8")
    cli.run("source", "add", "md", str(docs), "-i", "**/*.md", "-x", "skip/**", "--grace-days", "0")
    code, payload = cli.json("ingest", "md")
    (result,) = payload["results"]
    assert code == 0 and result["new"] == 10


def test_real_process_entry_point_and_exit_codes(tmp_path: Path, docs: Path) -> None:
    env = {**os.environ, "RAG_STORAGE__DB_PATH": str(tmp_path / "real" / "rag.db")}
    root = Path(__file__).resolve().parents[2]

    def rag(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "rag_engine", *args],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert rag("--version").returncode == 0
    assert rag("source", "add", "notes", str(docs)).returncode == 0
    first = rag("ingest", "notes", "--json")
    assert first.returncode == 0 and json.loads(first.stdout)["results"][0]["new"] == 10
    second = rag("ingest", "notes", "--json")
    assert json.loads(second.stdout)["results"][0]["unchanged"] == 10
    assert rag("ingest", "ghost").returncode == 4
    assert (tmp_path / "real" / "rag.db").is_file()
