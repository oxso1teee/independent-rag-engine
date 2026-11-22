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
from rag_engine.cli.context import Processing, Services
from rag_engine.config import load_settings
from rag_engine.container import build_processing, build_services, filesystem_adapter
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

    def processing(self, config: Path | None) -> Processing:
        return build_processing(load_settings(config), self.clock, self.ids)


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
    assert source["documents"] == {"active": 10} and source["sync"] == {"changed": 10}
    assert source["awaiting_first_ingest"] == 10
    code, out, _ = cli.run("status")
    assert "jobs: 10 queued" in out and "notes: 10 active (10 awaiting first ingestion)" in out

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
    for i in range(10):  # every known document vanishes
        (docs / "notes" / f"n{i}.md").unlink()
    code, _, err = cli.run("ingest", "notes")
    assert code == 4 and "E_MASS_DELETE_GUARD" in err and "--accept-mass-delete" in err
    code, payload = cli.json("ingest", "notes")
    assert code == 4 and payload["error"]["code"] == "E_MASS_DELETE_GUARD"
    assert cli.run("ingest", "notes", "--dry-run")[0] == 4
    assert cli.run("ingest", "notes", "--allow-mass-delete")[0] == 2  # the old flag is gone
    code, out, _ = cli.run("ingest", "notes", "--accept-mass-delete")
    assert code == 0 and "10 missing" in out


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
    for i in range(10):
        (docs / "notes" / f"n{i}.md").unlink()
    (other / "f0.md").write_text("changed", encoding="utf-8")
    code, payload = cli.json("ingest")
    assert code == 4
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
    assert cli.run("ingest", "unknown")[0] == 4  # E_SOURCE_NOT_FOUND: category source
    assert cli.run("source", "add", "missing", str(tmp_path / "nope"))[0] == 4
    assert cli.run("source", "add", "notes", str(tmp_path))[0] == 4  # E_SOURCE_LOCATOR_CHANGED
    assert cli.run("source", "add", "bad", str(docs), "--include", "[x")[0] == 3  # config
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


def test_a_second_writer_is_reported_as_a_storage_error(cli: Cli, docs: Path) -> None:
    cli.run("source", "add", "notes", str(docs))
    with (
        open_store(cli.container.default_db, cli.container.clock) as holder,
        holder.unit_of_work().begin(),
    ):
        code, _, err = cli.run("ingest", "notes")
    assert code == 8 and "E_STORE_LOCKED" in err
    assert cli.run("ingest", "notes")[0] == 0  # released: works again


# ---------------------------------------------------------------------- rag chunks

NOTE = (
    "# Notes\n\nFirst paragraph of the note.\n\nSecond paragraph of the note.\n\n"
    "Footer: generated automatically\n"
)


@pytest.fixture
def note(tmp_path: Path) -> Path:
    path = tmp_path / "note.md"
    path.write_text(NOTE, encoding="utf-8")
    return path


def test_chunks_of_a_file_need_no_database(cli: Cli, note: Path) -> None:
    code, out, err = cli.run("chunks", str(note))
    assert code == 0 and err == ""
    assert out.startswith("note.md: md -> structure_aware (")
    assert "Footer: generated automatically" in out and "[Notes]" in out
    assert not cli.container.default_db.exists()


def test_chunks_json_describes_document_chunker_and_every_chunk(cli: Cli, note: Path) -> None:
    code, payload = cli.json("chunks", str(note))
    assert code == 0
    assert payload["document"]["parser"] == "md" and payload["document"]["title"] == "Notes"
    assert payload["document"]["normalizer_version"] == "norm/1"
    assert payload["chunking"]["chunker"] == "structure_aware"
    assert payload["chunking"]["target_tokens"] == 256
    assert payload["summary"]["leaves"] == 1 and payload["summary"]["sections"] == 0
    (chunk,) = payload["chunks"]
    assert chunk["ordinal"] == 0 and chunk["parent"] is None and chunk["heading_path"] == ["Notes"]
    assert chunk["text"].startswith("Notes\n\nFirst paragraph")
    assert "id" not in chunk


def test_chunks_output_is_the_same_every_time(cli: Cli, note: Path) -> None:
    first, second = cli.run("chunks", str(note), "--full"), cli.run("chunks", str(note), "--full")
    assert first == second
    assert cli.run("chunks", str(note), "--json") == cli.run("chunks", str(note), "--json")


def test_full_prints_whole_texts_and_the_preview_is_short(cli: Cli, tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("word " * 60 + "THE-END", encoding="utf-8")
    assert "THE-END" not in cli.run("chunks", str(path))[1]
    assert "THE-END" in cli.run("chunks", str(path), "--full")[1]


def test_a_cleaner_drops_matching_lines_and_versions_the_normaliser(cli: Cli, note: Path) -> None:
    code, payload = cli.json("chunks", str(note), "-C", "^Footer:.*$")
    assert code == 0
    assert "Footer" not in payload["chunks"][0]["text"]
    assert payload["document"]["normalizer_version"].startswith("norm/1+")


def test_the_strategy_can_be_chosen(cli: Cli, note: Path) -> None:
    code, payload = cli.json("chunks", str(note), "--strategy", "fixed")
    assert code == 0 and payload["chunking"]["chunker"] == "fixed"
    code, _, err = cli.run("chunks", str(note), "--strategy", "semantic")
    assert code == 3 and "E_CONFIG_INVALID" in err


def test_source_cleaners_apply_to_chunks_of_that_source(cli: Cli, docs: Path) -> None:
    (docs / "notes" / "n0.md").write_text(NOTE, encoding="utf-8")
    assert cli.run("source", "add", "notes", str(docs), "-C", "^Footer:.*$")[0] == 0
    code, payload = cli.json("chunks", "--source", "notes", "notes/n0.md")
    assert code == 0 and payload["document"]["name"] == "notes/n0.md"
    assert "Footer" not in payload["chunks"][0]["text"]
    extra = cli.json("chunks", "-s", "notes", "notes/n0.md", "-C", "First paragraph.*")[1]
    assert "First paragraph" not in extra["chunks"][0]["text"]
    assert extra["document"]["normalizer_version"] != payload["document"]["normalizer_version"]


def test_an_unsafe_cleaner_is_refused_when_the_source_is_added(cli: Cli, docs: Path) -> None:
    code, _, err = cli.run("source", "add", "notes", str(docs), "-C", "(a+)+$")
    assert code == 3 and "E_CONFIG_INVALID" in err
    assert not cli.container.default_db.exists() or cli.json("source", "list")[1]["sources"] == []


def test_chunks_failures_have_their_own_exit_codes(cli: Cli, docs: Path, tmp_path: Path) -> None:
    assert cli.run("chunks", str(tmp_path / "absent.md"))[0] == 4
    assert cli.run("chunks", str(tmp_path))[0] == 4  # a directory is not a document
    picture = tmp_path / "pic.md"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    code, _, err = cli.run("chunks", str(picture))
    assert code == 5 and "E_UNSUPPORTED_FORMAT" in err
    code, payload = cli.json("chunks", str(picture))
    assert code == 5 and payload["error"]["code"] == "E_UNSUPPORTED_FORMAT"
    code, _, err = cli.run("chunks", "--source", "notes", "notes/n0.md")  # no database yet
    assert code == 4 and "no database yet" in err
    cli.run("source", "add", "notes", str(docs))
    assert cli.run("chunks", "--source", "other", "notes/n0.md")[0] == 4
    assert cli.run("chunks", "--source", "notes", "notes/absent.md")[0] == 4
    assert cli.run("chunks", "--source", "notes", "../outside.md")[0] == 4


def test_chunks_help_is_listed(cli: Cli) -> None:
    assert "chunks" in cli.run("--help")[1]
    code, out, _ = cli.run("chunks", "--help")
    assert code == 0 and all(w in out for w in ("--source", "--cleaner", "--strategy", "--full"))


def test_html_is_parsed_in_the_isolated_worker_by_default(cli: Cli, tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<main><h1>Title</h1><p>Body of the page.</p></main>", encoding="utf-8")
    code, payload = cli.json("chunks", str(page))
    assert code == 0 and payload["document"]["parser"] == "html"
    assert payload["chunks"][0]["text"].startswith("Title")


def test_isolation_can_be_turned_off_in_the_configuration(cli: Cli, tmp_path: Path) -> None:
    config = tmp_path / "rag.toml"
    config.write_text("[parsing]\nisolate = []\n", encoding="utf-8")
    page = tmp_path / "page.html"
    page.write_text("<main><p>Body.</p></main>", encoding="utf-8")
    code, payload = cli.json("--config", str(config), "chunks", str(page))
    assert code == 0 and payload["document"]["parser"] == "html"


def test_ingest_reports_copies_of_identical_files(cli: Cli, docs: Path) -> None:
    (docs / "notes" / "copy-of-n0.md").write_text("# note 0\n", encoding="utf-8")
    cli.run("source", "add", "notes", str(docs))
    code, payload = cli.json("ingest", "notes")
    assert code == 0 and payload["results"][0]["duplicates"] == 1
    code, out, _ = cli.run("ingest")
    assert code == 0 and "duplicate link" not in out  # nothing changed, nothing to report
    (docs / "notes" / "copy-of-n0.md").unlink()  # the copy disappears: its link is cleared
    code, out, _ = cli.run("ingest")
    assert code == 0 and "1 duplicate link(s) updated" in out
