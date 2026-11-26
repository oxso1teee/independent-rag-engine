"""Model-based test: random sequences of file operations and rescans against a naive model.

The model knows only the rules of the specification: a file that appears becomes an active
document, one that disappears becomes a tombstone (and is dropped after the grace period), a
tombstone whose file returns is restored with the same identity, and a file that is renamed
keeps its document. After every rescan the database must agree with the model, identities
must never change for keys that still exist, and all data-model invariants must hold.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from rag_engine.container import build_services, filesystem_adapter
from rag_engine.domain import Document, DocumentId
from rag_engine.storage.integrity import check_integrity
from rag_engine.storage.sqlite import open_store
from rag_engine.testing import FakeClock
from tests.factories import make_ids
from tests.integrity.checks import check_store_invariants

GRACE_DAYS = 30
NAMES = ["a.md", "b.md", "c.md", "d.md", "x/a.md", "x/b.md", "x/y/c.md"]


class IngestionMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.workdir = Path(tempfile.mkdtemp(prefix="rag-model-"))
        self.root = self.workdir / "root"
        self.root.mkdir()
        self.clock = FakeClock()
        self.day = 0
        self.store = open_store(self.workdir / "rag.db", self.clock)
        self.services = build_services(
            self.store, self.workdir / "rag.db", self.clock, make_ids(), filesystem_adapter
        )
        self.services.registry.add("docs", self.root)
        self.counter = 0
        self.disk: dict[str, str] = {}  # key -> content currently on disk
        self.known: dict[str, tuple[str, int | None]] = {}  # key -> (content, missing_day)
        self.identity: dict[str, DocumentId] = {}

    def teardown(self) -> None:
        self.store.close()
        shutil.rmtree(self.workdir, ignore_errors=True)

    # -- operations on the disk ------------------------------------------------------

    def _fresh_content(self, key: str) -> str:
        self.counter += 1
        return f"unique content #{self.counter} of {key}\n"

    def _write(self, key: str, content: str) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.disk[key] = content

    @rule(key=st.sampled_from(NAMES))
    def create_or_modify(self, key: str) -> None:
        self._write(key, self._fresh_content(key))

    @precondition(lambda self: bool(self.disk) and len(self.disk) < len(NAMES))
    @rule(data=st.data())
    def copy_content(self, data: st.DataObject) -> None:
        """A new file with the same bytes as an existing one: a duplicate."""
        source = data.draw(st.sampled_from(sorted(self.disk)))
        target = data.draw(st.sampled_from([n for n in NAMES if n not in self.disk]))
        self._write(target, self.disk[source])

    @precondition(lambda self: bool(self.disk))
    @rule(data=st.data())
    def delete(self, data: st.DataObject) -> None:
        key = data.draw(st.sampled_from(sorted(self.disk)))
        (self.root / key).unlink()
        del self.disk[key]

    @precondition(lambda self: bool(self.disk) and len(self.disk) < len(NAMES))
    @rule(data=st.data())
    def rename(self, data: st.DataObject) -> None:
        source = data.draw(st.sampled_from(sorted(self.disk)))
        target = data.draw(st.sampled_from([n for n in NAMES if n not in self.disk]))
        content = self.disk.pop(source)
        (self.root / source).unlink()
        self._write(target, content)  # same content under a new key

    @rule(days=st.sampled_from([0, 1, 7, 29, 30, 31, 45]))  # around the grace boundary
    def wait(self, days: int) -> None:
        self.clock.advance(timedelta(days=days))
        self.day += days

    # Composite steps make the interesting sequences (a file vanishes, time passes, the scan
    # runs again around the grace boundary) likely instead of rare.
    @precondition(lambda self: bool(self.disk))
    @rule(data=st.data())
    def delete_and_rescan(self, data: st.DataObject) -> None:
        self.delete(data)
        self.rescan()

    @rule(days=st.sampled_from([0, 1, 7, 29, 30, 31, 45]))
    def wait_and_rescan(self, days: int) -> None:
        self.wait(days)
        self.rescan()

    # -- the scan and the model's prediction ---------------------------------------

    @rule()
    def rescan(self) -> None:
        before = {d.source_key.value for d in self._documents()}
        self._predict()
        self.services.ingest.run("docs", accept_mass_delete=True)
        by_key = {d.source_key.value: d for d in self._documents()}
        assert {k: (d.status.value, d.missing_since is not None) for k, d in by_key.items()} == {
            k: ("missing" if missing is not None else "active", missing is not None)
            for k, (_, missing) in self.known.items()
        }
        for key, document in by_key.items():
            if key in before and key in self.identity:
                assert document.id == self.identity[key], f"{key} changed identity"
        self.identity = {k: d.id for k, d in by_key.items()}
        assert all(d.current_version_id is None for d in by_key.values())  # nothing parses yet

    def _documents(self) -> list[Document]:
        source = self.store.reader().get_source_by_name("docs")
        assert source is not None
        return list(self.store.reader().list_documents(source.id))

    def _predict(self) -> None:
        """Advance the naive model through one scan (the rules of the specification)."""
        live = {k: c for k, (c, missing) in self.known.items() if missing is None}
        vanished = {k: c for k, c in live.items() if k not in self.disk}
        appeared = {k: c for k, c in self.disk.items() if k not in self.known}
        held_elsewhere = {c for k, c in live.items() if k in self.disk}
        moves: dict[str, str] = {}
        for old, content in vanished.items():
            twins_old = [k for k, c in vanished.items() if c == content]
            twins_new = [k for k, c in appeared.items() if c == content]
            if len(twins_old) == 1 and len(twins_new) == 1 and content not in held_elsewhere:
                moves[old] = twins_new[0]
        for old, new in moves.items():
            self.known[new] = (self.known.pop(old)[0], None)
            self.identity[new] = self.identity.pop(old)
        for key, content in self.disk.items():
            if key not in moves.values():
                self.known[key] = (content, None)  # new, modified or restored
        for key in vanished:
            if key not in moves:
                self.known[key] = (self.known[key][0], self.day)
        for key, (_, missing) in list(self.known.items()):
            if missing is not None and missing + GRACE_DAYS <= self.day:
                del self.known[key]  # garbage-collected
                self.identity.pop(key, None)

    @invariant()
    def data_model_invariants_hold(self) -> None:
        check_store_invariants(self.store.reader())
        assert check_integrity(self.workdir / "rag.db").ok


@pytest.mark.integrity
def test_ingestion_follows_the_naive_model() -> None:
    run = IngestionMachine.TestCase
    run.settings = settings(max_examples=80, stateful_step_count=22, deadline=None)
    run().runTest()
