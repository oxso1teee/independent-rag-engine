"""Stable identity of sources and documents, locator management and rebase.

A document is identified by ``(source, key)``; its ``DocumentId`` never changes while the
key exists. A source is identified by its ``SourceId``; the locator (root directory) may
change only through an explicit, previewable rebase. Documents whose key changed are
re-attached only when the match is provable: identical content with exactly one candidate
on each side. Anything ambiguous stays separate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from rag_engine.domain import (
    ContentHash,
    DocumentId,
    ErrorCode,
    IdFactory,
    RagError,
    Sha256Hex,
    Source,
    SourceKey,
    SourceKind,
    SourcePolicy,
    SyncState,
)
from rag_engine.ingestion.change import Move, hash_stream
from rag_engine.ingestion.sources.fs import locator_fingerprint, normalize_locator
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.ports import Clock, SourceAdapter, Store

AdapterFactory = Callable[[Source], SourceAdapter]
"""Builds the adapter that reads a registered source (injected by the composition root)."""


def match_moves(
    vanished: Sequence[tuple[SourceKey, DocumentId, ContentHash]],
    appeared: Sequence[tuple[SourceKey, ContentHash]],
    also_present: Collection[ContentHash] = (),
) -> tuple[list[tuple[SourceKey, DocumentId, SourceKey, ContentHash]], list[ContentHash]]:
    """Pair vanished and appeared entries that share a content hash exactly once on each side.

    A content that is also held by a document that did not move is never matched: the
    appeared entry could just as well be a copy of that document.

    Returns the proven moves ``(old_key, document_id, new_key, hash)`` and the hashes that
    were ambiguous (several candidates on either side) and therefore left unmatched.
    """
    old_by_hash: dict[ContentHash, list[tuple[SourceKey, DocumentId]]] = defaultdict(list)
    new_by_hash: dict[ContentHash, list[SourceKey]] = defaultdict(list)
    for key, document_id, digest in vanished:
        old_by_hash[digest].append((key, document_id))
    for key, digest in appeared:
        new_by_hash[digest].append(key)
    moves: list[tuple[SourceKey, DocumentId, SourceKey, ContentHash]] = []
    ambiguous: list[ContentHash] = []
    for digest, olds in sorted(old_by_hash.items(), key=lambda item: item[0].hex):
        news = new_by_hash.get(digest, [])
        if not news:
            continue
        if len(olds) == 1 and len(news) == 1 and digest not in also_present:
            moves.append((olds[0][0], olds[0][1], news[0], digest))
        else:
            ambiguous.append(digest)
    return moves, ambiguous


@dataclass(frozen=True, slots=True)
class RebasePlan:
    source_name: str
    old_locator_fingerprint: Sha256Hex
    new_locator_fingerprint: Sha256Hex
    kept: int  # documents whose key exists under the new root
    moves: tuple[Move, ...]  # documents re-attached by unique content match
    unmatched_old: tuple[SourceKey, ...]
    unmatched_new: tuple[SourceKey, ...]
    ambiguous: int  # distinct contents with several candidates, deliberately not merged
    unchanged_locator: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source_name,
            "kept": self.kept,
            "moved": len(self.moves),
            "unmatched_existing": len(self.unmatched_old),
            "unmatched_new": len(self.unmatched_new),
            "ambiguous": self.ambiguous,
            "locator_unchanged": self.unchanged_locator,
        }


class SourceRegistry:
    """Registers sources and keeps their identity consistent."""

    def __init__(
        self, store: Store, clock: Clock, ids: IdFactory, adapter_factory: AdapterFactory
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self._adapter_factory = adapter_factory

    # -- registration -----------------------------------------------------------------

    def add(self, name: str, root: Path | str, policy: SourcePolicy | None = None) -> Source:
        """Register a filesystem source; adding the same name and root again is a no-op."""
        locator = normalize_locator(root)
        if not Path(locator).is_dir():
            raise RagError(
                ErrorCode.E_SOURCE_NOT_FOUND,
                "the source root does not exist or is not a directory",
                context={"name": name},
            )
        existing = self._store.reader().get_source_by_name(name)
        if existing is not None:
            if existing.locator == locator:
                return existing
            raise RagError(
                ErrorCode.E_SOURCE_LOCATOR_CHANGED,
                "a source with this name already points to a different root; "
                "use an explicit rebase to move it",
                context={"name": name},
            )
        now = self._clock.now()
        source = Source(
            id=self._ids.source_id(),
            name=name,
            kind=SourceKind.FS,
            locator=locator,
            locator_fingerprint=locator_fingerprint(locator),
            policy=policy or SourcePolicy(),
            created_at=now,
            updated_at=now,
        )
        self._adapter_factory(source)  # validates the policy (globs) before anything is stored
        compile_cleaners(source.policy.cleaners)  # and the cleaners
        with self._store.unit_of_work().begin() as writer:
            writer.put_source(source)
        return source

    def list(self) -> Sequence[Source]:
        return self._store.reader().list_sources()

    def get(self, name: str) -> Source:
        source = self._store.reader().get_source_by_name(name)
        if source is None:
            raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, "unknown source", context={"name": name})
        return source

    def open(self, name: str, key: SourceKey) -> tuple[Source, BinaryIO]:
        """The source and a read stream of one of its documents (the caller closes it)."""
        source = self.get(name)
        adapter = self._adapter_factory(source)
        self.verify(source, adapter)
        return source, adapter.open(key)

    def verify(self, source: Source, adapter: SourceAdapter) -> None:
        """Refuse to work when the adapter reads a different root than the registered one."""
        if adapter.identity().locator_fingerprint != source.locator_fingerprint:
            raise RagError(
                ErrorCode.E_SOURCE_LOCATOR_CHANGED,
                "the source root changed; use an explicit rebase to move it",
                context={"name": source.name},
            )

    # -- rebase -----------------------------------------------------------------------

    def rebase(self, name: str, new_root: Path | str, *, apply: bool) -> RebasePlan:
        """Plan (``apply=False``) or perform the move of a source to a new root."""
        source = self.get(name)
        locator = normalize_locator(new_root)
        fingerprint = locator_fingerprint(locator)
        assert source.locator_fingerprint is not None  # noqa: S101 - filesystem sources have one
        if fingerprint == source.locator_fingerprint:
            return RebasePlan(name, source.locator_fingerprint, fingerprint, 0, (), (), (), 0, True)
        candidate = replace(source, locator=locator, locator_fingerprint=fingerprint)
        adapter = self._adapter_factory(candidate)
        report = adapter.scan({})  # E_SOURCE_NOT_FOUND when the new root is unusable
        reader = self._store.reader()
        documents = {d.source_key: d for d in reader.list_documents(source.id)}
        states = {s.source_key: s for s in reader.list_sync_states(source.id)}
        new_entries = {e.key: e.revision for e in report.entries}
        kept = sum(1 for key in documents if key in new_entries)
        old_left = [
            (key, doc.id, states[key].last_content_hash)
            for key, doc in documents.items()
            if key not in new_entries and key in states
        ]
        vanished = [(k, d, h) for k, d, h in old_left if h is not None]
        new_left = sorted((k for k in new_entries if k not in documents), key=lambda k: k.value)
        appeared: list[tuple[SourceKey, ContentHash]] = []
        for key in new_left:
            try:
                with adapter.open(key) as stream:
                    appeared.append((key, hash_stream(stream)))
            except RagError:
                continue  # unreadable files cannot be proven to be anything
        held_elsewhere = {
            states[key].last_content_hash
            for key in documents
            if key in new_entries and key in states
        }
        found, ambiguous = match_moves(
            vanished, appeared, [h for h in held_elsewhere if h is not None]
        )
        moves = tuple(
            Move(document_id, old, new, new_entries[new], digest)
            for old, document_id, new, digest in sorted(found, key=lambda m: m[0].value)
        )
        moved_old = {m.old_key for m in moves}
        moved_new = {m.new_key for m in moves}
        plan = RebasePlan(
            source_name=name,
            old_locator_fingerprint=source.locator_fingerprint,
            new_locator_fingerprint=fingerprint,
            kept=kept,
            moves=moves,
            unmatched_old=tuple(
                sorted(
                    (k for k in documents if k not in new_entries and k not in moved_old),
                    key=lambda k: k.value,
                )
            ),
            unmatched_new=tuple(k for k in new_left if k not in moved_new),
            ambiguous=len(ambiguous),
        )
        if apply:
            self._apply_rebase(source, locator, fingerprint, plan, states)
        return plan

    def _apply_rebase(
        self,
        source: Source,
        locator: str,
        fingerprint: Sha256Hex,
        plan: RebasePlan,
        states: dict[SourceKey, SyncState],
    ) -> None:
        now = self._clock.now()
        with self._store.unit_of_work().begin() as writer:
            writer.rebase_source(source.id, locator, fingerprint, now)
            for move in plan.moves:
                document = writer.get_document(move.document_id)
                state = states.get(move.old_key)
                if document is None or state is None:
                    continue
                writer.delete_sync_state(source.id, move.old_key)
                writer.put_document(replace(document, source_key=move.new_key, updated_at=now))
                writer.put_sync_state(
                    replace(
                        state,
                        source_key=move.new_key,
                        last_revision=move.revision,
                        last_seen_at=now,
                    )
                )
