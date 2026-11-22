"""Which documents of a source are copies of another document of the same source.

Two documents are copies when their raw bytes are identical (equal ``content_hash``). Among
the *active* documents of one hash the one with the smallest ``DocumentId`` is the original: it
keeps ``duplicate_of`` empty and every other member points to it, so there are never chains.
Documents that are not active hold no link. The links are recomputed for the whole source, so
they follow additions, modifications, renames, disappearances and restorations, and an original
that goes away is replaced by the next one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime

from rag_engine.domain import (
    ContentHash,
    Document,
    DocumentId,
    DocumentStatus,
    SourceId,
    SourceKey,
)
from rag_engine.ports import MetadataWriter


def plan_duplicates(
    documents: Sequence[Document], hashes: Mapping[SourceKey, ContentHash]
) -> dict[DocumentId, DocumentId | None]:
    """The wanted ``duplicate_of`` of every document of one source."""
    groups: dict[ContentHash, list[Document]] = defaultdict(list)
    for document in documents:
        digest = hashes.get(document.source_key)
        if document.status is DocumentStatus.ACTIVE and digest is not None:
            groups[digest].append(document)
    wanted: dict[DocumentId, DocumentId | None] = {document.id: None for document in documents}
    for members in groups.values():
        original = min(members, key=lambda document: document.id)
        for document in members:
            if document is not original:
                wanted[document.id] = original.id
    return wanted


def resolve_duplicates(writer: MetadataWriter, source_id: SourceId, now: datetime) -> int:
    """Bring ``duplicate_of`` of the source in line with its content; the links that changed."""
    documents = writer.list_documents(source_id)
    hashes = {
        state.source_key: state.last_content_hash
        for state in writer.list_sync_states(source_id)
        if state.last_content_hash is not None
    }
    wanted = plan_duplicates(documents, hashes)
    changed = 0
    for document in documents:
        if document.duplicate_of != wanted[document.id]:
            writer.put_document(replace(document, duplicate_of=wanted[document.id], updated_at=now))
            changed += 1
    return changed
