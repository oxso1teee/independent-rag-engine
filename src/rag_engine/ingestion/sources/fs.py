"""Local filesystem source: recursive discovery with a strict root boundary."""

from __future__ import annotations

import os
import stat
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

from rag_engine.domain import (
    ErrorCode,
    RagError,
    Sha256Hex,
    SourceKey,
    SourceKind,
    SourcePolicy,
    SourceRevision,
)
from rag_engine.domain.globs import GlobSet
from rag_engine.ports import ScanIssue, ScanReport, SourceEntry, SourceIdentity


def normalize_locator(root: Path | str) -> str:
    """Canonical text of a root directory: absolute, symlinks resolved, NFC, ``/``-separated.

    The locator is what identifies a filesystem source; its SHA-256 is the fingerprint.
    """
    resolved = Path(root).expanduser().resolve()
    return unicodedata.normalize("NFC", resolved.as_posix())


def locator_fingerprint(locator: str) -> Sha256Hex:
    return Sha256Hex.of_text(locator)


def _not_found(key: str, reason: str) -> RagError:
    return RagError(
        ErrorCode.E_SOURCE_NOT_FOUND,
        "entry does not exist or cannot be read",
        context={"key": key, "reason": reason},
    )


def _encodable(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class FilesystemSource:
    """``SourceAdapter`` over a directory tree.

    * Keys are the NFC-normalised, ``/``-separated paths relative to the root.
    * Nothing outside the root is ever read: symlinks are skipped unless the policy follows
      them, and followed links must still resolve inside the root.
    * Unreadable directories are reported as unreadable *subtrees*, so callers never mistake
      them for deleted content.
    """

    def __init__(self, root: Path | str, policy: SourcePolicy | None = None) -> None:
        self._policy = policy or SourcePolicy()
        self._globs = GlobSet(self._policy.include, self._policy.exclude)
        self._root = Path(root).expanduser().resolve()
        self._locator = normalize_locator(self._root)
        self._paths: dict[SourceKey, str] = {}  # key -> real relative path from the last scan

    @property
    def locator(self) -> str:
        return self._locator

    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            kind=SourceKind.FS, locator_fingerprint=locator_fingerprint(self._locator)
        )

    # ------------------------------------------------------------------ discovery

    def discover(self, previous: Mapping[SourceKey, SourceRevision]) -> tuple[SourceEntry, ...]:
        return self.scan(previous).entries

    def scan(self, previous: Mapping[SourceKey, SourceRevision]) -> ScanReport:
        del previous  # a stat per file is the cheap change signal; nothing to skip
        if not self._root.is_dir():
            raise RagError(
                ErrorCode.E_SOURCE_NOT_FOUND,
                "the source root does not exist or is not a directory",
                context={"locator_fingerprint": locator_fingerprint(self._locator).hex},
            )
        found: dict[str, list[tuple[str, SourceRevision]]] = {}
        unreadable: list[ScanIssue] = []
        skipped: list[ScanIssue] = []
        root_real = os.path.realpath(self._root)
        visited: set[str] = {root_real}
        stack: list[str] = [""]  # real relative directory paths, "" is the root
        while stack:
            rel_dir = stack.pop()
            directory = self._root / rel_dir if rel_dir else self._root
            try:
                with os.scandir(directory) as iterator:
                    children = sorted(iterator, key=lambda e: e.name)
            except OSError:
                if rel_dir == "":
                    raise RagError(
                        ErrorCode.E_SOURCE_NOT_FOUND,
                        "the source root cannot be read",
                        context={"locator_fingerprint": locator_fingerprint(self._locator).hex},
                    ) from None
                unreadable.append(
                    ScanIssue(self._key_text(rel_dir), "directory_unreadable", subtree=True)
                )
                continue
            for child in children:
                self._visit(child, rel_dir, root_real, visited, stack, found, unreadable, skipped)
        entries: list[SourceEntry] = []
        paths: dict[SourceKey, str] = {}
        for key_text, variants in found.items():
            if len(variants) > 1:  # two names normalise to one key: refuse to guess
                skipped.append(ScanIssue(key_text, "key_collision"))
                continue
            real_rel, revision = variants[0]
            key = SourceKey(key_text)
            paths[key] = real_rel
            entries.append(SourceEntry(key=key, revision=revision))
        entries.sort(key=lambda e: e.key.value)
        self._paths = paths
        return ScanReport(
            entries=tuple(entries),
            unreadable=tuple(sorted(unreadable, key=lambda i: i.path)),
            skipped=tuple(sorted(skipped, key=lambda i: (i.path, i.reason))),
        )

    def _visit(
        self,
        child: os.DirEntry[str],
        rel_dir: str,
        root_real: str,
        visited: set[str],
        stack: list[str],
        found: dict[str, list[tuple[str, SourceRevision]]],
        unreadable: list[ScanIssue],
        skipped: list[ScanIssue],
    ) -> None:
        real_rel = f"{rel_dir}/{child.name}" if rel_dir else child.name
        if not _encodable(real_rel) or "\\" in real_rel:
            skipped.append(ScanIssue(real_rel.encode("utf-8", "replace").decode(), "invalid_name"))
            return
        key_text = self._key_text(real_rel)
        try:
            is_link = child.is_symlink()
            if is_link:
                if not self._policy.follow_symlinks:
                    skipped.append(ScanIssue(key_text, "symlink_skipped"))
                    return
                target = os.path.realpath(child.path)
                if os.path.commonpath([root_real, target]) != root_real:
                    skipped.append(ScanIssue(key_text, "symlink_escape"))
                    return
            info = child.stat(follow_symlinks=self._policy.follow_symlinks)
        except OSError:
            unreadable.append(ScanIssue(key_text, "unreadable"))
            return
        if stat.S_ISDIR(info.st_mode):
            if self._globs.prunes_directory(key_text):
                return
            real = os.path.realpath(child.path)
            if real in visited:  # a followed link must not create a cycle
                skipped.append(ScanIssue(key_text, "directory_cycle"))
                return
            visited.add(real)
            stack.append(real_rel)
            return
        if not stat.S_ISREG(info.st_mode):
            skipped.append(ScanIssue(key_text, "special_file"))
            return
        if not self._globs.selects(key_text):
            return
        if info.st_size > self._policy.max_file_bytes:
            skipped.append(ScanIssue(key_text, "too_large"))
            return
        revision = SourceRevision(info.st_size, max(0, info.st_mtime_ns))
        found.setdefault(key_text, []).append((real_rel, revision))

    @staticmethod
    def _key_text(real_rel: str) -> str:
        return unicodedata.normalize("NFC", real_rel)

    # --------------------------------------------------------------------- reading

    def open(self, key: SourceKey) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")
        except FileNotFoundError:
            raise _not_found(key.value, "missing") from None
        except PermissionError:
            raise _not_found(key.value, "permission_denied") from None
        except OSError:
            raise _not_found(key.value, "unreadable") from None

    def _resolve(self, key: SourceKey) -> Path:
        """Map a key to a file below the root, refusing anything that would leave it."""
        real_rel = self._paths.get(key) or self._find_real_name(key)
        if real_rel is None:
            raise _not_found(key.value, "missing")
        path = self._root.joinpath(*real_rel.split("/"))
        if not self._policy.follow_symlinks and any(
            (self._root.joinpath(*real_rel.split("/")[: i + 1])).is_symlink()
            for i in range(len(real_rel.split("/")))
        ):
            raise _not_found(key.value, "symlink_not_followed")
        root_real = os.path.realpath(self._root)
        target = os.path.realpath(path)
        if os.path.commonpath([root_real, target]) != root_real:
            raise _not_found(key.value, "outside_root")
        if not os.path.isfile(target):
            raise _not_found(key.value, "missing")
        return path

    def _find_real_name(self, key: SourceKey) -> str | None:
        """Locate a file whose on-disk name is a different Unicode form of the key."""
        current = self._root
        real_parts: list[str] = []
        for segment in key.value.split("/"):
            try:
                names = os.listdir(current)
            except OSError:
                return None
            match = next(
                (n for n in sorted(names) if unicodedata.normalize("NFC", n) == segment), None
            )
            if match is None:
                return None
            real_parts.append(match)
            current = current / match
        return "/".join(real_parts)
