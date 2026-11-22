"""Glob rules for include/exclude patterns."""

from __future__ import annotations

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.ingestion.sources.globbing import GlobSet, compile_glob


@pytest.mark.parametrize(
    ("pattern", "key", "expected"),
    [
        ("**/*", "a.md", True),
        ("**/*", "deep/er/a.md", True),
        ("*.md", "a.md", True),
        ("*.md", "deep/er/a.md", True),  # no slash: matches the name at any depth
        ("*.md", "a.txt", False),
        ("docs/*.md", "docs/a.md", True),
        ("docs/*.md", "docs/sub/a.md", False),
        ("docs/**", "docs/sub/a.md", True),
        ("docs/**/a.md", "docs/a.md", True),
        ("docs/**/a.md", "docs/x/y/a.md", True),
        ("a?.md", "ab.md", True),
        ("a?.md", "a/.md", False),
        ("[ab].md", "a.md", True),
        ("[!ab].md", "a.md", False),
        ("[!ab].md", "c.md", True),
        ("node_modules/", "node_modules/x.js", True),
        ("Папка/*.md", "Папка/файл.md", True),
        ("файл.md", "Папка/файл.md", True),
        ("+(x).md", "+(x).md", True),  # regex metacharacters are literals
        ("a.md", "ab.md", False),
    ],
)
def test_matching(pattern: str, key: str, expected: bool) -> None:
    assert bool(compile_glob(pattern).fullmatch(key)) is expected


def test_patterns_are_nfc_normalised() -> None:
    decomposed = "ё.md"  # ё as e + combining diaeresis
    assert compile_glob(decomposed).fullmatch("ё.md")


@pytest.mark.parametrize(
    "pattern",
    ["", "   ", "/abs/*.md", "../x", "a/../b", "a\\b", "[abc", "[]", "[a/b]", "/", "a\x00"],
)
def test_invalid_patterns_are_rejected_with_a_clear_error(pattern: str) -> None:
    with pytest.raises(RagError) as info:
        compile_glob(pattern)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_glob_set_combines_include_and_exclude() -> None:
    rules = GlobSet(["**/*.md", "*.txt"], ["drafts/**", "*.tmp.md"])
    assert rules.selects("a.md")
    assert rules.selects("x/y/z.txt")
    assert not rules.selects("drafts/a.md")
    assert not rules.selects("a.tmp.md")
    assert not rules.selects("a.png")


def test_directories_are_pruned_by_exclude_rules() -> None:
    rules = GlobSet(["**/*"], ["node_modules", ".git/**", "build/"])
    assert rules.prunes_directory("node_modules")
    assert rules.prunes_directory("pkg/node_modules")
    assert rules.prunes_directory(".git")
    assert rules.prunes_directory("build")
    assert not rules.prunes_directory("src")
