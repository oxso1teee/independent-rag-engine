"""Every relative link in the Markdown documentation points at something that exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
SKIP_SCHEMES = ("http://", "https://", "mailto:", "#")
IGNORED = {"CLAUDE.md"}  # a local file, not part of the repository


def documents() -> list[Path]:
    found = [p for p in ROOT.glob("*.md") if p.name not in IGNORED]
    found += list((ROOT / "docs").rglob("*.md"))
    found += [ROOT / "baselines" / "README.md"]
    return sorted(found)


def test_every_relative_link_resolves() -> None:
    broken = []
    for document in documents():
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if target.startswith(SKIP_SCHEMES):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_the_documents_the_readme_promises_exist() -> None:
    for name in ("CONTRIBUTING.md", "SECURITY.md", "PROVENANCE.md", "THIRD_PARTY_NOTICES.md",
                 "sbom.cdx.json", "CHANGELOG.md"):  # fmt: skip
        assert (ROOT / name).is_file(), name
