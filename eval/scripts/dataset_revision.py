"""Make a new dataset version from the previous one and record exactly what changed and why.

A dataset version is frozen once people may have reviewed it, so a correction is a *new* version.
``revise`` compares the new questions with the previous version, refuses a change that has no stated
reason (and a reason for something that did not change), keeps the previous provenance and review
of every question that is untouched, marks a changed question as revised in its provenance, and
never changes a review: every question of the new version is still ``unreviewed``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_UNTRACKED = ("provenance", "review")


def revise(
    previous: Mapping[str, Any],
    current: dict[str, Any],
    reasons: Mapping[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Update ``current`` in place (provenance of untouched and changed questions) and return the
    list of changes. ``reasons`` maps a question id to ``(kind, reason)``."""
    version = current["version"]
    old = {q["id"]: q for q in previous["questions"]}
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in current["questions"]:
        before = old.get(question["id"])
        seen.add(question["id"])
        if before is None:
            fields = [k for k in question if k not in _UNTRACKED]
        else:
            fields = [k for k in question if k not in _UNTRACKED and question[k] != before.get(k)]
            if not fields:
                question["provenance"] = before["provenance"]
                question["review"] = before["review"]
                continue
        kind, reason = _reason(reasons, question["id"])
        provenance = dict(before["provenance"] if before else question["provenance"])
        provenance["method"] = f"revised in v{version} ({kind}): {reason}; awaiting human review"
        question["provenance"] = provenance
        question["review"] = {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}
        changes.append(
            {
                "id": question["id"],
                "kind": "added" if before is None else kind,
                "fields": fields,
                "reason": reason,
                "before": {k: before.get(k) for k in fields} if before else None,
                "after": {k: question[k] for k in fields},
            }
        )
    for identifier in old.keys() - seen:
        kind, reason = _reason(reasons, identifier)
        changes.append({"id": identifier, "kind": "removed", "fields": [], "reason": reason,
                        "before": None, "after": None})  # fmt: skip
    unexplained = set(reasons) - {c["id"] for c in changes}
    if unexplained:
        raise SystemExit(f"reasons for questions that did not change: {sorted(unexplained)}")
    return changes


def _reason(reasons: Mapping[str, tuple[str, str]], identifier: str) -> tuple[str, str]:
    if identifier not in reasons:
        raise SystemExit(f"{identifier} changed without a stated reason; add it to REASONS")
    return reasons[identifier]


def record(path: Path, name: str, previous: int, current: int, date: str,
           changes: Sequence[Mapping[str, Any]]) -> None:  # fmt: skip
    """Merge one dataset's changes into the shared history file, keeping the other datasets'."""
    data: dict[str, Any] = {"revision": current, "date": date, "datasets": {}}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["datasets"][name] = {"from": previous, "to": current, "changes": list(changes)}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
