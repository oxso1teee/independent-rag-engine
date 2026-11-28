"""A small corpus and question files for calibration tests: separable on purpose.

Every document is about its own invented subsystem; an answerable question names that subsystem,
an unanswerable one asks about something that no document mentions.
"""

from __future__ import annotations

import json
from pathlib import Path

_SYLLABLES = ["bar", "kel", "dro", "fim", "gul", "han", "jor", "lun", "mep", "nax", "pyr", "quo"]
COUNT = 48


def _word(i: int, salt: int) -> str:
    a, b, c = _SYLLABLES[i % 12], _SYLLABLES[(i // 12 + salt) % 12], _SYLLABLES[(i * 5 + salt) % 12]
    return f"{a}{b}{c}{i}"


def documents() -> dict[str, str]:
    return {
        f"topic{i}.md": (
            f"# {_word(i, 1).title()}\n\nThe {_word(i, 1)} subsystem relies on the {_word(i, 2)} "
            f"protocol to synchronise {_word(i, 3)} records between nodes.\n"
        )
        for i in range(COUNT)
    }


def answerable(i: int) -> str:
    return (
        f"Which protocol does the {_word(i, 1)} subsystem rely on to synchronise "
        f"{_word(i, 3)} records?"
    )


def unanswerable(i: int) -> str:
    return f"How tall is the mountain {_word(i, 7)} in the northern glacier range?"


def write_questions(path: Path, rows: range, *, offset: int = 0) -> Path:
    """JSON Lines with an answerable and an unanswerable question per index in ``rows``."""
    lines = []
    for i in rows:
        lines.append({"question": answerable(i), "answerable": True})
        lines.append({"question": unanswerable(i + offset), "answerable": False})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path
