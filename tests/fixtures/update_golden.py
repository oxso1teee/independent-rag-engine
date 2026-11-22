"""Rewrites the stored golden snapshot of ``corpus_mini``.

Run ``uv run python tests/fixtures/update_golden.py`` after a deliberate change to parsing,
normalisation or chunking, then review the diff of ``tests/fixtures/golden/corpus_mini.json``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from tests.corpus_snapshot import GOLDEN, dump_snapshot, snapshot_all

from rag_engine.observability import configure_logging


def main() -> None:
    configure_logging("ERROR", stream=io.StringIO())
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(dump_snapshot(snapshot_all()), encoding="utf-8")


if __name__ == "__main__":
    main()
