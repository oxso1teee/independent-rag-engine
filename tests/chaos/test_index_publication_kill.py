"""SIGKILL during FTS publication leaves no partial visible version and can be retried."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rag_engine.storage.integrity import check_integrity
from rag_engine.storage.sqlite import open_store_read_only

REPO = Path(__file__).resolve().parents[2]


def child(database: Path, mode: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.chaos.publication_child", str(database), mode],
        cwd=REPO,
        capture_output=True,
        check=False,
    )


@pytest.mark.chaos
def test_kill_mid_publication_rolls_back_and_retry_publishes(tmp_path: Path) -> None:
    database = tmp_path / "rag.db"
    assert child(database, "init").returncode == 0
    with open_store_read_only(database) as store:
        assert store.reader().get_index_metadata().index_generation == 0  # type: ignore[union-attr]
    killed = child(database, "kill")
    assert killed.returncode == -9, killed.stderr.decode()
    assert check_integrity(database).ok
    with open_store_read_only(database) as store:
        metadata = store.reader().get_index_metadata()
        assert metadata is not None and metadata.index_generation == 0
        assert metadata.active_embedding_model_id is None
        assert not store.reader().list_embedding_models()
    resumed = child(database, "finish")
    assert resumed.returncode == 0, resumed.stderr.decode()
    report = check_integrity(database)
    assert report.ok and report.published_chunks == 1 and report.index_generation == 1
