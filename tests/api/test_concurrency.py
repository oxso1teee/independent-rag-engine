"""Management requests, a writer and queries together: no crash, only documented outcomes."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.api.conftest import Api, make_api

pytestmark = pytest.mark.api

ALLOWED = {200, 202, 503}  # 503 is the documented, retryable "store is locked"


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Api]:
    for api in make_api(tmp_path, monkeypatch):
        assert api.client.post("/v1/sources", json={"name": "uploads"}).status_code == 201
        yield api


def test_readers_and_a_writer_in_parallel_never_cause_a_server_error(uploads: Api) -> None:
    statuses: list[int] = []
    problems: list[str] = []
    lock = threading.Lock()

    def reader(path: str) -> None:
        for _ in range(12):
            response = uploads.client.get(path)
            with lock:
                statuses.append(response.status_code)
                if response.status_code == 503:
                    problems.append(response.json()["code"])

    def writer() -> None:
        for n in range(6):
            response = uploads.client.post(
                "/v1/documents",
                json={"source": "uploads", "key": f"doc{n}.md", "text": f"Document number {n}.\n"},
            )
            with lock:
                statuses.append(response.status_code)
                if response.status_code == 503:
                    problems.append(response.json()["code"])

    threads = [
        threading.Thread(target=reader, args=(p,))
        for p in ("/v1/stats", "/v1/documents", "/v1/jobs", "/v1/sources")
    ]
    threads.append(threading.Thread(target=writer))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads)
    assert set(statuses) <= ALLOWED, sorted(set(statuses))
    assert set(problems) <= {"E_STORE_LOCKED"}
    final = uploads.client.get("/v1/documents", params={"limit": 50}).json()["items"]
    assert len(final) == 6  # every accepted write is there, none twice


def test_a_burst_of_identical_idempotent_uploads_writes_the_document_once(uploads: Api) -> None:
    headers = {"Idempotency-Key": "burst"}
    body = {"source": "uploads", "key": "once.md", "text": "Only one copy of this text.\n"}
    codes: list[int] = []

    def send() -> None:
        codes.append(uploads.client.post("/v1/documents", json=body, headers=headers).status_code)

    threads = [threading.Thread(target=send) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert set(codes) <= ALLOWED
    documents = uploads.client.get("/v1/documents").json()["items"]
    assert [d["key"] for d in documents] == ["once.md"]
