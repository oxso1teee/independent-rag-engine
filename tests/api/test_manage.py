"""Sources, documents, jobs, health and statistics through the API."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from tests.api.conftest import Api

pytestmark = pytest.mark.api

NOTE = "Backups run every night at 02:30 UTC and are kept for 14 days.\n"


def problem_code(response: Any) -> str:
    assert response.headers["content-type"].startswith("application/problem+json")
    return str(response.json()["code"])


def add_note(api: Api, key: str = "ops/backups.md", text: str = NOTE, **extra: Any) -> Any:
    return api.client.post(
        "/v1/documents",
        json={"source": "uploads", "key": key, "text": text},
        **extra,
    )


@pytest.fixture
def uploads(api: Api) -> Api:
    assert api.client.post("/v1/sources", json={"name": "uploads"}).status_code == 201
    return api


def test_health_reports_ok_and_the_version(api: Api) -> None:
    body = api.client.get("/v1/health").json()
    assert body["status"] == "ok" and body["version"]
    assert api.client.get("/v1/health").headers["x-request-id"]


def test_readiness_follows_the_index(api: Api) -> None:
    before = api.client.get("/v1/ready")
    assert before.status_code == 503 and before.json()["ready"] is False
    api.seed()
    after = api.client.get("/v1/ready")
    assert after.status_code == 200 and after.json()["checks"]["index_published"] is True


def test_stats_have_counts_and_no_server_paths(api: Api) -> None:
    assert api.client.get("/v1/stats").json()["sources"] == []
    api.seed()
    body = api.client.get("/v1/stats").json()
    assert body["sources"][0]["name"] == "notes" and body["index_generation"] >= 1
    assert (
        "locator" not in body["sources"][0]
        and str(api.root) not in api.client.get("/v1/stats").text
    )


def test_a_managed_source_can_be_created_twice_and_has_no_locator(api: Api) -> None:
    first = api.client.post("/v1/sources", json={"name": "uploads"})
    again = api.client.post("/v1/sources", json={"name": "uploads"})
    assert first.status_code == 201 and again.status_code == 201
    assert first.json()["managed"] is True and "locator" not in first.json()
    assert [s["name"] for s in api.client.get("/v1/sources").json()] == ["uploads"]
    assert api.client.get("/v1/sources/uploads").json()["id"] == first.json()["id"]


def test_source_names_are_validated_and_unknown_sources_are_404(api: Api) -> None:
    assert api.client.post("/v1/sources", json={"name": "../etc"}).status_code == 422
    assert api.client.post("/v1/sources", json={"name": "A B"}).status_code == 422
    missing = api.client.get("/v1/sources/nope")
    assert missing.status_code == 404 and problem_code(missing) == "E_SOURCE_NOT_FOUND"


def test_an_uploaded_document_is_stored_indexed_and_listed(uploads: Api) -> None:
    response = add_note(uploads)
    assert response.status_code == 202
    body = response.json()
    assert body["indexing"] == "scheduled" and body["document"]["key"] == "ops/backups.md"
    listed = uploads.client.get("/v1/documents").json()["items"]
    assert [d["key"] for d in listed] == ["ops/backups.md"] and listed[0]["indexed"] is True
    detail = uploads.client.get(f"/v1/documents/{listed[0]['id']}").json()
    assert detail["chunks"] >= 1 and detail["versions"][0]["seq"] == 1
    assert detail["versions"][0]["size_bytes"] == len(NOTE)


def test_a_changed_upload_makes_a_new_version_and_the_search_sees_it(uploads: Api) -> None:
    add_note(uploads)
    add_note(uploads, text="Printers are on the third floor next to the kitchen.\n")
    (document,) = uploads.client.get("/v1/documents").json()["items"]
    detail = uploads.client.get(f"/v1/documents/{document['id']}").json()
    assert [v["seq"] for v in detail["versions"]] == [1, 2]
    hits = uploads.client.post("/v1/search", json={"query": "printers kitchen", "mode": "lexical"})
    assert hits.json()["hits"][0]["path"] == "ops/backups.md"


def test_base64_content_is_accepted_and_invalid_base64_is_refused(uploads: Api) -> None:
    good = base64.b64encode(NOTE.encode()).decode()
    ok = uploads.client.post(
        "/v1/documents", json={"source": "uploads", "key": "b.md", "content_base64": good}
    )
    assert ok.status_code == 202
    bad = uploads.client.post(
        "/v1/documents", json={"source": "uploads", "key": "c.md", "content_base64": "!!!"}
    )
    assert bad.status_code == 400 and problem_code(bad) == "E_QUERY_INVALID"


def test_exactly_one_content_field_is_required(uploads: Api) -> None:
    for body in ({}, {"text": "a", "content_base64": "YQ=="}):
        response = uploads.client.post(
            "/v1/documents", json={"source": "uploads", "key": "d.md", **body}
        )
        assert response.status_code == 422 and problem_code(response) == "E_QUERY_INVALID"


@pytest.mark.parametrize("key", ["../escape.md", "/etc/passwd.md", ".hidden.md", "a b.md", ""])
def test_keys_that_could_leave_the_source_are_refused(uploads: Api, key: str) -> None:
    assert add_note(uploads, key).status_code == 422


@pytest.mark.parametrize("key", ["a/../b.md", "a//b.md", "a/./b.md"])
def test_unclean_paths_inside_a_key_are_refused(uploads: Api, key: str) -> None:
    response = add_note(uploads, key)
    assert response.status_code in {400, 422}
    assert not list((uploads.root / "data" / "inline").rglob("b.md"))


def test_an_unsupported_extension_and_an_empty_document_are_refused(uploads: Api) -> None:
    exe = add_note(uploads, "tool.exe")
    assert exe.status_code == 415 and problem_code(exe) == "E_UNSUPPORTED_FORMAT"
    assert add_note(uploads, "empty.md", "").status_code == 400


def test_documents_can_only_be_written_into_managed_sources(api: Api) -> None:
    api.seed()  # "notes" was registered from the command line
    response = api.client.post(
        "/v1/documents", json={"source": "notes", "key": "x.md", "text": NOTE}
    )
    assert response.status_code == 409 and problem_code(response) == "E_SOURCE_LOCATOR_CHANGED"
    assert not (api.root / "notes" / "x.md").exists()


def test_a_retry_with_the_same_key_replays_and_a_different_body_conflicts(uploads: Api) -> None:
    headers = {"Idempotency-Key": "upload-1"}
    first = add_note(uploads, headers=headers)
    replay = add_note(uploads, headers=headers)
    assert first.status_code == replay.status_code == 202 and first.json() == replay.json()
    assert replay.headers["idempotent-replay"] == "true"
    assert "idempotent-replay" not in first.headers
    clash = add_note(uploads, text="Another text.\n", headers=headers)
    assert clash.status_code == 409 and problem_code(clash) == "E_IDEMPOTENCY_CONFLICT"
    assert uploads.client.get("/v1/documents").json()["items"][0]["key"] == "ops/backups.md"


def test_a_replay_does_not_write_the_file_again(uploads: Api) -> None:
    headers = {"Idempotency-Key": "k"}
    add_note(uploads, headers=headers)
    path = uploads.root / "data" / "inline" / "uploads" / "ops" / "backups.md"
    path.write_text("changed behind the API\n")
    add_note(uploads, headers=headers)
    assert path.read_text() == "changed behind the API\n"


def test_listing_is_paginated_with_an_opaque_cursor(uploads: Api) -> None:
    for name in ("a.md", "b.md", "c.md"):
        add_note(uploads, name, f"Document {name} has some text in it.\n")
    first = uploads.client.get("/v1/documents", params={"limit": 2}).json()
    assert [d["key"] for d in first["items"]] == ["a.md", "b.md"] and first["next_cursor"]
    second = uploads.client.get(
        "/v1/documents", params={"limit": 2, "cursor": first["next_cursor"]}
    ).json()
    assert [d["key"] for d in second["items"]] == ["c.md"] and second["next_cursor"] is None
    bad = uploads.client.get("/v1/documents", params={"cursor": "###"})
    assert bad.status_code == 400 and problem_code(bad) == "E_QUERY_INVALID"
    assert uploads.client.get("/v1/documents", params={"limit": 0}).status_code == 422
    assert uploads.client.get("/v1/documents", params={"limit": 1000}).status_code == 422


def test_ingesting_a_source_queues_work_and_jobs_can_be_read(api: Api) -> None:
    api.seed()
    response = api.client.post("/v1/sources/notes/ingest")
    assert response.status_code == 202 and response.json()["ingest"]
    jobs = api.client.get("/v1/jobs", params={"limit": 2}).json()
    assert len(jobs["items"]) == 2 and jobs["next_cursor"]
    job = api.client.get(f"/v1/jobs/{jobs['items'][0]['id']}").json()
    assert job["source"] == "notes" and job["status"] in {"queued", "leased", "done", "dead"}
    assert api.client.get("/v1/jobs/not-an-id").status_code == 400
    assert api.client.get("/v1/jobs", params={"status": "bogus"}).status_code == 422


def test_deleting_a_managed_document_removes_it_from_the_index(uploads: Api) -> None:
    add_note(uploads)
    (document,) = uploads.client.get("/v1/documents").json()["items"]
    assert uploads.client.delete(f"/v1/documents/{document['id']}").status_code == 202
    assert not (uploads.root / "data" / "inline" / "uploads" / "ops" / "backups.md").exists()
    hits = uploads.client.post("/v1/search", json={"query": "backups nightly", "mode": "lexical"})
    assert hits.json()["hits"] == []
    assert uploads.client.delete(f"/v1/documents/{document['id']}").status_code in {202, 404}


def test_unknown_documents_and_malformed_ids_are_reported(api: Api) -> None:
    api.seed()
    assert api.client.get("/v1/documents/not-an-id").status_code == 400
    unknown = api.client.get("/v1/documents/01a21dc5-d6ef-7117-af12-1365cf7451a5")
    assert unknown.status_code == 404


def test_errors_are_problem_json_with_the_request_id(api: Api) -> None:
    response = api.client.get("/v1/nothing-here")
    body = response.json()
    assert response.status_code == 404 and problem_code(response) == "E_NOT_FOUND"
    assert (
        body["instance"] == "/v1/nothing-here"
        and body["request_id"] == response.headers["x-request-id"]
    )
    assert api.client.put("/v1/sources").status_code == 405
