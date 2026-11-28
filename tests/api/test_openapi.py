"""The OpenAPI document is a committed snapshot; changing the API means changing it in review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.api.openapi import build_document, render

pytestmark = pytest.mark.api

SNAPSHOT = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"


def test_the_snapshot_matches_the_generated_document() -> None:
    assert SNAPSHOT.read_text(encoding="utf-8") == render(), (
        "the API changed: regenerate the snapshot with "
        "`uv run python -m rag_engine.api.openapi docs/openapi.json` and review the diff"
    )


def test_the_contract_exposes_the_specified_endpoints() -> None:
    paths = build_document()["paths"]
    expected = {
        ("get", "/v1/health"), ("get", "/v1/ready"), ("get", "/v1/stats"),
        ("get", "/v1/sources"), ("post", "/v1/sources"), ("get", "/v1/sources/{name}"),
        ("post", "/v1/sources/{name}/ingest"),
        ("get", "/v1/documents"), ("post", "/v1/documents"),
        ("get", "/v1/documents/{document_id}"), ("delete", "/v1/documents/{document_id}"),
        ("get", "/v1/jobs"), ("get", "/v1/jobs/{job_id}"),
        ("post", "/v1/search"), ("post", "/v1/answers"),
    }  # fmt: skip
    assert {(m, p) for p, ops in paths.items() for m in ops} == expected


def test_every_operation_but_health_is_secured_and_documents_its_errors() -> None:
    document = build_document()
    for path, operations in document["paths"].items():
        for method, operation in operations.items():
            if path == "/v1/health":
                assert "security" not in operation
                continue
            assert operation["security"] == [{"HTTPBearer": []}], (method, path)
            assert "401" in operation["responses"] and "422" in operation["responses"]
            problem = operation["responses"]["401"]["content"]["application/problem+json"]
            assert problem is not None


def test_the_answers_endpoint_documents_server_sent_events() -> None:
    answers = build_document()["paths"]["/v1/answers"]["post"]
    assert "text/event-stream" in answers["responses"]["200"]["content"]


def test_post_documents_accepts_an_idempotency_key_header() -> None:
    operation = build_document()["paths"]["/v1/documents"]["post"]
    names = {p["name"] for p in operation.get("parameters", [])}
    assert "Idempotency-Key" in names


def test_the_snapshot_is_valid_json_with_a_version() -> None:
    document = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert document["openapi"].startswith("3.") and document["info"]["version"]
