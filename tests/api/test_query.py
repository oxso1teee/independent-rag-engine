"""Search and answers through the API: JSON, errors, server-sent events and refusals."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.api.conftest import Api, make_api

pytestmark = pytest.mark.api

QUESTION = "how are chunks stored in SQLite"
GOOD = "Chunks are stored in SQLite [E1]."


@pytest.fixture
def seeded(api: Api) -> Api:
    api.seed()
    return api


def events(text: str) -> list[tuple[str, dict[str, object]]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# ---------------------------------------------------------------------------- search


def test_search_returns_ranked_hits_with_provenance(seeded: Api) -> None:
    response = seeded.client.post("/v1/search", json={"query": QUESTION, "mode": "dense"})
    body = response.json()
    assert response.status_code == 200 and body["mode"] == "dense"
    first = body["hits"][0]
    assert first["rank"] == 1 and first["path"] == "alpha.md" and first["source"] == "notes"
    assert first["heading_path"] == ["Storage"] and "SQLite" in first["text"]
    assert body["index_generation"] >= 1 and body["diagnostics"] is None


def test_search_filters_narrow_the_result_and_debug_adds_diagnostics(seeded: Api) -> None:
    response = seeded.client.post(
        "/v1/search",
        json={
            "query": "rank fusion lists",
            "mode": "hybrid",
            "filters": {"path_prefix": "beta/"},
            "debug": True,
        },
    )
    body = response.json()
    assert {h["path"] for h in body["hits"]} == {"beta/ranking.md"} and body["diagnostics"]


def test_bad_filters_are_named_by_field_not_by_command_line_flag(seeded: Api) -> None:
    response = seeded.client.post(
        "/v1/search", json={"query": "x", "filters": {"language": "klingon"}}
    )
    body = response.json()
    assert response.status_code == 400 and body["code"] == "E_FILTER_INVALID"
    assert "language" in body["detail"] and "--" not in body["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "x", "top_k": 0},
        {"query": "x", "mode": "fuzzy"},
        {"query": "x", "extra": 1},
    ],
)
def test_search_validates_its_request(seeded: Api, body: dict[str, object]) -> None:
    response = seeded.client.post("/v1/search", json=body)
    assert response.status_code == 422 and response.json()["code"] == "E_QUERY_INVALID"
    assert response.json()["context"]["errors"]


def test_search_without_an_index_is_a_clear_404(api: Api) -> None:
    response = api.client.post("/v1/search", json={"query": "x"})
    assert response.status_code == 404 and response.json()["code"] == "E_SOURCE_NOT_FOUND"


# --------------------------------------------------------------------------- answers


def test_an_answer_has_status_text_citations_and_diagnostics(seeded: Api) -> None:
    seeded.chat.replies.append(GOOD)
    response = seeded.client.post(
        "/v1/answers", json={"question": QUESTION, "mode": "dense", "include_sources": True}
    )
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered" and body["text"] == GOOD
    (citation,) = body["citations"]
    assert citation["evidence_id"] == "E1" and citation["path"] == "alpha.md"
    assert body["sources"][0]["cited"] is True and body["diagnostics"]["llm_called"] is True
    assert response.headers["x-request-id"] == body["request_id"]


def test_the_models_refusal_is_a_200_with_insufficient_evidence(seeded: Api) -> None:
    seeded.chat.replies.append("<<INSUFFICIENT>>")
    body = seeded.client.post("/v1/answers", json={"question": "wifi password?"}).json()
    assert (
        body["status"] == "insufficient_evidence" and body["text"] == "" and body["citations"] == []
    )


def test_fabricated_citations_are_unverified_and_the_text_is_withheld(seeded: Api) -> None:
    seeded.chat.replies.extend(["Chunks are stored in SQLite [E9].", "Chunks live in SQLite [E9]."])
    body = seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"}).json()
    assert body["status"] == "unverified" and body["text"] == "" and body["unverified_text"] is None
    seeded.chat.replies.extend(["Chunks are stored in SQLite [E9].", "Chunks live in SQLite [E9]."])
    shown = seeded.client.post(
        "/v1/answers",
        json={"question": QUESTION, "mode": "dense", "include_unverified_text": True},
    ).json()
    assert shown["unverified_text"] == "Chunks live in SQLite [E9]."


def test_a_provider_failure_is_a_problem_with_the_engine_code(seeded: Api) -> None:
    seeded.chat.replies.append(lambda: httpx.Response(404, text="model not found"))
    response = seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"})
    body = response.json()
    assert response.status_code == 502 and body["code"] == "E_PROVIDER_UNAVAILABLE"
    assert body["context"]["stage"] == "generation" and body["retryable"] is False


def test_without_a_language_model_search_works_and_answers_say_why_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for api in make_api(tmp_path, monkeypatch, llm=False):
        api.seed()
        assert api.client.post("/v1/search", json={"query": QUESTION}).status_code == 200
        response = api.client.post("/v1/answers", json={"question": QUESTION})
        assert response.status_code == 503 and response.json()["code"] == "E_CONFIG_INVALID"


# ------------------------------------------------------------------------------ SSE


def test_a_stream_is_evidence_then_the_validated_text_then_citations_then_final(
    seeded: Api,
) -> None:
    seeded.chat.replies.append(GOOD)
    response = seeded.client.post(
        "/v1/answers", json={"question": QUESTION, "mode": "dense", "stream": True}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    parsed = events(response.text)
    names = [n for n, _ in parsed]
    assert names[0] == "evidence" and names[-2:] == ["citations", "final"]
    deltas = [d for n, d in parsed if n == "delta"]
    assert len(deltas) == 1 and deltas[0]["provisional"] is False and deltas[0]["text"] == GOOD
    final = parsed[-1][1]
    assert final["status"] == "answered" and final["text"] == GOOD and final["citations"]


def test_a_live_stream_marks_drafts_provisional_and_retracts_a_failed_one(seeded: Api) -> None:
    seeded.chat.replies.append("Chunks are stored in SQLite [E9].")
    response = seeded.client.post(
        "/v1/answers", json={"question": QUESTION, "mode": "dense", "stream": True, "live": True}
    )
    parsed = events(response.text)
    names = [n for n, _ in parsed]
    assert "retract" in names and names[-1] == "final"
    assert all(d["provisional"] for n, d in parsed if n == "delta")
    assert parsed[-1][1]["status"] == "unverified" and parsed[-1][1]["text"] == ""


def test_a_refusal_stream_carries_no_text_and_a_stream_failure_is_a_final_event(
    seeded: Api,
) -> None:
    seeded.chat.replies.append("<<INSUFFICIENT>>")
    refusal = seeded.client.post(
        "/v1/answers", json={"question": "wifi?", "mode": "dense", "stream": True}
    )
    names = [n for n, _ in events(refusal.text)]
    assert "delta" not in names and names[-1] == "final"
    seeded.chat.replies.append(lambda: httpx.Response(404, text="model not found"))
    failed = seeded.client.post(
        "/v1/answers", json={"question": QUESTION, "mode": "dense", "stream": True}
    )
    final = events(failed.text)[-1][1]
    error = final["error"]
    assert isinstance(error, dict) and failed.status_code == 200 and final["status"] == "error"
    assert error["code"] == "E_PROVIDER_UNAVAILABLE"


def test_the_stream_and_the_plain_answer_agree(seeded: Api) -> None:
    seeded.chat.replies.extend([GOOD, GOOD])
    plain = seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"}).json()
    streamed = events(
        seeded.client.post(
            "/v1/answers", json={"question": QUESTION, "mode": "dense", "stream": True}
        ).text
    )[-1][1]
    assert streamed["text"] == plain["text"] and streamed["citations"] == plain["citations"]


def test_requests_are_independent_across_a_burst(seeded: Api) -> None:
    seeded.chat.replies.extend([GOOD] * 4)
    codes = [
        seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"}).status_code
        for _ in range(4)
    ]
    assert codes == [200] * 4 and len(seeded.chat.bodies) == 4


def test_the_query_services_are_opened_once_and_reused(seeded: Api) -> None:
    seeded.chat.replies.extend([GOOD, GOOD])
    state = seeded.client.app.state.api  # type: ignore[attr-defined]
    seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"})
    first = state._backend
    seeded.client.post("/v1/answers", json={"question": QUESTION, "mode": "dense"})
    assert state._backend is first


def _unused(_: Iterator[Api]) -> None:  # pragma: no cover - keeps the import used
    return None
