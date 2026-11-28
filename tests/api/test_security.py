"""Authentication, the public-bind rule, size limits and CORS."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_engine.api.app import resolve_token
from rag_engine.config.settings import ApiSettings
from rag_engine.domain import ErrorCode, RagError
from tests.api.conftest import TOKEN, Api, make_api

pytestmark = pytest.mark.api


@pytest.fixture
def secured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Api]:
    yield from make_api(tmp_path, monkeypatch, token=TOKEN)


def test_health_is_open_and_everything_else_needs_the_token(secured: Api) -> None:
    anonymous = secured.client.__class__(secured.client.app)
    assert anonymous.get("/v1/health").status_code == 200
    for path in ("/v1/stats", "/v1/sources", "/v1/documents", "/v1/jobs", "/v1/ready"):
        response = anonymous.get(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["content-type"].startswith("application/problem+json")
    assert anonymous.post("/v1/search", json={"query": "x"}).status_code == 401
    assert anonymous.post("/v1/answers", json={"question": "x"}).status_code == 401


def test_a_wrong_token_is_refused_and_the_right_one_accepted(secured: Api) -> None:
    secured.client.headers["Authorization"] = "Bearer " + "x" * 32
    assert secured.client.get("/v1/sources").status_code == 401
    secured.client.headers["Authorization"] = "Basic " + TOKEN
    assert secured.client.get("/v1/sources").status_code == 401
    secured.client.headers["Authorization"] = "Bearer " + TOKEN
    assert secured.client.get("/v1/sources").status_code == 200


def test_the_token_never_appears_in_a_response(secured: Api) -> None:
    secured.client.headers["Authorization"] = "Bearer wrong-" + TOKEN
    text = secured.client.get("/v1/sources").text
    assert TOKEN not in text


def test_a_loopback_server_without_a_token_is_open(api: Api) -> None:
    assert api.client.get("/v1/sources").status_code == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.org", "::"])  # noqa: S104
def test_a_public_bind_without_a_token_is_refused_when_serving(host: str) -> None:
    settings = ApiSettings(host=host)  # the default names RAG_API_TOKEN
    with pytest.raises(RagError) as info:
        resolve_token(settings, {})
    assert info.value.code is ErrorCode.E_CONFIG_INVALID and "non-loopback" in info.value.message
    assert resolve_token(settings, {"RAG_API_TOKEN": TOKEN}) == TOKEN


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.org", "::"])  # noqa: S104
def test_authentication_cannot_be_switched_off_on_a_public_bind(host: str) -> None:
    with pytest.raises(ValueError, match="token_env"):
        ApiSettings(host=host, token_env=None)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_binds_need_no_token(host: str) -> None:
    assert resolve_token(ApiSettings(host=host), {}) is None
    assert ApiSettings(host=host, token_env=None).token_env is None


def test_setting_the_token_variable_switches_authentication_on_by_default() -> None:
    assert resolve_token(ApiSettings(), {"RAG_API_TOKEN": TOKEN}) == TOKEN


def test_a_weak_token_is_refused_everywhere() -> None:
    for host in ("127.0.0.1", "0.0.0.0"):  # noqa: S104
        with pytest.raises(RagError) as info:
            resolve_token(ApiSettings(host=host), {"RAG_API_TOKEN": "short"})
        assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_a_custom_variable_name_is_honoured() -> None:
    settings = ApiSettings(token_env="MY_TOKEN")
    assert resolve_token(settings, {"RAG_API_TOKEN": TOKEN}) is None
    assert resolve_token(settings, {"MY_TOKEN": TOKEN}) == TOKEN


def test_a_body_over_the_limit_is_refused_with_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for api in make_api(tmp_path, monkeypatch, api="[api]\nmax_body_bytes = 2048\n"):
        response = api.client.post("/v1/search", json={"query": "x" * 5000})
        assert response.status_code == 413
        assert response.json()["code"] == "E_BODY_TOO_LARGE"
        assert response.headers["content-type"].startswith("application/problem+json")


def test_a_long_question_is_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for api in make_api(tmp_path, monkeypatch, api="[api]\nmax_question_chars = 20\n"):
        response = api.client.post("/v1/answers", json={"question": "q" * 21})
        assert response.status_code == 400 and response.json()["code"] == "E_QUERY_INVALID"


def test_cors_is_off_by_default_and_on_only_for_listed_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = {"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    for api in make_api(tmp_path, monkeypatch):
        response = api.client.options("/v1/sources", headers=headers)
        assert "access-control-allow-origin" not in response.headers
    other = tmp_path / "other"
    other.mkdir()
    for api in make_api(other, monkeypatch, api='[api]\ncors_origins = ["https://app.example"]\n'):
        allowed = api.client.options(
            "/v1/sources",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "GET"},
        )
        denied = api.client.options("/v1/sources", headers=headers)
        assert allowed.headers["access-control-allow-origin"] == "https://app.example"
        assert "access-control-allow-origin" not in denied.headers


# ------------------------------------------------------------------------------- metrics


def test_metrics_are_off_by_default(api: Api) -> None:
    assert api.client.get("/v1/metrics").status_code == 404


def test_metrics_count_requests_by_route_template_and_never_by_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for api in make_api(tmp_path, monkeypatch, api="[api]\nmetrics = true\n"):
        api.client.get("/v1/health")
        api.client.get("/v1/documents/01a21dc5-d6ef-7117-af12-1365cf7451a5")
        api.client.get("/v1/nothing-here")
        text = api.client.get("/v1/metrics")
        body = text.text
        assert text.status_code == 200 and text.headers["content-type"].startswith("text/plain")
        assert 'rag_http_requests_total{method="GET",route="/v1/health",status="2xx"} 1' in body
        assert 'route="/v1/documents/{document_id}",status="4xx"' in body
        assert "01a21dc5" not in body and "nothing-here" not in body  # no ids, no raw paths
        assert 'route="unmatched"' in body and "rag_uptime_seconds" in body


def test_metrics_need_the_token_when_one_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for api in make_api(tmp_path, monkeypatch, api="[api]\nmetrics = true\n", token=TOKEN):
        anonymous = api.client.__class__(api.client.app)
        assert anonymous.get("/v1/metrics").status_code == 401
        assert api.client.get("/v1/metrics").status_code == 200
