from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.config import (
    PROFILE_LIMITS,
    ResourceProfile,
    Settings,
    load_settings,
    read_secret,
)
from rag_engine.domain import ErrorCategory, ErrorCode, RagError


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rag.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_use_the_low_profile() -> None:
    settings = load_settings()
    assert settings.profile is ResourceProfile.LOW
    assert settings.effective_limits() == PROFILE_LIMITS[ResourceProfile.LOW]
    assert settings.effective_limits().llm_concurrency == 1
    assert settings.llm.enabled is False
    assert settings.embedding.enabled is False


def test_toml_file_is_loaded(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
profile = "low"

[storage]
db_path = "var/index.db"

[llm]
enabled = true
provider = "local"
model = "chat-small"
context_window = 8192
max_new_tokens = 1024

[embedding]
enabled = true
provider = "local"
model = "embed-small"

[logging]
level = "DEBUG"
""",
    )
    settings = load_settings(path)
    assert settings.profile is ResourceProfile.LOW
    assert settings.storage.db_path == Path("var/index.db")
    assert settings.llm.context_window == 8192
    assert settings.embedding.model == "embed-small"
    assert settings.logging.level == "DEBUG"


def test_environment_overrides_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(
        tmp_path, '[llm]\nenabled = true\nprovider = "local"\nmodel = "m"\ncontext_window = 4096\n'
    )
    monkeypatch.setenv("RAG_LLM__CONTEXT_WINDOW", "16384")
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "DEBUG")
    settings = load_settings(path)
    assert settings.llm.context_window == 16384
    assert settings.llm.provider == "local"  # merged, not replaced
    assert settings.logging.level == "DEBUG"


def test_explicit_limits_override_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LIMITS__EMBED_CONCURRENCY", "3")
    monkeypatch.setenv("RAG_LIMITS__INGEST_WORKERS", "2")
    limits = load_settings().effective_limits()
    assert (limits.embed_concurrency, limits.ingest_workers) == (3, 2)
    assert limits.llm_concurrency == PROFILE_LIMITS[ResourceProfile.LOW].llm_concurrency


def test_settings_are_immutable() -> None:
    settings = load_settings()
    with pytest.raises(ValueError, match="frozen"):
        settings.profile = ResourceProfile.LOW  # type: ignore[misc]


def _fails(tmp_path: Path, body: str) -> RagError:
    with pytest.raises(RagError) as info:
        load_settings(write(tmp_path, body))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert info.value.category is ErrorCategory.CONFIG
    return info.value


def test_enabled_llm_requires_context_window(tmp_path: Path) -> None:
    error = _fails(tmp_path, '[llm]\nenabled = true\nprovider = "local"\nmodel = "m"\n')
    assert "llm" in error.message
    assert "context_window" in error.message


def test_enabled_embedding_requires_a_model(tmp_path: Path) -> None:
    error = _fails(tmp_path, '[embedding]\nenabled = true\nprovider = "local"\n')
    assert "embedding.model" in error.message


def test_max_new_tokens_must_fit_the_window(tmp_path: Path) -> None:
    error = _fails(
        tmp_path,
        '[llm]\nenabled = true\nprovider = "p"\nmodel = "m"\ncontext_window = 1024\n'
        "max_new_tokens = 2048\n",
    )
    assert "max_new_tokens" in error.message


@pytest.mark.parametrize(
    "body",
    [
        'profile = "huge"',
        'profile = "standard"',
        "[limits]\nllm_concurrency = 0",
        "[limits]\nembed_batch = 100000",
        '[llm]\nprovider = "Bad Name"',
        '[llm]\napi_key_env = "lowercase"',
        '[logging]\nlevel = "LOUD"',
        "[llm]\ncontext_window = 10",
        "unknown_key = 1",
        "[storage]\nunknown = 1",
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, body: str) -> None:
    _fails(tmp_path, body)


def test_secrets_in_the_file_are_rejected_with_guidance(tmp_path: Path) -> None:
    error = _fails(tmp_path, '[llm]\napi_key = "sk-super-secret-value"\n')
    assert "environment variables" in error.message
    assert "sk-super-secret-value" not in error.message
    assert "sk-super-secret-value" not in str(error.to_dict())


def test_error_messages_never_echo_values(tmp_path: Path) -> None:
    error = _fails(tmp_path, 'profile = "very-private-value"')
    assert "very-private-value" not in error.message


def test_malformed_toml_is_a_config_error(tmp_path: Path) -> None:
    error = _fails(tmp_path, "this is = = not toml")
    assert "TOML" in error.message


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        load_settings(tmp_path / "absent.toml")
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_invalid_environment_value_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "LOUD")
    with pytest.raises(RagError) as info:
        load_settings()
    assert "logging.level" in info.value.message


def test_dotenv_files_are_not_a_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("RAG_LOGGING__LEVEL=DEBUG\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_settings().logging.level == "INFO"


def test_read_secret_comes_from_the_environment_only() -> None:
    assert read_secret("MY_KEY", {"MY_KEY": "value"}) == "value"
    for environ in ({}, {"MY_KEY": ""}):
        with pytest.raises(RagError) as info:
            read_secret("MY_KEY", environ)
        assert info.value.code is ErrorCode.E_CONFIG_INVALID
        assert "MY_KEY" in info.value.message


def test_profiles_are_consistent() -> None:
    assert set(PROFILE_LIMITS) == set(ResourceProfile) == {ResourceProfile.LOW}
    assert Settings().profile is ResourceProfile.LOW


# ------------------------------------------------------- parsing and chunking


def test_parsing_and_chunking_defaults() -> None:
    settings = load_settings()
    assert settings.parsing.max_text_chars == 20_000_000
    assert settings.parsing.isolate == ("pdf", "docx", "html")
    assert settings.parsing.timeout_s == 60
    assert (settings.chunking.target_tokens, settings.chunking.max_tokens) == (256, 384)
    assert settings.chunking.min_tokens == 48
    assert settings.chunking.overlap_ratio == 0.12
    assert settings.chunking.counter == "heuristic" and settings.chunking.strategy == "auto"


def test_parsing_and_chunking_can_be_configured(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[parsing]\nmax_pages = 50\nisolate = ["pdf"]\n\n'
        '[chunking]\ntarget_tokens = 192\nmax_tokens = 256\nstrategy = "recursive"\n'
        'counter = "tokenizers"\n'
        'tokenizer_json = "models/tokenizer.json"\n',
    )
    settings = load_settings(path)
    assert settings.parsing.max_pages == 50 and settings.parsing.isolate == ("pdf",)
    assert settings.chunking.target_tokens == 192 and settings.chunking.strategy == "recursive"
    assert settings.chunking.tokenizer_json == Path("models/tokenizer.json")


def test_environment_overrides_parsing_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_PARSING__TIMEOUT_S", "5")
    monkeypatch.setenv("RAG_CHUNKING__OVERLAP_RATIO", "0.2")
    settings = load_settings()
    assert settings.parsing.timeout_s == 5 and settings.chunking.overlap_ratio == 0.2


@pytest.mark.parametrize(
    "body",
    [
        "[chunking]\nmin_tokens = 300\n",  # min above target
        "[chunking]\ntarget_tokens = 400\n",  # target above max
        '[chunking]\ncounter = "tokenizers"\n',  # needs a tokenizer file
        '[chunking]\ncounter = "magic"\n',
        '[chunking]\nstrategy = "semantic"\n',
        "[chunking]\noverlap_ratio = 0.5\n",
        "[parsing]\ntimeout_s = 0\n",
        "[parsing]\nmax_pages = 0\n",
        '[parsing]\nisolate = ["pdf", "exe"]\n',
        "[parsing]\nunknown_limit = 1\n",
    ],
)
def test_invalid_parsing_and_chunking_settings_are_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(RagError) as info:
        load_settings(write(tmp_path, body))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    fields = str(info.value.context["fields"])
    assert "parsing" in fields or "chunking" in fields


def test_parse_limits_validate_and_default_to_the_settings_defaults() -> None:
    from rag_engine.ports import ParseHints, ParseLimits

    limits = ParseLimits()
    settings = load_settings().parsing
    assert limits.max_text_chars == settings.max_text_chars
    assert limits.max_pages == settings.max_pages
    assert limits.timeout_s == settings.timeout_s
    assert ParseHints().limits == limits and ParseHints().encoding is None
    for bad in ({"max_blocks": 0}, {"timeout_s": 0}, {"max_nesting": -1}):
        with pytest.raises(RagError):
            ParseLimits(**bad)
