from app.config import Settings, get_settings


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("MAX_MESSAGES", raising=False)
    monkeypatch.delenv("MAX_MESSAGE_CHARS", raising=False)
    monkeypatch.delenv("MAX_TOTAL_CHARS", raising=False)
    monkeypatch.delenv("MAX_TOKENS_LIMIT", raising=False)
    monkeypatch.delenv("MAX_N", raising=False)
    monkeypatch.delenv("TEMPERATURE_MIN", raising=False)
    monkeypatch.delenv("TEMPERATURE_MAX", raising=False)
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LOG_FULL_PROMPTS", raising=False)

    settings = Settings()

    assert settings.vllm_base_url == "http://127.0.0.1:8000/v1"
    assert settings.gateway_api_key is None
    assert settings.max_messages == 20
    assert settings.max_message_chars == 8000
    assert settings.max_total_chars == 16000
    assert settings.max_tokens_limit == 1024
    assert settings.max_n == 1
    assert settings.temperature_min == 0.0
    assert settings.temperature_max == 2.0
    assert settings.request_timeout_seconds == 120
    assert settings.log_full_prompts is False


def test_get_settings_cached() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second
