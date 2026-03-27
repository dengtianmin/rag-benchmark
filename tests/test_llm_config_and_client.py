from __future__ import annotations

from typing import Any

import requests

from benchmark_builder.clients.deepseek_client import LLMClient, LLMClientError
import benchmark_builder.config as config_module
from benchmark_builder.config import Settings, load_settings


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_load_settings_prefers_generic_llm_env_vars(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_STYLE", "openai")
    monkeypatch.setenv("LLM_API_KEY", "local-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Llama-3.1-8B")
    monkeypatch.setenv("LLM_DISABLE_AUTH", "false")

    settings = load_settings()

    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_api_style == "openai"
    assert settings.llm_api_key == "local-key"
    assert settings.llm_base_url == "http://127.0.0.1:8000/v1"
    assert settings.llm_model == "Llama-3.1-8B"
    assert settings.deepseek_api_key == "deepseek-key"


def test_load_settings_falls_back_to_deepseek_env_vars(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    settings = load_settings()

    assert settings.llm_api_key == "deepseek-key"
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_disable_auth is False


def test_llm_client_builds_openai_compatible_url_and_headers(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "local-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Llama-3.1-8B")
    settings = load_settings()
    client = LLMClient(settings)

    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse({"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    monkeypatch.setattr(requests, "post", fake_post)

    response = client.chat_completion(system_prompt="system", user_prompt="user")

    assert response["content"] == "{\"ok\": true}"
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer local-key"
    assert captured["json"]["model"] == "Llama-3.1-8B"
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_llm_client_can_disable_auth(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_DISABLE_AUTH", "true")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Llama-3.1-8B")
    settings = load_settings()
    client = LLMClient(settings)

    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _DummyResponse:
        captured["headers"] = headers
        return _DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(requests, "post", fake_post)

    response = client.chat_completion(system_prompt="system", user_prompt="user", json_mode=False)

    assert response["content"] == "ok"
    assert "Authorization" not in captured["headers"]


def test_llm_client_requires_auth_when_not_disabled(monkeypatch) -> None:
    settings = Settings(
        llm_api_key=None,
        llm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Llama-3.1-8B",
        llm_disable_auth=False,
    )
    client = LLMClient(settings)

    try:
        client.chat_completion(system_prompt="system", user_prompt="user")
    except LLMClientError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("Expected LLMClientError when auth is missing.")
