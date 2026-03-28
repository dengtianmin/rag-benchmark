from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


class ChatLLMClientError(RuntimeError):
    """Raised when the chat completion request fails or returns invalid data."""


@dataclass(slots=True)
class ChatLLMResponse:
    content: str
    model_name: str
    finish_reason: str | None
    raw: dict[str, Any]
    latency_ms: int


class ChatLLMClient:
    """Minimal OpenAI-style chat completion client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_mode: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GENERATOR_API_KEY")
        self.base_url = (base_url if base_url is not None else os.getenv("GENERATOR_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model if model is not None else os.getenv("GENERATOR_MODEL", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.session = session or requests.Session()

        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def _build_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ChatLLMClientError("GENERATOR_API_KEY is not configured.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> ChatLLMResponse:
        if not self.model:
            raise ChatLLMClientError("GENERATOR_MODEL is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.json_mode if json_mode is None else json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self.session.post(
                self._build_url(),
                headers=self._build_headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ChatLLMClientError(f"LLM request timed out after {self.timeout}s.") from exc
        except requests.RequestException as exc:
            raise ChatLLMClientError(f"LLM request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ChatLLMClientError("LLM response is not valid JSON.") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatLLMClientError(f"Unexpected LLM response structure: {data}") from exc

        if not isinstance(content, str) or not content.strip():
            raise ChatLLMClientError("LLM returned empty content.")

        latency_seconds = response.elapsed.total_seconds() if response.elapsed is not None else 0.0
        return ChatLLMResponse(
            content=content,
            model_name=str(data.get("model") or self.model),
            finish_reason=choice.get("finish_reason"),
            raw=data,
            latency_ms=int(latency_seconds * 1000),
        )
