from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential


REQUEST_TIMEOUT_ERRORS = (requests.exceptions.Timeout,)
REQUEST_RETRYABLE_CONNECTION_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ProxyError,
)
REQUEST_EXCEPTION_ERRORS = (requests.exceptions.RequestException,)
REQUEST_HTTP_ERROR = requests.exceptions.HTTPError


class ChatLLMClientError(RuntimeError):
    """Raised when the chat completion request fails or returns invalid data."""


class ChatLLMRetryableError(ChatLLMClientError):
    """Raised for transient LLM failures that should be retried."""


@dataclass(slots=True)
class ChatLLMResponse:
    content: str
    model_name: str
    finish_reason: str | None
    raw: dict[str, Any]
    latency_ms: int


LOGGER = logging.getLogger(__name__)


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
        max_retries: int = 5,
        retry_min_seconds: float = 2.0,
        retry_max_seconds: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GENERATOR_API_KEY")
        self.base_url = (base_url if base_url is not None else os.getenv("GENERATOR_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model if model is not None else os.getenv("GENERATOR_MODEL", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.retry_min_seconds = retry_min_seconds
        self.retry_max_seconds = retry_max_seconds
        self.logger = logger or LOGGER

        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive.")
        if self.retry_min_seconds <= 0 or self.retry_max_seconds <= 0:
            raise ValueError("retry wait bounds must be positive.")
        if self.retry_min_seconds > self.retry_max_seconds:
            raise ValueError("retry_min_seconds must be <= retry_max_seconds.")

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

    def _request_with_retries(self, *, payload: dict[str, Any]) -> requests.Response:
        last_exception: Exception | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=self.retry_min_seconds, max=self.retry_max_seconds),
            retry=retry_if_exception_type(ChatLLMRetryableError),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                try:
                    response = self.session.post(
                        self._build_url(),
                        headers=self._build_headers(),
                        json=payload,
                        timeout=self.timeout,
                    )
                    self._raise_for_status(response)
                    return response
                except REQUEST_TIMEOUT_ERRORS as exc:
                    last_exception = exc
                    self.logger.warning(
                        "LLM request timed out on attempt %s/%s: %s",
                        attempt.retry_state.attempt_number,
                        self.max_retries,
                        exc,
                    )
                    raise ChatLLMRetryableError(f"LLM request timed out after {self.timeout}s.") from exc
                except REQUEST_RETRYABLE_CONNECTION_ERRORS as exc:
                    last_exception = exc
                    self.logger.warning(
                        "LLM request connection failed on attempt %s/%s: %s",
                        attempt.retry_state.attempt_number,
                        self.max_retries,
                        exc,
                    )
                    raise ChatLLMRetryableError(f"LLM transient connection failure: {exc}") from exc
                except REQUEST_EXCEPTION_ERRORS as exc:
                    last_exception = exc
                    raise ChatLLMClientError(f"LLM request failed: {exc}") from exc
        if last_exception is not None:
            raise ChatLLMClientError(f"LLM request failed after {self.max_retries} attempts: {last_exception}") from last_exception
        raise ChatLLMClientError("LLM request failed without a captured exception.")

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except REQUEST_HTTP_ERROR as exc:
            status_code = response.status_code
            if status_code == 429 or 500 <= status_code < 600:
                self.logger.warning(
                    "LLM server returned retryable status %s on %s.",
                    status_code,
                    self._build_url(),
                )
                raise ChatLLMRetryableError(f"LLM server returned retryable status {status_code}.") from exc
            raise

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
            response = self._request_with_retries(payload=payload)
        except ChatLLMRetryableError as exc:
            raise ChatLLMClientError(f"LLM request failed after {self.max_retries} attempts: {exc}") from exc

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
