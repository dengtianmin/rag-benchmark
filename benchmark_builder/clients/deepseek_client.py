from __future__ import annotations

from typing import Any
import logging

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from benchmark_builder.config import Settings

logger = logging.getLogger(__name__)


class DeepSeekClientError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.api_key = settings.deepseek_api_key
        self.model = settings.deepseek_model

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not self.settings.dry_run

    def _build_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/chat/completions"

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise DeepSeekClientError("DEEPSEEK_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm.temperature if temperature is None else temperature,
            "max_tokens": self.settings.llm.max_tokens if max_tokens is None else max_tokens,
        }
        use_json_mode = self.settings.llm.json_mode if json_mode is None else json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self._build_url()

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self.settings.retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self.settings.retry_backoff_min,
                max=self.settings.retry_backoff_max,
            ),
            retry=retry_if_exception_type((requests.RequestException, DeepSeekClientError)),
        )

        for attempt in retrying:
            with attempt:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                logger.info("DeepSeek raw response: %s", data)
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise DeepSeekClientError(f"Unexpected DeepSeek response: {data}") from exc
                return {
                    "raw": data,
                    "content": content,
                }

        raise DeepSeekClientError("DeepSeek request failed after retries.")
