from __future__ import annotations

from datetime import timedelta

import pytest
import requests

from clients.chat_llm_client import ChatLLMClient, ChatLLMClientError


class _DummyResponse:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.elapsed = timedelta(milliseconds=12)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict:
        return self._payload


class _TransientProxySession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        self.calls += 1
        if self.calls < 3:
            raise requests.exceptions.ProxyError("proxy disconnected")
        return _DummyResponse(
            {
                "model": "stub-model",
                "choices": [
                    {
                        "message": {"content": '{"query":"alpha"}'},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


class _BadRequestSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        self.calls += 1
        return _DummyResponse({"error": "bad request"}, status_code=400)


def test_chat_completion_retries_transient_proxy_errors() -> None:
    session = _TransientProxySession()
    client = ChatLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
        session=session,  # type: ignore[arg-type]
        max_retries=3,
        retry_min_seconds=0.01,
        retry_max_seconds=0.02,
    )

    response = client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert response.content == '{"query":"alpha"}'
    assert session.calls == 3


def test_chat_completion_does_not_retry_non_retryable_http_errors() -> None:
    session = _BadRequestSession()
    client = ChatLLMClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test-model",
        session=session,  # type: ignore[arg-type]
        max_retries=3,
        retry_min_seconds=0.01,
        retry_max_seconds=0.02,
    )

    with pytest.raises(ChatLLMClientError, match="status=400"):
        client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert session.calls == 1
