from __future__ import annotations

from clients.chat_llm_client import ChatLLMClient
from generators.llm_generator import LLMGenerator
from pipelines.base import MockGenerator
from runtime_config import RuntimeSettings


def build_generator(settings: RuntimeSettings) -> MockGenerator | LLMGenerator:
    backend = settings.generator.backend.strip().lower()
    if backend in {"", "mock"}:
        return MockGenerator()
    if backend not in {"llm", "openai"}:
        raise ValueError(f"Unsupported generator backend: {settings.generator.backend}")
    client = ChatLLMClient(
        api_key=settings.generator.api_key,
        base_url=settings.generator.base_url,
        model=settings.generator.model,
        timeout=settings.generator.timeout,
        max_tokens=settings.generator.max_tokens,
        temperature=settings.generator.temperature,
        json_mode=settings.generator.json_mode,
    )
    return LLMGenerator(client=client, fallback_generator=MockGenerator())
